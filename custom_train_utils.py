import os
import pickle
import sys

import yaml
import numpy as np
import random
import sys
from typing import Optional
import queue
import glob
import time
from logging import Logger
import logging

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchtext.legacy.data import Dataset, Example, Field, BucketIterator, Iterator

sys.path.append('./ProgressiveTransformersSLP/')

from helpers import *
from vocabulary import build_vocab
from initialization import initialize_model
from transformer_layers import TransformerEncoderLayer, PositionalEncoding, TransformerDecoderLayer
from vocabulary import Vocabulary
from batch import Batch
from loss import RegLoss
from builders import build_optimizer, build_scheduler, build_gradient_clipper
from dtw import dtw
from plot_videos import plot_video, alter_DTW_timing

from torchtext.data.utils import get_tokenizer

from custom_dataset import *
from custom_train_utils import *
from custom_vis import *

from constants import *


def make_data_iter(dataset: Dataset,
                   batch_size: int,
                   train: bool = False,
                   shuffle: bool = False) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """
    if train:
        # optionally shuffle and sort during training
        data_iter = BucketIterator(
            repeat=False, sort=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=None,
            train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=shuffle)
    else:
        # don't sort/shuffle for validation/inference
        data_iter = BucketIterator(
            repeat=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=None,
            train=False, sort=False)

    return data_iter

class Embeddings(nn.Module):

    """
    Simple embeddings class
    """

    # pylint: disable=unused-argument
    def __init__(self,
                 embedding_dim: int = 64,
                 scale: bool = False,
                 vocab_size: int = 0,
                 padding_idx: int = 1,
                 freeze: bool = False,
                 **kwargs):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim:
        :param scale:
        :param vocab_size:
        :param padding_idx:
        :param freeze: freeze the embeddings during training
        """
        super(Embeddings, self).__init__()

        self.embedding_dim = embedding_dim
        self.scale = scale
        self.vocab_size = vocab_size
        self.lut = nn.Embedding(vocab_size, self.embedding_dim,
                                padding_idx=padding_idx)

        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor) -> Tensor:
        """
        Perform lookup for input `x` in the embedding table.

        :param x: index in the vocabulary
        :return: embedded representation for `x`
        """
        if self.scale:
            return self.lut(x) * math.sqrt(self.embedding_dim)
        return self.lut(x)

    def __repr__(self):
        return "%s(embedding_dim=%d, vocab_size=%d)" % (
            self.__class__.__name__, self.embedding_dim, self.vocab_size)
        
class Decoder(nn.Module):
    """
    Base decoder class
    """

    @property
    def output_size(self):
        """
        Return the output size (size of the target vocabulary)

        :return:
        """
        return self._output_size

class TransformerDecoder(Decoder):
    """
    A transformer decoder with N masked layers.
    Decoder layers are masked so that an attention head cannot see the future.
    """

    def __init__(self,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 vocab_size: int = 1,
                 freeze: bool = False,
                 trg_size: int = 97,
                 decoder_trg_trg_: bool = True,
                 **kwargs):
        """
        Initialize a Transformer decoder.

        :param num_layers: number of Transformer layers
        :param num_heads: number of heads for each layer
        :param hidden_size: hidden size
        :param ff_size: position-wise feed-forward size
        :param dropout: dropout probability (1-keep)
        :param emb_dropout: dropout probability for embeddings
        :param vocab_size: size of the output vocabulary
        :param freeze: set to True keep all decoder parameters fixed
        :param kwargs:
        """
        super(TransformerDecoder, self).__init__()

        self._hidden_size = hidden_size

        # Dynamic output size depending on the target size
        self._output_size = trg_size

        # create num_layers decoder layers and put them in a list
        self.layers = nn.ModuleList([TransformerDecoderLayer(
                size=hidden_size, ff_size=ff_size, num_heads=num_heads,
                dropout=dropout, decoder_trg_trg=decoder_trg_trg_) for _ in range(num_layers)])

        self.pe = PositionalEncoding(hidden_size,mask_count=True)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.emb_dropout = nn.Dropout(p=emb_dropout)

        # Output layer to be the size of joints vector + 1 for counter (total is trg_size)
        self.output_layer = nn.Linear(hidden_size, trg_size, bias=False)

        if freeze:
            freeze_params(self)

    def forward(self,
                trg_embed: Tensor = None,
                encoder_output: Tensor = None,
                src_mask: Tensor = None,
                trg_mask: Tensor = None,
                **kwargs):
        """
        Transformer decoder forward pass.

        :param trg_embed: embedded targets
        :param encoder_output: source representations
        :param encoder_hidden: unused
        :param src_mask:
        :param unroll_steps: unused
        :param hidden: unused
        :param trg_mask: to mask out target paddings
                         Note that a subsequent mask is applied here.
        :param kwargs:
        :return:
        """
        assert trg_mask is not None, "trg_mask required for Transformer"

        # add position encoding to word embedding
        x = self.pe(trg_embed)
        # Dropout if given
        x = self.emb_dropout(x)

        padding_mask = trg_mask
        # Create subsequent mask for decoding
        sub_mask = subsequent_mask(
            trg_embed.size(1)).type_as(trg_mask)

        # Apply each layer to the input
        for layer in self.layers:
            x = layer(x=x, memory=encoder_output,
                      src_mask=src_mask, trg_mask=sub_mask, padding_mask=padding_mask)

        # Apply a layer normalisation
        x = self.layer_norm(x)
        # Output layer turns it back into vectors of size trg_size
        output = self.output_layer(x)

        return output, x, None, None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].trg_trg_att.num_heads)

class Encoder(nn.Module):
    """
    Base encoder class
    """
    @property
    def output_size(self):
        """
        Return the output size

        :return:
        """
        return self._output_size

class TransformerEncoder(Encoder):
    """
    Transformer Encoder
    """

    #pylint: disable=unused-argument
    def __init__(self,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 num_layers: int = 8,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 freeze: bool = False,
                 **kwargs):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super(TransformerEncoder, self).__init__()

        # build all (num_layers) layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(size=hidden_size, ff_size=ff_size,
                                    num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self._output_size = hidden_size

        if freeze:
            freeze_params(self)

    #pylint: disable=arguments-differ
    def forward(self,
                embed_src: Tensor,
                src_length: Tensor,
                mask: Tensor) -> (Tensor, Tensor):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """

        x = embed_src
        # Add position encoding to word embeddings
        x = self.pe(x)
        # Add Dropout
        x = self.emb_dropout(x)

        # Apply each layer to the input
        for layer in self.layers:
            x = layer(x, mask)

        return self.layer_norm(x), None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].src_src_att.num_heads)


class RegLoss(nn.Module):
    """
    Regression Loss
    """

    def __init__(self, cfg, target_pad=0.0):
        super(RegLoss, self).__init__()

        self.loss = cfg["training"]["loss"].lower()

        if self.loss == "l1":
            self.criterion = nn.L1Loss()
        elif self.loss == "mse":
            self.criterion = nn.MSELoss()

        else:
            print("Loss not found - revert to default L1 loss")
            self.criterion = nn.L1Loss()

        model_cfg = cfg["model"]

        self.target_pad = target_pad
        self.loss_scale = model_cfg.get("loss_scale", 1.0)

    # pylint: disable=arguments-differ
    def forward(self, preds, targets):

        loss_mask = (targets != self.target_pad)
       
        # Find the masked predictions and targets using loss mask
        preds_masked = preds * loss_mask
        targets_masked = targets * loss_mask

        # Calculate loss just over the masked predictions
        loss = self.criterion(preds_masked, targets_masked)

        # Multiply loss by the loss scale
        if self.loss_scale != 1.0:
            loss = loss * self.loss_scale

        return loss

class Model(nn.Module):
    """
    Base Model class
    """

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: Embeddings,
                 trg_embed: Embeddings,
                 src_vocab: Vocabulary,
                 trg_vocab: Vocabulary,
                 cfg: dict,
                 in_trg_size: int,
                 out_trg_size: int,
                 ) -> None:
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        super(Model, self).__init__()

        model_cfg = cfg["model"]

        self.src_embed = src_embed
        self.trg_embed = trg_embed

        self.encoder = encoder
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = self.src_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.src_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.src_vocab.stoi[EOS_TOKEN]
        self.target_pad = TARGET_PAD

        self.use_cuda = cfg["training"]["use_cuda"]

        self.in_trg_size = in_trg_size
        self.out_trg_size = out_trg_size
        self.count_in = model_cfg.get("count_in",True)
        # Just Counter
        self.just_count_in = model_cfg.get("just_count_in",False)
        # Gaussian Noise
        self.gaussian_noise = model_cfg.get("gaussian_noise",False)
        # Gaussian Noise
        if self.gaussian_noise:
            self.noise_rate = model_cfg.get("noise_rate", 1.0)

        # Future Prediction - predict for this many frames in the future
        self.future_prediction = model_cfg.get("future_prediction", 0)

    # pylint: disable=arguments-differ
    def forward(self,
                src: Tensor,
                trg_input: Tensor,
                src_mask: Tensor,
                src_lengths: Tensor,
                trg_mask: Tensor = None) -> (
        Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param src_lengths: length of source inputs
        :param trg_mask: target mask
        :return: decoder outputs
        """

        # Encode the source sequence
        encoder_output, encoder_hidden = self.encode(src=src,
                                                     src_length=src_lengths,
                                                     src_mask=src_mask)
        unroll_steps = trg_input.size(1)

        # Add gaussian noise to the target inputs, if in training
        if (self.gaussian_noise) and (self.training) and (self.out_stds is not None):

            # Create a normal distribution of random numbers between 0-1
            noise = trg_input.data.new(trg_input.size()).normal_(0, 1)
            # Zero out the noise over the counter
            noise[:,:,-1] = torch.zeros_like(noise[:, :, -1])

            # Need to add a zero on the end of
            if self.future_prediction != 0:
                self.out_stds = torch.cat((self.out_stds,torch.zeros_like(self.out_stds)))[:trg_input.shape[-1]]

            # Need to multiply by the standard deviations
            noise = noise * self.out_stds

            # Add to trg_input multiplied by the noise rate
            trg_input = trg_input + self.noise_rate*noise

        # Decode the target sequence
        skel_out, dec_hidden, _, _ = self.decode(encoder_output=encoder_output,
                                                 src_mask=src_mask, trg_input=trg_input,
                                                 trg_mask=trg_mask)

        gloss_out = None

        return skel_out, gloss_out

    def encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor) \
        -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        # Encode an embedded source
        encode_output = self.encoder(self.src_embed(src), src_length, src_mask)

        return encode_output


    def decode(self, encoder_output: Tensor,
               src_mask: Tensor, trg_input: Tensor,
               trg_mask: Tensor = None) \
        -> (Tensor, Tensor, Tensor, Tensor):

        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """

        # Enbed the target using a linear layer
        trg_embed = self.trg_embed(trg_input)
        # Apply decoder to the embedded target
        decoder_output = self.decoder(trg_embed=trg_embed, encoder_output=encoder_output,
                               src_mask=src_mask,trg_mask=trg_mask)

        return decoder_output

    def get_loss_for_batch(self, batch: Batch, loss_function: nn.Module):
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param loss_function: loss function, computes for input and target
            a scalar loss for the complete batch
        :return: batch_loss: sum of losses over non-pad elements in the batch
        """
        # Forward through the batch input
        skel_out, _ = self.forward(
            src=batch.src, trg_input=batch.trg_input,
            src_mask=batch.src_mask, src_lengths=batch.src_lengths,
            trg_mask=batch.trg_mask)

        # compute batch loss using skel_out and the batch target
        # do it by weight
        # 25/70/21/21
        torso_loss = loss_function(skel_out[:, :, 0:50], batch.trg[:, :, 0:50])
        face_loss = loss_function(skel_out[:, :, 50:50+140], batch.trg[:, :, 50:50+140])
        hand_loss = loss_function(skel_out[:, :, 50+140:], batch.trg[:, :, 50+140:])
        
        #batch_loss = loss_function(skel_out, batch.trg)
        batch_loss = torso_loss * 0.2 + hand_loss  + face_loss * 0.1

        # If gaussian noise, find the noise for the next epoch
        if self.gaussian_noise:
            # Calculate the difference between prediction and GT, to find STDs of error
            with torch.no_grad():
                noise = skel_out.detach() - batch.trg.detach()

            if self.future_prediction != 0:
                # Cut to only the first frame prediction + add the counter
                noise = noise[:, :, :noise.shape[2] // (self.future_prediction)]

        else:
            noise = None

        dtw = 0 #calculate_dtw(batch.trg, skel_out)
        # return batch loss = sum over all elements in batch that are not pad
        return batch_loss, torso_loss.item(), hand_loss.item(), face_loss.item(), noise

    def run_batch(self, batch: Batch, max_output_length: int,) -> (np.array, np.array):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param max_output_length: maximum length of hypotheses
        :param beam_size: size of the beam for beam search, if 0 use greedy
        :param beam_alpha: alpha value for beam search
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """
        # First encode the batch, as this can be done in all one go
        encoder_output, encoder_hidden = self.encode(
            batch.src, batch.src_lengths,
            batch.src_mask)

        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(batch.src_lengths.cpu().numpy()) * 1.5)

        # Then decode the batch separately, as needs to be done iteratively
        # greedy decoding
        stacked_output, stacked_attention_scores = greedy(
                encoder_output=encoder_output,
                src_mask=batch.src_mask,
                embed=self.trg_embed,
                decoder=self.decoder,
                trg_input=batch.trg_input,
                model=self)

        return stacked_output, stacked_attention_scores

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return "%s(\n" \
               "\tencoder=%s,\n" \
               "\tdecoder=%s,\n" \
               "\tsrc_embed=%s,\n" \
               "\ttrg_embed=%s)" % (self.__class__.__name__, self.encoder,
                   self.decoder, self.src_embed, self.trg_embed)
               
def build_model(cfg: dict = None,
                src_vocab = None,
                trg_vocab = None) -> Model:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """

    full_cfg = cfg
    cfg = cfg["model"]

    src_padding_idx = None
    trg_padding_idx = 0

    # Input target size is the joint vector length plus one for counter
    in_trg_size = cfg["trg_size"] + 1
    # Output target size is the joint vector length plus one for counter
    out_trg_size = cfg["trg_size"] + 1

    just_count_in = cfg.get("just_count_in", False)
    future_prediction = cfg.get("future_prediction", 0)

    #  Just count in limits the in target size to 1
    if just_count_in:
        in_trg_size = 1

    # Future Prediction increases the output target size
    if future_prediction != 0:
        # Times the trg_size (minus counter) by amount of predicted frames, and then add back counter
        out_trg_size = (out_trg_size - 1 ) * future_prediction + 1

    # Define source embedding
    src_embed = Embeddings(
        **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
        padding_idx=src_padding_idx)

    # Define target linear
    # Linear layer replaces an embedding layer - as this takes in the joints size as opposed to a token
    trg_linear = nn.Linear(in_trg_size, cfg["decoder"]["embeddings"]["embedding_dim"])

    ## Encoder -------
    enc_dropout = cfg["encoder"].get("dropout", 0.) # Dropout
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
           cfg["encoder"]["hidden_size"], \
           "for transformer, emb_size must be hidden_size"

    # Transformer Encoder
    encoder = TransformerEncoder(**cfg["encoder"],
                                 emb_size=src_embed.embedding_dim,
                                 emb_dropout=enc_emb_dropout)

    ## Decoder -------
    dec_dropout = cfg["decoder"].get("dropout", 0.) # Dropout
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
    decoder_trg_trg = cfg["decoder"].get("decoder_trg_trg", True)
    # Transformer Decoder
    decoder = TransformerDecoder(
        **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
        emb_size=trg_linear.out_features, emb_dropout=dec_emb_dropout,
        trg_size=out_trg_size, decoder_trg_trg_=decoder_trg_trg)

    # Define the model
    model = Model(encoder=encoder,
                  decoder=decoder,
                  src_embed=src_embed,
                  trg_embed=trg_linear,
                  src_vocab=src_vocab,
                  trg_vocab=trg_vocab,
                  cfg=full_cfg,
                  in_trg_size=in_trg_size,
                  out_trg_size=out_trg_size)

    # Custom initialization of model parameters
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx)

    return model

class TrainManager:

    def __init__(self, model: Model, config: dict, test=False) -> None:

        train_config = config["training"]
        model_dir = os.path.join(MODEL_DIR)
        # If model continue, continues model from the latest checkpoint
        model_continue = train_config.get("continue", True)
        # If the directory has not been created, can't continue from anything
        if not os.path.isdir(model_dir):
            model_continue = False
        if test:
            model_continue = True

        # files for logging and storing
        self.model_dir = model_dir
        
        # Build validation files
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.logger = make_logger(model_dir=self.model_dir)
        self.logging_freq = train_config.get('logging_freq', 100)

        # model
        self.model = model
        self.pad_index = self.model.pad_index
        self.bos_index = self.model.bos_index
        self._log_parameters_list()
        self.target_pad = TARGET_PAD

        # New Regression loss - depending on config
        self.loss = RegLoss(cfg = config,
                            target_pad=self.target_pad)

        self.normalization = "batch"

        # optimization
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(config=train_config, parameters=model.parameters())

        # validation & early stopping
        self.validation_freq = train_config.get("validation_freq", 1000)
        self.ckpt_best_queue = queue.Queue(maxsize=train_config.get("keep_last_ckpts", 1))
        self.ckpt_queue = queue.Queue(maxsize=1)




        self.val_on_train = config["data"].get("val_on_train", True)

        # TODO - Include Back Translation
        self.eval_metric = train_config.get("eval_metric", "dtw").lower()
        if self.eval_metric not in ['bleu', 'chrf', "dtw"]:
            raise ConfigurationError("Invalid setting for 'eval_metric', "
                                     "valid options: 'bleu', 'chrf', 'DTW'")
        self.early_stopping_metric = train_config.get("early_stopping_metric",
                                                       "eval_metric")

        # if we schedule after BLEU/chrf, we want to maximize it, else minimize
        # early_stopping_metric decides on how to find the early stopping point:
        # ckpts are written when there's a new high/low score for this metric
        if self.early_stopping_metric in ["loss","dtw"]:
            self.minimize_metric = True
        else:
            raise ConfigurationError("Invalid setting for 'early_stopping_metric', "
                                    "valid options: 'loss', 'dtw',.")

        # learning rate scheduling
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["encoder"]["hidden_size"])

        # data & batch handling
        self.level = "word"
        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config.get('epochs')
        self.batch_size = train_config["batch_size"]
        self.batch_type = "sentence"
        self.eval_batch_size = train_config.get("eval_batch_size",self.batch_size)
        self.eval_batch_type = train_config.get("eval_batch_type",self.batch_type)
        self.batch_multiplier = train_config.get("batch_multiplier", 1)

        # generation
        self.max_output_length = train_config.get("max_output_length", None)

        # CPU / GPU
        self.use_cuda = train_config["use_cuda"]
        if self.use_cuda:
            self.model.cuda()
            self.loss.cuda()

        # initialize training statistics
        self.steps = 0
        # stop training if this flag is True by reaching learning rate minimum
        self.stop = False
        self.total_tokens = 0
        self.best_ckpt_iteration = 0
        # initial values for best scores
        self.best_ckpt_score = np.inf if self.minimize_metric else -np.inf
        # comparison function for scores
        self.is_best = lambda score: score < self.best_ckpt_score \
            if self.minimize_metric else score > self.best_ckpt_score

        ## Checkpoint restart
        # If continuing
        if model_continue:
            # Get the latest checkpoint
            ckpt = get_latest_checkpoint(model_dir)
            if ckpt is None:
                self.logger.info(f"Can't find checkpoint in directory {ckpt}")
            else:
                self.logger.info(f"Continuing model from {ckpt}", )
                self.init_from_checkpoint(ckpt)

        # Skip frames
        self.skip_frames = config["data"].get("skip_frames", 1)

        ## -- Data augmentation --
        # Just Counter
        self.just_count_in = config["model"].get("just_count_in",False)
        # Gaussian Noise
        self.gaussian_noise = config["model"].get("gaussian_noise", False)
        
        if self.gaussian_noise:
            # How much the noise is added in
            self.noise_rate = config["model"].get("noise_rate", 1.0)

        if self.just_count_in and (self.gaussian_noise):
            raise ConfigurationError("Can't have both just_count_in and gaussian_noise as True")

        self.future_prediction = config["model"].get("future_prediction", 0)
        if self.future_prediction != 0:
            frames_predicted = [i for i in range(self.future_prediction)]
            print(f"Future prediction. Frames predicted: {frames_predicted}")

    # Save a checkpoint
    def _save_checkpoint(self, type="every") -> None:
        # Define model path
        model_path = "{}/{}_{}.ckpt".format(self.model_dir, self.steps, type)
        # Define State
        state = {
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "best_ckpt_score": self.best_ckpt_score,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if \
            self.scheduler is not None else None,
        }
        torch.save(state, model_path)
        print(f'saved checkpoint with path {model_path}')
        # If this is the best checkpoint
        if type == "best":
            if self.ckpt_best_queue.full():
                to_delete = self.ckpt_best_queue.get()  # delete oldest ckpt
                try:
                    os.remove(to_delete)
                except FileNotFoundError:
                    print(f"Wanted to delete old checkpoint {to_delete} but file does not exist.")
            self.ckpt_best_queue.put(model_path)

            best_path = "{}/best.ckpt".format(self.model_dir)
            torch.save(state, best_path)

        # If this is just the checkpoint at every validation
        elif type == "every":
            if self.ckpt_queue.full():
                to_delete = self.ckpt_queue.get()  # delete oldest ckpt
                try:
                    os.remove(to_delete)
                except FileNotFoundError:
                    print(f"Wanted to delete old checkpoint {to_delete} but file does not exist.")

            self.ckpt_queue.put(model_path)
            every_path = "{}/every.ckpt".format(self.model_dir)
            # overwrite every.ckpt
            torch.save(state, every_path)

    # Initialise from a checkpoint
    def init_from_checkpoint(self, path: str) -> None:
        # Find last checkpoint
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint["model_state"])
        self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])

        if model_checkpoint["scheduler_state"] is not None and \
                self.scheduler is not None:
            # Load the scheduler state
            self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])

        # restore counts
        self.steps = model_checkpoint["steps"]
        self.total_tokens = model_checkpoint["total_tokens"]
        self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
        self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]

        # move parameters to cuda
        if self.use_cuda:
            self.model.cuda()

    # Train and validate function
    def train_and_validate(self, train_data: Dataset, valid_data: Dataset) \
            -> None:
        # Make training iterator
        train_iter = make_data_iter(train_data,
                                    batch_size=self.batch_size,
                                    train=True, shuffle=self.shuffle)

        val_step = 0
        if self.gaussian_noise:
            all_epoch_noise = []
        # Loop through epochs
        epoch_start_time = time.time()
        for epoch_no in range(self.epochs):
            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)

            self.model.train()

            # Reset statistics for each epoch.
            start = time.time()
            total_valid_duration = 0
            start_tokens = self.total_tokens
            count = self.batch_multiplier - 1
            epoch_loss = 0
            epoch_torso_loss = 0
            epoch_hand_loss = 0
            epoch_face_loss = 0

            # If Gaussian Noise, extract STDs for each joint position
            if self.gaussian_noise:
                if len(all_epoch_noise) != 0:
                    self.model.out_stds = torch.mean(torch.stack(([noise.std(dim=[0]) for noise in all_epoch_noise])),dim=-2)
                else:
                    self.model.out_stds = None
                all_epoch_noise = []

            for batch in iter(train_iter):
                # reactivate training
                self.model.train()

                # create a Batch object from torchtext batch
                batch = Batch(torch_batch=batch,
                              pad_index=self.pad_index,
                              model=self.model)

                update = count == 0
                # Train the model on a batch
                batch_loss, torso_loss, hand_loss, face_loss, noise = self._train_batch(batch, update=update)
                # If Gaussian Noise, collect the noise
                if self.gaussian_noise:
                    # If future Prediction, cut down the noise size to just one frame
                    if self.future_prediction != 0:
                        all_epoch_noise.append(noise.reshape(-1, self.model.out_trg_size // self.future_prediction))
                    else:
                        all_epoch_noise.append(noise.reshape(-1,self.model.out_trg_size))

                count = self.batch_multiplier if update else count
                count -= 1
                epoch_loss += batch_loss.detach().cpu().numpy()
                epoch_torso_loss += torso_loss
                epoch_hand_loss += hand_loss
                epoch_face_loss += face_loss

                if self.scheduler is not None and self.scheduler_step_at == "step" and update:
                    self.scheduler.step()

                # log learning progress
                if self.steps % self.logging_freq == 0 and update:
                    elapsed = time.time() - start - total_valid_duration
                    elapsed_tokens = self.total_tokens - start_tokens
                    self.logger.info(
                        "Epoch %3d Step: %8d Batch Loss: %12.6f [Torso : %12.6f, Hand : %12.6f, Face : %12.6f]"
                        "Tokens per Sec: %8.0f, Lr: %.6f",
                        epoch_no + 1, self.steps, batch_loss, torso_loss, hand_loss, face_loss,
                        elapsed_tokens / elapsed,
                        self.optimizer.param_groups[0]["lr"])
                    start = time.time()
                    total_valid_duration = 0
                    start_tokens = self.total_tokens

                # validate on the entire dev set
                if self.steps % self.validation_freq == 0 and update:
                    self.logger.info("Starting validation calculation.")
                    valid_start_time = time.time()

                    valid_score, valid_loss, valid_references, valid_hypotheses, \
                        valid_inputs, all_dtw_scores, valid_file_paths = \
                        validate_on_data(
                            batch_size=self.eval_batch_size,
                            data=valid_data,
                            eval_metric=self.eval_metric,
                            model=self.model,
                            max_output_length=self.max_output_length,
                            loss_function=self.loss,
                            batch_type=self.eval_batch_type,
                            type="val",
                        )

                    val_step += 1

                    if self.early_stopping_metric == "loss":
                        ckpt_score = valid_loss
                    elif self.early_stopping_metric == "dtw":
                        ckpt_score = valid_score
                    else:
                        ckpt_score = valid_score

                    new_best = False
                    self.best = False
                    if self.is_best(ckpt_score):
                        self.best = True
                        self.best_ckpt_score = ckpt_score
                        self.best_ckpt_iteration = self.steps
                        self.logger.info(
                            'Hooray! New best validation result [%s]!',
                            self.early_stopping_metric)
                        if self.ckpt_queue.maxsize > 0:
                            self.logger.info("Saving new checkpoint.")
                            new_best = True
                            self._save_checkpoint(type="best")

                        '''
                            display = list(range(0, len(valid_hypotheses), int(np.ceil(len(valid_hypotheses) / 13.15))))
                            self.produce_validation_video(
                                output_joints=valid_hypotheses,
                                inputs=valid_inputs,
                                references=valid_references,
                                model_dir=self.model_dir,
                                steps=self.steps,
                                display=display,
                                type="val_inf",
                                file_paths=valid_file_paths,
                            )
                        '''

                    self._save_checkpoint(type="every")

                    if self.scheduler is not None and self.scheduler_step_at == "validation":
                        self.scheduler.step(ckpt_score)

                    # append to validation report
                    self._add_report(
                        valid_score=valid_score, valid_loss=valid_loss,
                        eval_metric=self.eval_metric,
                        new_best=new_best, report_type="val",)

                    valid_duration = time.time() - valid_start_time
                    total_valid_duration += valid_duration
                    self.logger.info(
                        'Validation result at epoch %3d, step %8d: Val DTW Score: %6.2f, '
                        'loss: %8.4f,  duration: %.4fs',
                            epoch_no+1, self.steps, valid_score,
                            valid_loss, valid_duration)
                elif self.steps % 300 == 0:
                    self._save_checkpoint(type="every")

                if self.stop:
                    break
            if self.stop:
                self.logger.info(
                    'Training ended since minimum lr %f was reached.',
                     self.learning_rate_min)
                break

            self.logger.info('Epoch %3d: total training loss %.5f [torso: %.5f, hand: %.5f, face: %.5f', epoch_no+1,
                             epoch_loss, epoch_torso_loss, epoch_hand_loss, epoch_face_loss)
        else:
            self.logger.info('Training ended after %3d epochs.', epoch_no+1)
        self.logger.info('Best validation result at step %8d: %6.2f %s.',
                         self.best_ckpt_iteration, self.best_ckpt_score,
                         self.early_stopping_metric)


    # Train the batch
    def _train_batch(self, batch: Batch, update: bool = True):

        # Get loss from this batch
        batch_loss, torso_loss, hand_loss, face_loss, noise = self.model.get_loss_for_batch(
            batch=batch, loss_function=self.loss)

        # normalize batch loss
        if self.normalization == "batch":
            normalizer = batch.nseqs
        elif self.normalization == "tokens":
            normalizer = batch.ntokens
        else:
            raise NotImplementedError("Only normalize by 'batch' or 'tokens'")

        norm_batch_loss = batch_loss / normalizer
        norm_torso_loss = torso_loss / normalizer
        norm_hand_loss = hand_loss / normalizer
        norm_face_loss = face_loss /normalizer
        # division needed since loss.backward sums the gradients until updated
        norm_batch_multiply = norm_batch_loss / self.batch_multiplier

        # compute gradients
        norm_batch_multiply.backward()

        if self.clip_grad_fun is not None:
            # clip gradients (in-place)
            self.clip_grad_fun(params=self.model.parameters())

        if update:
            # make gradient step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # increment step counter
            self.steps += 1

        # increment token counter
        self.total_tokens += batch.ntokens

        return norm_batch_loss, norm_torso_loss, norm_hand_loss, norm_face_loss, noise

    def _add_report(self, valid_score: float, valid_loss: float, eval_metric: str,
                    new_best: bool = False, report_type: str = "val") -> None:

        current_lr = -1
        # ignores other param groups for now
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        if current_lr < self.learning_rate_min:
            self.stop = True

        if report_type == "val":
            with open(self.valid_report_file, 'a') as opened_file:
                opened_file.write(
                    "Steps: {} Loss: {:.5f}| DTW: {:.3f}|"
                    " LR: {:.6f} {}\n".format(
                        self.steps, valid_loss, valid_score,
                        current_lr, "*" if new_best else ""))

    def _log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log.
        """
        model_parameters = filter(lambda p: p.requires_grad,
                                  self.model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info("Total params: %d", n_params)
        trainable_params = [n for (n, p) in self.model.named_parameters()
                            if p.requires_grad]
        self.logger.info("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params
        
    # Produce the video of Phoenix MTC joints
    def produce_validation_video(self,output_joints, inputs, references, display, model_dir, type, steps="", file_paths=None):

        # If not at test
        if type != "test":
            dir_name = model_dir + "/videos/Step_{}/".format(steps)
            if not os.path.exists(model_dir + "/videos/"):
                os.mkdir(model_dir + "/videos/")

        # If at test time
        elif type == "test":
            dir_name = model_dir + "/test_videos/"

        # Create model video folder if not exist
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        # For sequence to display
        for i in display:

            seq = output_joints[i]
            ref_seq = references[i]
            input = inputs[i]
            # Write gloss label
            gloss_label = input[0] # ["word"]


            # Alter the dtw timing of the produced sequence, and collect the DTW score
            timing_hyp_seq, ref_seq_count, dtw_score = alter_DTW_timing(seq, ref_seq)
            video_ext = "{}_{}.mp4".format(gloss_label, "{0:.2f}".format(float(dtw_score)).replace(".", "_"))

            try :
                if file_paths is not None:
                    sequence_ID = file_paths[i]
                else:
                    sequence_ID = None
            except:
                sequence_ID = None

            # Plot this sequences video
            if "<" not in video_ext:
                plot_video(joints=timing_hyp_seq,
                            file_path=dir_name,
                            video_name=video_ext,
                            references=ref_seq_count,
                            skip_frames=self.skip_frames,
                            sequence_ID=gloss_label)
                


                
def greedy(
        src_mask: Tensor,
        embed: Embeddings,
        decoder: Decoder,
        encoder_output: Tensor,
        trg_input: Tensor,
        model,
        ) -> (np.array, np.array):
    """
    Special greedy function for transformer, since it works differently.
    The transformer remembers all previous states and attends to them.

    :param src_mask: mask for source inputs, 0 for positions after </s>
    :param embed: target embedding
    :param bos_index: index of <s> in the vocabulary
    :param max_output_length: maximum length for the hypotheses
    :param decoder: decoder to use for greedy decoding
    :param encoder_output: encoder hidden states for attention
    :param encoder_hidden: encoder final state (unused in Transformer)
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    # Initialise the input
    # Extract just the BOS first frame from the target
    ys = trg_input[:,:1,:].float()

    # If the counter is coming into the decoder or not
    ys_out = ys

    # Set the target mask, by finding the padded rows
    trg_mask = trg_input != 0.0
    trg_mask = trg_mask.unsqueeze(1)

    # Find the maximum output length for this batch
    max_output_length = trg_input.shape[1]

    # If just count in, input is just the counter
    if model.just_count_in:
        ys = ys[:,:,-1:]

    for i in range(max_output_length):

        # ys here is the input
        # Drive the timing by giving the GT timing - add in the counter to the last column

        if model.just_count_in:
            # If just counter, drive the input using the GT counter
            ys[:,-1] = trg_input[:, i, -1:]

        else:
            # Give the GT counter for timing, to drive the timing
            ys[:,-1,-1:] = trg_input[:, i, -1:]

        # Embed the target input before passing to the decoder
        trg_embed = embed(ys)

        # Cut padding mask to required size (of the size of the input)
        padding_mask = trg_mask[:, :, :i+1, :i+1]
        # Pad the mask (If required) (To make it square, and used later on correctly)
        pad_amount = padding_mask.shape[2] - padding_mask.shape[3]
        padding_mask = (F.pad(input=padding_mask.double(), pad=(pad_amount, 0, 0, 0), mode='replicate') == 1.0)

        # Pass the embedded input and the encoder output into the decoder
        with torch.no_grad():
            out, _, _, _ = decoder(
                trg_embed=trg_embed,
                encoder_output=encoder_output,
                src_mask=src_mask,
                trg_mask=padding_mask,
            )

            if model.future_prediction != 0:
                # Cut to only the first frame prediction
                out = torch.cat((out[:, :, :out.shape[2] // (model.future_prediction)],out[:,:,-1:]),dim=2)

            if model.just_count_in:
                # If just counter in trg_input, concatenate counters of output
                ys = torch.cat([ys, out[:,-1:,-1:]], dim=1)

            # Add this frame prediction to the overall prediction
            ys = torch.cat([ys, out[:,-1:,:]], dim=1)

            # Add this next predicted frame to the full frame output
            ys_out = torch.cat([ys_out, out[:,-1:,:]], dim=1)

    return ys_out, None


# Find the best timing match between a reference and a hypothesis, using DTW
def calculate_dtw(references, hypotheses):
    """
    Calculate the DTW costs between a list of references and hypotheses

    :param references: list of reference sequences to compare against
    :param hypotheses: list of hypothesis sequences to fit onto the reference

    :return: dtw_scores: list of DTW costs
    """
    # Euclidean norm is the cost function, difference of coordinates
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))

    dtw_scores = []

    # Remove the BOS frame from the hypothesis
    hypotheses = hypotheses[:, 1:]

    # For each reference in the references list
    for i, ref in enumerate(references):
        # Cut the reference down to the max count value
        _ , ref_max_idx = torch.max(ref[:, -1], 0)
        if ref_max_idx == 0: ref_max_idx += 1
        # Cut down frames by to the max counter value, and chop off counter from joints
        ref_count = ref[:ref_max_idx,:-1].cpu().numpy()

        # Cut the hypothesis down to the max count value
        hyp = hypotheses[i]
        _, hyp_max_idx = torch.max(hyp[:, -1], 0)
        if hyp_max_idx == 0: hyp_max_idx += 1
        # Cut down frames by to the max counter value, and chop off counter from joints
        hyp_count = hyp[:hyp_max_idx,:-1].cpu().numpy()

        # Calculate DTW of the reference and hypothesis, using euclidean norm
        d, cost_matrix, acc_cost_matrix, path = dtw(ref_count, hyp_count, dist=euclidean_norm)

        # Normalise the dtw cost by sequence length
        d = d/acc_cost_matrix.shape[0]

        dtw_scores.append(d)

    # Return dtw scores and the hypothesis with altered timing
    return dtw_scores

# Validate epoch given a dataset
def validate_on_data(model: Model,
                     data: Dataset,
                     batch_size: int,
                     max_output_length: int,
                     eval_metric: str,
                     loss_function: torch.nn.Module = None,
                     batch_type: str = "sentence",
                     type = "val",
                     BT_model = None):

    valid_iter = make_data_iter(
        dataset=data, batch_size=batch_size,
        shuffle=True, train=False)

    pad_index = model.src_vocab.stoi[PAD_TOKEN]
    # disable dropout
    model.eval()
    # don't track gradients during validation
    with torch.no_grad():
        valid_hypotheses = []
        valid_references = []
        valid_inputs = []
        file_paths = []
        all_dtw_scores = []

        valid_loss = 0
        total_ntokens = 0
        total_nseqs = 0

        batches = 0
        for valid_batch in iter(valid_iter):
            # Extract batch
            batch = Batch(torch_batch=valid_batch,
                          pad_index = pad_index,
                          model = model)
            targets = batch.trg

            # run as during training with teacher forcing
            if loss_function is not None and batch.trg is not None:
                # Get the loss for this batch
                batch_loss, _, _, _, _, = model.get_loss_for_batch(
                    batch, loss_function=loss_function)

                valid_loss += batch_loss
                total_ntokens += batch.ntokens
                total_nseqs += batch.nseqs

            # If not just count in, run inference to produce translation videos
            if not model.just_count_in:
                # Run batch through the model in an auto-regressive format
                output, attention_scores = model.run_batch(
                                            batch=batch,
                                            max_output_length=max_output_length)

            # If future prediction
            if model.future_prediction != 0:
                # Cut to only the first frame prediction + add the counter
                # output = torch.cat((output[:, :, :output.shape[2] // (model.future_prediction)], output[:, :, -1:]),dim=2)
                # Cut to only the first frame prediction + add the counter
                targets = torch.cat((targets[:, :, :targets.shape[2] // (model.future_prediction)], targets[:, :, -1:]),dim=2)

            # For just counter, the inference is the same as GTing
            if model.just_count_in:
                output = train_output

            # Add references, hypotheses and file paths to list
            valid_references.extend(targets)
            valid_hypotheses.extend(output)
            #file_paths.extend(batch.file_paths)
            # Add the source sentences to list, by using the model source vocab and batch indices
            valid_inputs.extend([[model.src_vocab.itos[batch.src[i][j]] for j in range(len(batch.src[i]))] for i in
                                 range(len(batch.src))])

            # Calculate the full Dynamic Time Warping score - for evaluation
            dtw_score = calculate_dtw(targets, output)
            all_dtw_scores.extend(dtw_score)

            # Can set to only run a few batches
            # if batches == math.ceil(100/batch_size):
            #     break
            batches += 1

        # Dynamic Time Warping scores
        current_valid_score = np.mean(all_dtw_scores)

    return current_valid_score, valid_loss, valid_references, valid_hypotheses, \
           valid_inputs, all_dtw_scores, file_paths