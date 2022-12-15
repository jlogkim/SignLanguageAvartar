import os
import pickle
import sys

import yaml
import numpy as np
import random
import pickle
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

from constants import *

class SignProdDataset(Dataset):
    def __init__(self, 
                 fields,
                 path,
                 trg_size,
                 skip_frames=1,
                 **kwargs):
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]
        
        examples = []
        with open(path, 'rb') as data_file:
            dataset = pickle.load(data_file)
    
        for i, data in enumerate(dataset):
            kpts = data['ttl_kpts_seq']
            kpts_reshaped = np.squeeze(np.reshape(kpts,(kpts.shape[0],-1,1)), axis=2) # LX274(137x2)
            src_sentnce = data['sentence']
            
            # start_frame = np.zeros((1, kpts_reshaped.shape[-1]))
            # with_start_frame = np.concatenate((start_frame, kpts_reshaped)) 
            normalized = kpts_reshaped + 1e-8
            
            counters = np.arange(0,len(normalized),1)/len(normalized)
            with_counter = np.concatenate((normalized, counters[:, np.newaxis]), axis=1)
    
            examples.append(Example.fromlist([src_sentnce, with_counter], fields))
            super(SignProdDataset, self).__init__(examples, fields, **kwargs)
            

def load_data(cfg: dict, tokenizer):
    
    train_path = os.path.join(DATA_DIR, 'train.pkl')
    val_path = os.path.join(DATA_DIR, 'val.pkl')
    level = "word"
    data_cfg = cfg["data"]
    lowercase = False

    trg_size = cfg["model"]["trg_size"] + 1 # to account for counter
    skip_frames = data_cfg.get("skip_frames", 1)

    EOS_TOKEN = '</s>'

    src_field = Field(init_token=None,
                      pad_token=PAD_TOKEN, tokenize=tokenizer,
                      batch_first=True, lower=lowercase,
                      unk_token=UNK_TOKEN,
                      include_lengths=True)


    reg_trg_field = Field(sequential=True,
                          use_vocab=False,
                          dtype=torch.float32,
                          batch_first=True,
                          include_lengths=False,
                          pad_token=torch.ones((trg_size))*TARGET_PAD)
    # TARGET_PAD for padding number 

    train_data = SignProdDataset(fields=(src_field, reg_trg_field),
                                 path=train_path,
                                 trg_size=trg_size,
                                 skip_frames=skip_frames)

    src_vocab = build_vocab(field="src", min_freq=1,
                            max_size=sys.maxsize,
                            dataset=train_data, vocab_file=None)

    src_field.vocab = src_vocab
    trg_vocab = [None]*(trg_size)

    dev_data = SignProdDataset(fields=(src_field, reg_trg_field),
                                   path=val_path,
                                   trg_size=trg_size,
                                   skip_frames=skip_frames)
    
    return train_data, dev_data, src_vocab, trg_vocab