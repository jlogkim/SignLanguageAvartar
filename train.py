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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "2"  # Set the GPU 2 to use
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

from custom_dataset import *
from custom_train_utils import *
from custom_vis import *

from constants import *

if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)
    
cfg = load_config(os.path.join(CONF_DIR,'Base.yaml'))

cfg["training"]["use_cuda"] = True
cfg["data"]["max_sent_length"] = 1
cfg["data"]["skip_frames"] = 2
cfg["model"]["trg_size"] = lookup_trg_size
cfg["training"]["logging_freq"] = 40
cfg["training"]["validation_freq"] = 2000
cfg['training']["max_output_length"] = 20
cfg["training"]["batch_size"] = 16
cfg["model"]["encoder"]["num_layers"] = 1
cfg["training"]["epochs"] = 1000
cfg["model"]["gaussian_noise"] = True
cfg["model"]["future_prediction"] = 5

tok_fun = lambda s: list(s) if level == "char" else s.split()
tokenizer = get_tokenizer("basic_english")

train_data, dev_data, src_vocab, trg_vocab = load_data(cfg, tokenizer)

model = build_model(cfg, src_vocab=src_vocab, trg_vocab=trg_vocab)
trainer = TrainManager(model=model, config=cfg)
trainer.logger.info(model)
trainer.train_and_validate(train_data=train_data, valid_data=dev_data)