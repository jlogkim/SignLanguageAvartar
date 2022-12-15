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
import json

RES_DIR = './val_output'
MODEL_DIR = './model_20221215_17'

if not os.path.isdir(RES_DIR):
    os.mkdir(RES_DIR)

cfg = load_config(os.path.join(CONF_DIR, 'Base.yaml'))

cfg["training"]["use_cuda"] = True
cfg["data"]["max_sent_length"] = 1
cfg["data"]["skip_frames"] = 2
cfg["model"]["trg_size"] = lookup_trg_size
cfg["training"]["logging_freq"] = 40
cfg["training"]["validation_freq"] = 300
cfg['training']["max_output_length"] = 20
cfg["training"]["batch_size"] = 32
cfg["model"]["encoder"]["num_layers"] = 1
cfg["training"]["epochs"] = 1000
cfg["model"]["gaussian_noise"] = True
cfg["model"]["future_prediction"] = 5

def get_jsoned_pred(dt):
    dt_concated=np.concatenate([dt, np.ones((dt.shape[0],1))],axis=1)
    pose_keypoints_2d = dt_concated[:25].flatten()
    face_keypoints_2d = dt_concated[25:25+70].flatten()
    hand_left_keypoints_2d = dt_concated[25+70:25+70+21].flatten()
    hand_right_keypoints_2d = dt_concated[25+70+21:].flatten()

    output_json = {"pose_keypoints_2d":pose_keypoints_2d.tolist(),"face_keypoints_2d":face_keypoints_2d.tolist(),\
        "hand_left_keypoints_2d":hand_left_keypoints_2d.tolist(),"hand_right_keypoints_2d":hand_right_keypoints_2d.tolist(),\
        "pose_keypoints_3d":[],"face_keypoints_3d":[],"hand_left_keypoints_3d":[],"hand_right_keypoints_3d":[]}
    
    return output_json

run_flag = True
if run_flag:
  set_seed(seed=cfg["training"].get("random_seed", 42))
  tokenizer = get_tokenizer("basic_english")

  train_data, dev_data, src_vocab, trg_vocab = load_data(cfg, tokenizer)

  model = build_model(cfg, src_vocab=src_vocab, trg_vocab=trg_vocab)
  ckpt = os.path.join(MODEL_DIR, 'best.ckpt')
  model_checkpoint = load_checkpoint(path=ckpt, use_cuda=True)
  model.load_state_dict(model_checkpoint["model_state"])
  model.cuda()

  trainer = TrainManager(model=model, config=cfg)

  # score = average dtw score
  # references = targets
  # hypotheses = outputs
  # inputs = words
  # all_dtw_scores (all dtw scores)
  trainer.logger.info("Running validation")
  score, loss, references, hypotheses, \
  inputs, all_dtw_scores, file_paths = validate_on_data(model=model, 
                                                        data=dev_data,
                  batch_size=cfg["training"]["batch_size"],
                  max_output_length=cfg['training']["max_output_length"],
                  eval_metric='dtw',
                  loss_function=None,
                  batch_type='sentence',
                  type='val')
  trainer.logger.info("Mean DTW: {}".format(score))
  display = list(range(len(hypotheses)))

  ground_truth= []
  for r in references:
    ground_truth.append(r.detach().to('cpu').numpy())

  predictions = []
  for h in hypotheses:
    predictions.append(h.detach().to('cpu').numpy())

  # save everything
  all_outputs ={
      "mean_dtw": score,
      "dtw": all_dtw_scores,
      'words': np.array(inputs).squeeze(),
      "ground_truth": np.array(ground_truth),
      "predictions": np.array(predictions)
  }

  with open(os.path.join(MODEL_DIR, 'val_predictions.pkl'), 'wb') as f:
    pickle.dump(all_outputs, f)

  sentences =  np.array(inputs).squeeze()
else:
  with open(os.path.join(MODEL_DIR, 'val_predictions.pkl'), 'rb') as f:
      all_outputs = pickle.load(f)
      sentences = all_outputs["words"]
      ground_truth = all_outputs["ground_truth"]
      predictions = all_outputs["predictions"]
    

L = len(predictions)

for i in range(L): # for all instance
  tg_dir = os.path.join(RES_DIR, str(i))
  
  if not os.path.isdir(tg_dir):
    os.mkdir(tg_dir)
  os.makedirs(os.path.join(tg_dir,'gt'), exist_ok=True)
  os.makedirs(os.path.join(tg_dir,'pd'), exist_ok=True)
    
  gt_seqs = ground_truth[i]
  pd_seqs = predictions[i]
  
  for idx, gt in enumerate(gt_seqs):
    dt = np.reshape(gt[:-1],(-1,2))
    output_json = get_jsoned_pred(dt)
    
    f_name = os.path.join(os.path.join(tg_dir,'gt'), '%05d.json'%idx)
    with open(f_name, "w") as json_file:
        print(f"make file {f_name}")
        json.dump({'version':1.3, 'people':[output_json]}, json_file)
        
  for idx, pd in enumerate(pd_seqs):
    dt = np.reshape(pd[:-1],(-1,2))
    output_json = get_jsoned_pred(dt)
    
    f_name = os.path.join(os.path.join(tg_dir,'pd'), '%05d.json'%idx)
    with open(f_name, "w") as json_file:
        print(f"make file {f_name}")
        json.dump({'version':1.3, 'people':[output_json]}, json_file)
        
      

