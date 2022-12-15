import pickle
import os 

# constant
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
EOS_TOKEN = '</s>'
BOS_TOKEN = '<s>'

DATA_DIR = './data'
CONF_DIR = './ProgressiveTransformersSLP/Configs/'
MODEL_DIR = './model'

TARGET_PAD = 0.0
DEFAULT_UNK_ID = lambda: 0

import pickle
import os 

# load training_data
with open(os.path.join(DATA_DIR, 'train.pkl'), 'rb') as f:
    train_data = pickle.load(f)

lookup_trg_size = train_data[0]['ttl_kpts_seq'].shape[1]*2