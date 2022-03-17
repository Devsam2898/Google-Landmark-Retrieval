import os
import gc
import cv2
import copy
import time
import random
from PIL import Image

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict

# Sklearn Imports
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split

import timm

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# For colored terminal text
from colorama import Fore, Back, Style
g_ = Fore.GREEN
c_ = Fore.CYAN
b_ = Fore.BLUE
sr_ = Style.RESET_ALL

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

ROOT_DIR = "../input/landmark-retrieval-2021"
TRAIN_DIR = "../input/landmark-retrieval-2021/train"
TEST_DIR = "../input/landmark-retrieval-2021/test"

CONFIG = dict(
    seed = 42,
    model_name = 'tf_mobilenetv3_small_100',
    train_batch_size = 384,
    valid_batch_size = 768,
    img_size = 224,
    epochs = 3,
    learning_rate = 5e-4,
    scheduler = None,
    # min_lr = 1e-6,
    # T_max = 20,
    # T_0 = 25,
    # warmup_epochs = 0,
    weight_decay = 1e-6,
    n_accumulate = 1,
    n_fold = 5,
    num_classes = 81313,
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    competition = 'GOOGL',
    _wandb_kernel = 'deb'
)

def get_train_file_path(id):
    return f"{TRAIN_DIR}/{id[0]}/{id[1]}/{id[2]}/{id}.jpg"
    
df = pd.read_csv(f"{ROOT_DIR}/train.csv")

le = LabelEncoder()
df.landmark_id = le.fit_transform(df.landmark_id)
joblib.dump(le, 'label_encoder.pkl')

df['file_path'] = df['id'].apply(get_train_file_path)

df_train, df_test = train_test_split(df, test_size=0.4, stratify=df.landmark_id, 
                                     shuffle=True, random_state=CONFIG['seed'])
df_valid, df_test = train_test_split(df_test, test_size=0.5, shuffle=True, 
                                     random_state=CONFIG['seed'])
