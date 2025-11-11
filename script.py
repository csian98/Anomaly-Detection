import os, sys
sys.path.append("pylib/")
import warnings
warnings.filterwarnings("ignore")
import logging
from IPython.display import display

from typing import Tuple, List
import time
from datetime import datetime

import numpy as np
import pandas as pd


from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from sklearn.model_selection import train_test_split

from customize.NetFlowUtility import NetFlowLabelEncoder, NetFlowBatchLoader
from customize.NetFlowBertClassifier import NetFlowBertClassifier

tf.config.set_visible_devices([], 'GPU')
data_path = "data/nf-pre/NF-UNSW-NB15-v2-pre.csv"
window_size = 8
batch_size = 32
embed_size = 32
n_layers = 2
internal_size = 128
n_heads = 2
dropout = 0.1
random_state = 4444

df = pd.read_csv(data_path)
le = NetFlowLabelEncoder()
df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])

train_idx, temp_idx = train_test_split(
    df[window_size:].index,
    test_size=0.3,
    stratify=df.iloc[window_size:, -1],
    random_state=random_state
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    stratify=df.iloc[temp_idx, -1],
    random_state=random_state
)

train_ds = NetFlowBatchLoader(df, train_idx, window_size, batch_size)

model = NetFlowBertClassifier(
    embed_size=embed_size,
    n_layers=n_layers,
    internal_size=internal_size,
    n_heads=n_heads,
    dropout=dropout,
    num_classes=le.size
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
