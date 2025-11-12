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

from customize.NetFlowUtility import NetFlowLabelEncoder, NetFlowSlicing, NetFlowBatchLoader, print_result
from customize.NetFlowBertClassifier import NetFlowBertClassifier

tf.config.set_visible_devices([], 'GPU')
data_path = "data/nf-pre/NF-UNSW-NB15-v2-pre.csv"
window_size = 8
batch_size = 32
epochs = 1
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
    df.index,
    test_size=0.3,
    stratify=df.iloc[:, -1],
    random_state=random_state
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    stratify=df.iloc[temp_idx, -1],
    random_state=random_state
)

valid_idx = df.index[window_size-1:]
train_mask = np.isin(valid_idx.to_numpy(), train_idx)
val_mask = np.isin(valid_idx.to_numpy(), val_idx)
test_mask = np.isin(valid_idx.to_numpy(), test_idx)

X, y = NetFlowSlicing(df, window_size=window_size)

train_ds = NetFlowBatchLoader(X, y, np.where(train_mask)[0], batch_size)
val_ds = NetFlowBatchLoader(X, y, np.where(val_mask)[0], batch_size)
test_ds = NetFlowBatchLoader(X, y, np.where(test_mask)[0], batch_size)

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

# model.build(input_shape=train_ds.element_spec[0].shape)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    steps_per_epoch=len(train_idx)//batch_size,
    callbacks=[early_stop, lr_scheduler],
    verbose=1)

y_true = np.concatenate([])
y_pred = model.predict(test_ds)
y_pred = np.argmax(y_pred, axis=1)

print_result(y_true, y_pred, le)
