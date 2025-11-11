#!/usr/bin/env python3
"""main.py
Description
Flow based Network Anomaly Detection with Transformer based model

Date
Nov 10, 2025
"""

__author__ = "Jeong Hoon Choi"
__version__ = "1.0.0"


# Import #
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
from tensorflow.keras import mixed_precision
from sklearn.model_selection import train_test_split

from customize.NetFlowUtility import NetFlowLabelEncoder, NetFlowSlicing, NetFlowBatchLoader
from customize.NetFlowBertClassifier import NetFlowBertClassifier

# dup2 #
#fp = open(f"tmp/{datetime.now().strftime('%Y%m%d%H%M%S')}")
#orig = os.dup2(fp.fileno(), sys.stdout.fileno())

# Data Structures define - class #


# Main Function Define
def main(
        data_path:str,
        window_size:int=8,
        batch_size:int=64,
        epochs:int=20,
        embed_size:int=32,
        n_layers:int=2,
        internal_size:int=128,
        n_heads:int=2,
        dropout:float=0.1,
        random_state:int=4444):
    print(tf.config.list_physical_devices("GPU"))
    # tf.debugging.set_log_device_placement=True
    # mixed_precision.set_global_policy("mixed_float16")
    
    df = pd.read_csv(data_path)
    le = NetFlowLabelEncoder()
    df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])
    
    train_idx, temp_idx = train_test_split(
        df[:-window_size].index,
        test_size=0.3,
        stratify=df.iloc[:-window_size, -1],
        random_state=random_state
    )
    
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=df.iloc[temp_idx, -1],
        random_state=random_state
    )

    X, y = NetFlowSlicing(df, window_size=window_size)
    
    train_ds = NetFlowBatchLoader(X, y, train_idx, batch_size)
    val_ds = NetFlowBatchLoader(X, y, val_idx, batch_size)
    test_ds = NetFlowBatchLoader(X, y, test_idx, batch_size)
    
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

# EP
if __name__ == "__main__":
    #sys.exit(main(sys.argv[1:]))
    sys.exit(main(
        data_path="data/nf-pre/NF-UNSW-NB15-v2-pre.csv",
        window_size=8,
        batch_size=256,
        epochs=20,
        embed_size=32,
        n_layers=2,
        internal_size=128,
        n_heads=2,
        dropout=0.1,
        random_state=4444))

#fp.close()
#os.close(orig)
