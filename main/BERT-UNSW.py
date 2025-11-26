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
from sklearn.utils.class_weight import compute_class_weight

from customize.NetFlowUtility import NetFlowLabelEncoder, NetFlowSlicing, NetFlowBatchLoader
from customize.NetFlowMetrics import PrecisionMetric, RecallMetric, F1Macro, print_result
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
        epochs:int=300,
        embed_size:int=64,
        n_layers:int=2,
        internal_size:int=128,
        n_heads:int=2,
        dropout:float=0.1,
        class_weight:bool=False,
        fout:str=None,
        random_state:int=4444):
    params = locals()

    fp = sys.stdout
    if fout:
        fp = open(fout, 'a')
    
    print(tf.config.list_physical_devices("GPU"), file=fp)
    # tf.debugging.set_log_device_placement=True
    # mixed_precision.set_global_policy("mixed_float16")
    
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
    train_mask = np.isin(valid_idx, train_idx)
    val_mask = np.isin(valid_idx, val_idx)
    test_mask = np.isin(valid_idx, test_idx)

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
    
    metrics = ["accuracy", F1Macro(), PrecisionMetric(), RecallMetric()] # ["accuracy"]
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=metrics,
    )
    
    # model.build(input_shape=train_ds.element_spec[0].shape)

    monitor = "val_loss"
    mode = "auto"

    if not class_weight:
        monitor = "val_f1_macro"
        mode = "max"
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=5,
        restore_best_weights=True
    )
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        mode=mode,
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    class_weights = None
    
    if class_weight:
        classes = np.unique(y[np.where(train_mask)[0]])
        class_weights = compute_class_weight("balanced",
                                             classes=classes,
                                             y=y[np.where(train_mask)[0]])
        class_weights = dict(zip(classes, class_weights))
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=len(train_idx)//batch_size,
        callbacks=[early_stop, lr_scheduler],
        class_weight=class_weights,
        verbose=1)

    y_true = y[np.where(test_mask)[0]]
    y_pred = model.predict(test_ds)
    y_pred = np.argmax(y_pred, axis=1)

    print("=== Model Training Configuration ===", file=fp)
    for key, value in params.items():
        print(f"{key:<15}: {value}", file=fp)
    print(file=fp)
    
    print_result(y_true, y_pred, le, fp)

    if fout:
        fp.close()

# EP
if __name__ == "__main__":
    #sys.exit(main(sys.argv[1:]))
    sys.exit(main(
        data_path="data/nf-pre/NF-UNSW-NB15-v2-pre.csv",
        window_size=16,#
        batch_size=256,
        epochs=300,
        embed_size=64,
        n_layers=4,
        internal_size=512,
        n_heads=8,#
        dropout=0.1,
        class_weight=True,
        fout="tmp/metrics.txt",
        random_state=4444))

#fp.close()
#os.close(orig)

