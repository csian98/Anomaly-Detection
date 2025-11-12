#!/usr/bin/env python3
"""NetFlowUtility.py
Description
Utility functions for NetFlowBertClassifier

Date
Nov 10, 2025
"""

__author__ = "Jeong Hoon Choi"
__version__ = "1.0.0"

# Import #
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Data Structures define - class #

class NetFlowLabelEncoder:
    def __init__(self):
        self.encoder = {}
        self.decoder = {}
        self.fitted = False
        self.size = 0

    def fit(self, y):
        labels = sorted(y.unique())
        self.encoder = {label:index for index, label in enumerate(labels)}
        self.decoder = {index:label for index, label in enumerate(labels)}
        self.fitted = True
        self.size = len(labels)

    def transform(self, y):
        if not self.fitted:
            raise ValueError("NetFlowLabelEncoder not fitted")
        return np.array([self.encoder[label] for label in y])

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y_hat):
        if not self.fitted:
            raise ValueError("NetFlowLabelEncoder not fitted")
        return np.array([self.decoder[int(idx)]] for idx in y_hat)
        

# Functions define #

def NetFlowSlicing(df, window_size:int=16):
    array = df.to_numpy(dtype=np.float32)
    features = array[:, :-1]
    labels = array[:, -1].astype(np.int32)

    # https://numpy.org/devdocs/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
    X = np.lib.stride_tricks.sliding_window_view(features, (window_size, features.shape[1]))
    X = X.reshape(-1, window_size, features.shape[1])
    y = labels[window_size - 1:]
    
    return X, y


def NetFlowBatchLoader(X, y, indices, batch_size:int=32):
    def generator():
        batch_X, batch_y = [], []
        for index in indices:
            batch_X.append(X[index])
            batch_y.append(y[index])
            if len(batch_X) == batch_size:
                yield np.array(batch_X, dtype=np.float32), np.array(batch_y, dtype=np.int32)
                batch_X, batch_y = [], []

        if batch_X:
            yield np.array(batch_X, dtype=np.float32), np.array(batch_y, dtype=np.int32)
            
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, X.shape[1], X.shape[2]), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    ).prefetch(tf.data.AUTOTUNE)
    return dataset
