#!/usr/bin/env python3
"""BaselineFCN.py
Description
Baseline FCN model

Date
Nov 17, 2025
"""

__author__ = "Jeong Hoon Choi"
__version__ = "1.0.0"

# Import #
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Data Structures define - class #
class NetFlowBaselineFCN(tf.keras.Model):
    def __init__(self, hidden_sizes=[128, 64], dropout=0.3, num_classes=10):
        super().__init__()
        self.hidden_layer = []
        for h in hidden_sizes:
            self.hidden_layer.append(layers.Dense(h, activation="relu"))
            self.hidden_layer.append(layers.Dropout(dropout))

        self.output_layer = layers.Dense(num_classes, activation="softmax")
        
    @tf.function
    def call(self, X, training=False):
        for layer in self.hidden_layer:
            X = layer(X, training=training)
        return self.output_layer(X)

# Functions define #
def NetFlowBaseBatchLoader(X, y, indices, batch_size:int=32):
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
            tf.TensorSpec(shape=(None, X.shape[1]), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    ).prefetch(tf.data.AUTOTUNE)
    return dataset
