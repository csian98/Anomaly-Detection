#!/usr/bin/env python3
"""NetFlowBertClassifier.py
Description
Flow based BERT model

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
from tensorflow.keras import layers, models

# Data Structures define - class #
class TransformerEncoder(layers.Layer):
    def __init__(self, num_hiddens:int, num_heads:int,
                 ffn_num_hiddens:int, dropout=0.1):
        super().__init__()
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=num_hiddens)
        self.ffn = models.Sequential([
            layers.Dense(ffn_num_hiddens, activation="relu"),
            layers.Dense(num_hiddens),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, X, training=False):
        fX = self.attention(X, X)	# self attention
        fX = self.dropout1(fX, training=training)
        X = self.layernorm1(X + fX)	# add & norm

        fX = self.ffn(X)
        fX = self.dropout2(fX, training=training)
        X = self.layernorm2(X + fX)	# add & norm
        return X

class NetFlowBertClassifier(tf.keras.Model):
    def __init__(self, embed_size=64, n_layers=2, internal_size=128,
                 n_heads=2, dropout=0.1, num_classes=10):
        super().__init__()
        self.embedding = layers.Dense(embed_size)
        self.encoders = [
            TransformerEncoder(embed_size, n_heads, internal_size, dropout)
            for _ in range(n_layers)]

        # self.pool = layers.GlobalAveragePooling1D()
        # self.dropout = layers.Dropout(dropout)
        self.classifier = layers.Dense(num_classes, activation="softmax")

    @tf.function
    def call(self, X, training=False):
        X = self.embedding(X)
        for encoder in self.encoders:
            X = encoder(X, training=training)

        # X = self.pool(X)
        # X = self.dropout(X, training=training)
        X = X[:, -1, :]
        X = self.classifier(X)
        return X
