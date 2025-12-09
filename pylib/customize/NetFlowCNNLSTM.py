#!/usr/bin/env python3
"""NetFlowCNNLSTM.py
Description
NetFlow CNN-LSTM model

Date
Nov 26, 2025
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
class NetFlowCNNLSTM(tf.keras.Model):
    def __init__(self, num_classes, cnn_filters, kernel_size,
                 lstm_units, dropout_rate):
        super().__init__()
        
        self.conv1 = layers.Conv1D(cnn_filters[0], kernel_size, padding="same", activation="relu")
        self.conv2 = layers.Conv1D(cnn_filters[1], kernel_size, padding="same", activation="relu")

        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

        self.maxpool1 = layers.MaxPooling1D(pool_size=2)
        self.maxpool2 = layers.MaxPooling1D(pool_size=2)

        self.drop1 = layers.Dropout(dropout_rate)
        self.drop2 = layers.Dropout(dropout_rate)

        self.lstm1 = layers.LSTM(lstm_units[0], return_sequences=True)
        self.lstm2 = layers.LSTM(lstm_units[0], return_sequences=False)

        self.drop3 = layers.Dropout(dropout_rate)
        self.drop4 = layers.Dropout(dropout_rate)

        self.classifier = layers.Dense(num_classes, activation="softmax")
       
    @tf.function
    def call(self, X, training=False):
        X = self.conv1(X, training=training)
        X = self.bn1(X, training=training)
        X = self.maxpool1(X, training=training)
        X = self.drop1(X, training=training)

        X = self.conv2(X, training=training)
        X = self.bn2(X, training=training)
        X = self.maxpool2(X, training=training)
        X = self.drop2(X, training=training)

        X = self.lstm1(X, training=training)
        X = self.drop3(X, training=training)

        X = self.lstm2(X, training=training)
        X = self.drop4(X, training=training)

        X = self.classifier(X, training=training)
        return X

# Functions define #

