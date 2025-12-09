#!/usr/bin/env python3
"""model_plot.py
Description
Flow based Network Anomaly Detection with Transformer based model

Date
Dec 8, 2025
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
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, LSTM, Flatten, Embedding, MultiHeadAttention, LayerNormalization, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

# dup2 #
#fp = open(f"tmp/{datetime.now().strftime('%Y%m%d%H%M%S')}")
#orig = os.dup2(fp.fileno(), sys.stdout.fileno())

# Data Structures define - class #

def build_baseline_fcn(
        input_shape=(269,), hidden_sizes=(64,128,256,512,256), dropout=0.3, num_classes=10):
    x_in = Input(shape=input_shape, name="Input")
    x = x_in
    for i, size in enumerate(hidden_sizes):
        x = Dense(size, activation='relu', name=f"Dense#{i+1}:{size}")(x)
        # x = Dropout(dropout)(x)
    x_out = Dense(num_classes, activation='softmax', name="Output")(x)
    model = Model(inputs=x_in, outputs=x_out, name="BaselineFCN")
    return model

def build_cnn_lstm(
        input_shape=(8,269), cnn_filters=[64,128],
        kernel_size=3, lstm_units=[128,64],
        dropout_rate=0.3, num_classes=10):
    x_in = Input(shape=input_shape, name="Input")
    x = x_in
    for i, f in enumerate(cnn_filters):
        x = Conv1D(
            filters=f, kernel_size=kernel_size, activation='relu',
            padding='same', name=f"Conv1D#{i+1}:{f}")(x)
    for i, u in enumerate(lstm_units):
        x = LSTM(
            u, return_sequences=True,
            name=f"LSTM#{i+1}:{u}")(x)
    x = Flatten(name="Flatten")(x)
    # x = Dropout(dropout_rate)(x)
    x_out = Dense(num_classes, activation='softmax', name="Output")(x)
    model = Model(inputs=x_in, outputs=x_out, name="NetFlowCNNLSTM")
    return model

# For Graph only (Not a Real Transformer Emcoder)
class TransformerEncoderBlock(Layer):
    def __init__(self, embed_size, internal_size, n_heads, name=None):
        super().__init__(name=name)
        self.embed_size = embed_size
        self.internal_size = internal_size
        self.n_heads = n_heads
        self.ff1 = Dense(internal_size, activation='relu')
        self.ff2 = Dense(embed_size)

    def call(self, x):
        y = self.ff1(x)
        y = self.ff2(y)
        return x + y

def build_bert_classifier(input_shape=(8,269), num_classes=10, n_blocks=4):
    x_in = Input(shape=input_shape, name="Input")
    x = Dense(64, name="Embedding")(x_in)
    
    for i in range(n_blocks):
        x = TransformerEncoderBlock(embed_size=64, internal_size=256, n_heads=2,
                                    name=f"Encoder#{i+1}:256")(x)
    
    x = Flatten(name="Flatten")(x)
    # x = Dropout(0.1, name="Dropout")(x)
    x_out = Dense(num_classes, activation='softmax', name="Output")(x)
    
    model = Model(inputs=x_in, outputs=x_out, name="NetFlowBERT")
    return model


# Main Function Define
def main():
    feature_size = 269
    window_size = 8

    model1 = build_baseline_fcn(input_shape=(feature_size,))
    model2 = build_cnn_lstm(input_shape=(window_size, feature_size))
    model3 = build_bert_classifier(input_shape=(window_size, feature_size))

    plot_param = (True, True)
    
    plot_model(model1,
               to_file="plot/BaselineFCN.png",
               show_shapes=True,
               show_dtype=False,
               show_layer_names=True,
               rankdir="TD",
               expand_nested=False,
               dpi=200,
               show_layer_activations=False,
               show_trainable=False)

    plot_model(model2,
               to_file="plot/CNN+LSTM.png",
               show_shapes=True,
               show_dtype=False,
               show_layer_names=True,
               rankdir="TD",
               expand_nested=False,
               dpi=200,
               show_layer_activations=False,
               show_trainable=False)

    plot_model(model3,
               to_file="plot/NetFlowBERT.png",
               show_shapes=True,
               show_dtype=False,
               show_layer_names=True,
               rankdir="TD",
               expand_nested=False,
               dpi=200,
               show_layer_activations=False,
               show_trainable=False)
    
# EP
if __name__ == "__main__":
    #sys.exit(main(sys.argv[1:]))
    sys.exit(main())

#fp.close()
#os.close(orig)

