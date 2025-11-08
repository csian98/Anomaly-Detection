#!/usr/bin/env python3
"""main.py
Description
Flow based Network Anomaly Detection with Transformer based model

Date
Sep 15, 2025
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

from typing import Tuple, List, Dict, Any, Optional, override
from enum import Enum
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

try:
    from tensorflow._api.v2.v2 import keras
except:
    from tensorflow import keras

import tensorflow as tf
from keras import Input, Model
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

# FlowTransformer Framework Library
from framework.base_classification_head import BaseClassificationHead
from framework.base_input_encoding import BaseInputEncoding
from framework.base_preprocessing import BasePreProcessing
from framework.dataset_specification import DatasetSpecification
from framework.enumerations import EvaluationDatasetSampling, CategoricalFormat
from framework.flow_transformer import FlowTransformer
from framework.flow_transformer_parameters import FlowTransformerParameters
from framework.framework_component import FunctionalComponent
from framework.model_input_specification import ModelInputSpecification
from framework.utilities import get_identifier, load_feather_plus_metadata, save_feather_plus_metadata

# Customize Library
from customize.preprocessing import *
from customize.input_encoder import *
from customize.transformer import *
from customize.classification_head import *


# dup2 #
#fp = open(f"tmp/{datetime.now().strftime('%Y%m%d%H%M%S')}")
#orig = os.dup2(fp.fileno(), sys.stdout.fileno())

# Data Structures define - class #
# SIAN
class FlowTransformerMultiClass(FlowTransformer):
    def __init__(self, pre_processing:BasePreProcessing,
                 input_encoding:BaseInputEncoding,
                 sequential_model:FunctionalComponent,
                 classification_head:BaseClassificationHead,
                 params:FlowTransformerParameters,
                 rs:np.random.RandomState=None):
        super().__init__(pre_processing, input_encoding, sequential_model,
                         classification_head, params, rs)

    # SIAN
    def multiclass_encoding(self, labels:List):
        if self.y is None:
            raise Exception("Please call load_dataset before calling multiclass_encoding")

        self.output_size = len(labels)
        self.label_encode = {label:index for index, label in enumerate(labels)}
        self.label_decode = {index:label for index, label in enumerate(labels)}
        self.y = np.array([self.label_encode[label] for label in self.y])

    @override
    def build_model(self, prefix:str=None):
        if prefix is None:
            prefix = ""

        if self.X is None:
            raise Exception("Please call load_dataset before calling build_model()")

        m_inputs = []
        for numeric_feature in self.model_input_spec.numeric_feature_names:
            m_input = Input((self.parameters.window_size, 1),
                            name=f"{prefix}input_{numeric_feature}", dtype="float32")
            m_inputs.append(m_input)

        for categorical_feature_name, categorical_feature_levels in \
            zip(self.model_input_spec.categorical_feature_names,
                self.model_input_spec.levels_per_categorical_feature):
            m_input = Input(
                (self.parameters.window_size,
                 1 if self.model_input_spec.categorical_format == CategoricalFormat.Integers else categorical_feature_levels),
                name=f"{prefix}input_{categorical_feature_name}",
                dtype="int32" if self.model_input_spec.categorical_format == CategoricalFormat.Integers else "float32"
            )
            m_inputs.append(m_input)

        self.input_encoding.build(self.parameters.window_size, self.model_input_spec)
        self.sequential_model.build(self.parameters.window_size, self.model_input_spec)
        self.classification_head.build(self.parameters.window_size, self.model_input_spec)

        m_x = self.input_encoding.apply(m_inputs, prefix)

        # in case the classification head needs to add tokens at this stage
        m_x = self.classification_head.apply_before_transformer(m_x, prefix)

        m_x = self.sequential_model.apply(m_x, prefix)
        m_x = self.classification_head.apply(m_x, prefix)

        for layer_i, layer_size in enumerate(self.parameters.mlp_layer_sizes):
            m_x = Dense(layer_size, activation="relu",
                        name=f"{prefix}classification_mlp_{layer_i}_{layer_size}")(m_x)
            m_x = Dropout(self.parameters.mlp_dropout)(m_x) if self.parameters.mlp_dropout > 0 else m_x

        # SIAN
        m_x = Dense(self.output_size, activation="softmax", name=f"{prefix}multiclass_classification_out")(m_x)
        m = Model(m_inputs, m_x)
        #m.summary()
        return m

    @override
    def evaluate(self, m:keras.Model, batch_size, early_stopping_patience:int,
                 epochs:int=100, steps_per_epoch:int=128):
        n_malicious_per_batch = int(0.5 * batch_size)
        n_legit_per_batch = batch_size - n_malicious_per_batch

        overall_y_preserve = np.zeros(dtype="float32", shape=(n_malicious_per_batch + n_legit_per_batch,))
        overall_y_preserve[:n_malicious_per_batch] = 1.

        selectable_mask = np.zeros(len(self.X), dtype=bool)
        selectable_mask[self.parameters.window_size:-self.parameters.window_size] = True
        train_mask = self.training_mask

        y_mask = ~(self.y == self.label_encode[str(self.dataset_specification.benign_label)])

        indices_train = np.argwhere(train_mask).reshape(-1)
        malicious_indices_train = np.argwhere(train_mask & y_mask & selectable_mask).reshape(-1)
        legit_indices_train = np.argwhere(train_mask & ~y_mask & selectable_mask).reshape(-1)

        indices_test:np.ndarray = np.argwhere(~train_mask).reshape(-1)

        def get_windows_for_indices(indices:np.ndarray, ordered) -> List[pd.DataFrame]:
            X: List[pd.DataFrame] = []

            if ordered:
                # we don't really want to include eval samples as part of context, because out of range values might be learned
                # by the model, _but_ we are forced to in the windowed approach, if users haven't just selected the
                # "take last 10%" as eval option. We warn them prior to this though.
                for i1 in indices:
                    X.append(self.X.iloc[(i1 - self.parameters.window_size) + 1:i1 + 1])
            else:
                context_indices_batch = np.random.choice(indices_train, size=(batch_size, self.parameters.window_size),
                                                         replace=False).reshape(-1)
                context_indices_batch[:, -1] = indices

                for index in context_indices_batch:
                    X.append(self.X.iloc[index])

            return X

        feature_columns_map = {}

        def samplewise_to_featurewise(X):
            sequence_length = len(X[0])

            combined_df = pd.concat(X)

            featurewise_X = []

            if len(feature_columns_map) == 0:
                for feature in self.model_input_spec.feature_names:
                    if feature in self.model_input_spec.numeric_feature_names \
                       or self.model_input_spec.categorical_format == CategoricalFormat.Integers:
                        feature_columns_map[feature] = feature
                    else:
                        # this is a one-hot encoded categorical feature
                        feature_columns_map[feature] = [c for c in X[0].columns if str(c).startswith(feature)]

            for feature in self.model_input_spec.feature_names:
                feature_columns = feature_columns_map[feature]
                combined_values = combined_df[feature_columns].values

                # maybe this can be faster with a reshape but I couldn't get it to work
                combined_values = np.array([combined_values[i:i+sequence_length] \
                                            for i in range(0, len(combined_values), sequence_length)])
                featurewise_X.append(combined_values)

            return featurewise_X

        print(f"Building eval dataset...")
        eval_X = get_windows_for_indices(indices_test, True)
        print(f"Splitting dataset to featurewise...")
        eval_featurewise_X = samplewise_to_featurewise(eval_X)

        print(f"Evaluation dataset is built!")

        # SIAN
        eval_y = self.y[indices_test]

        print("Evaluation dataset size: %d" %np.count_nonzero(indices_test))
        for key, value in self.label_encode.items():
            print("Label: %s(%d)" %(key, value))
            eval_P = (eval_y == value)
            n_eval_P = np.count_nonzero(eval_P)
            eval_N = ~eval_P
            n_eval_N = np.count_nonzero(eval_N)
            print(f"Positive samples in eval set: {n_eval_P}")
            print(f"Negative samples in eval set: {n_eval_N}")

        epoch_results = []

        def run_evaluation(epoch):
            # SIAN
            pred_y = m.predict(eval_featurewise_X, verbose=True)
            pred_y = pred_y.argmax(axis=1).reshape(-1)

            print(f"Epoch {epoch} yielded predictions: {pred_y.shape}")
            for key, value in self.label_encode.items():
                pred_P = (pred_y == value)
                n_pred_P = np.count_nonzero(pred_P)
                pred_N = ~pred_P
                n_pred_N = np.count_nonzero(pred_N)

                eval_P = (eval_y == value)
                n_eval_P = np.count_nonzero(eval_P)
                eval_N = ~eval_P
                n_eval_N = np.count_nonzero(eval_N)

                TP = np.count_nonzero(pred_P & eval_P)
                FP = np.count_nonzero(pred_P & eval_N)
                TN = np.count_nonzero(pred_N & eval_N)
                FN = np.count_nonzero(pred_N & eval_P)

                sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
                specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
                balanced_accuracy = (sensitivity + specificity) / 2

                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0

                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                print(f"[{key} -- overall balanced accuracy: {balanced_accuracy * 100:.2f}%, TP = {TP:,} / {n_eval_P:,}, TN = {TN:,} / {n_eval_N:,}")

                epoch_results.append({
                    "epoch": epoch,
                    "label": key,
                    "P": n_eval_P,
                    "N": n_eval_N,
                    "pred_P": n_pred_P,
                    "pred_N": n_pred_N,
                    "TP": TP,
                    "FP": FP,
                    "TN": TN,
                    "FN": FN,
                    "bal_acc": balanced_accuracy,
                    "f1": f1_score
                })

        # SIAN
        y = self.y

        class BatchYielder():
            def __init__(self, ordered, random, rs):
                self.ordered = ordered
                self.random = random
                self.cursor_malicious = 0
                self.cursor_legit = 0
                self.rs = rs

            def get_batch(self):
                malicious_indices_batch = self.rs.choice(malicious_indices_train, size=n_malicious_per_batch,
                                                         replace=False) \
                    if self.random else \
                    malicious_indices_train[self.cursor_malicious:self.cursor_malicious + n_malicious_per_batch]

                legitimate_indices_batch = self.rs.choice(legit_indices_train, size=n_legit_per_batch, replace=False) \
                    if self.random else \
                    legit_indices_train[self.cursor_legit:self.cursor_legit + n_legit_per_batch]

                indices = np.concatenate([malicious_indices_batch, legitimate_indices_batch])

                self.cursor_malicious = self.cursor_malicious + n_malicious_per_batch
                self.cursor_malicious = self.cursor_malicious % (len(malicious_indices_train) - n_malicious_per_batch)

                self.cursor_legit = self.cursor_legit + n_legit_per_batch
                self.cursor_legit = self.cursor_legit % (len(legit_indices_train) - n_legit_per_batch)

                X = get_windows_for_indices(indices, self.ordered)
                # each x in X contains a dataframe, with window_size rows and all the features of the flows. There are batch_size of these.

                # we have a dataframe containing batch_size x (window_size, features)
                # we actually want a result of features x (batch_size, sequence_length, feature_dimension)
                featurewise_X = samplewise_to_featurewise(X)
                # SIAN
                batch_y = y[indices]
                # return featurewise_X, overall_y_preserve
                return featurewise_X, batch_y

        batch_yielder = BatchYielder(self.parameters._train_ensure_flows_are_ordered_within_windows,
                                     not self.parameters._train_draw_sequential_windows, self.rs)

        min_loss = 100
        iters_since_loss_decrease = 0

        train_results = []
        final_epoch = 0

        last_print = time.time()
        elapsed_time = 0

        for epoch in range(epochs):
            final_epoch = epoch

            has_reduced_loss = False
            for step in range(steps_per_epoch):
                batch_X, batch_y = batch_yielder.get_batch()

                t0 = time.time()
                batch_results = m.train_on_batch(batch_X, batch_y)
                t1 = time.time()

                if epoch > 0 or step > 0:
                    elapsed_time += (t1 - t0)
                    if epoch == 0 and step == 1:
                        # include time for last "step" that we skipped with step > 0 for epoch == 0
                        elapsed_time *= 2

                train_results.append(batch_results + [elapsed_time, epoch])

                batch_loss = batch_results[0] if isinstance(batch_results, list) else batch_results

                if time.time() - last_print > 3:
                    last_print = time.time()
                    early_stop_phrase = "" if early_stopping_patience <= 0 else f" (early stop in {early_stopping_patience - iters_since_loss_decrease:,})"
                    print(f"Epoch = {epoch:,} / {epochs:,}{early_stop_phrase}, step = {step}, loss = {batch_loss:.5f}, results = {batch_results} -- elapsed (train): {elapsed_time:.2f}s")

                if batch_loss < min_loss:
                    has_reduced_loss = True
                    min_loss = batch_loss

            if has_reduced_loss:
                iters_since_loss_decrease = 0
            else:
                iters_since_loss_decrease += 1

            do_early_stop = early_stopping_patience > 0 and iters_since_loss_decrease > early_stopping_patience
            is_last_epoch = epoch == epochs - 1
            run_eval = epoch in [6] or is_last_epoch or do_early_stop

            if run_eval:
                run_evaluation(epoch)

            if do_early_stop:
                print(f"Early stopping at epoch: {epoch}")
                break

        eval_results = pd.DataFrame(epoch_results)

        return (train_results, eval_results, final_epoch)


# SIAN
class BERT(BaseSequential):
    @property
    def name(self) -> str:
        return "BERT Model"

    @property
    def parameters(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "internal_size": self.internal_size,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate,
            "head_size": self.head_size
        }

    def __init__(self, n_layers=12, internal_size=768, n_heads=12):
        super().__init__()
        self.n_layers = n_layers
        self.internal_size = internal_size
        self.n_heads = n_heads
        self.head_size = self.internal_size / self.n_heads
        self.dropout_rate = 0.02
        self.is_decoder = False

    def apply(self, X, prefix: str = None):
        #window_size = self.sequence_length
        real_size = X.shape[-1]

        m_x = X

        for layer_i in range(self.n_layers):
            m_x = TransformerEncoderBlock(real_size, self.internal_size, self.n_heads,
                                          dropout_rate=self.dropout_rate, prefix=f"block_{layer_i}_")(m_x)

        return m_x

# Main Function Define
def main():
    print(tf.config.list_physical_devices("GPU"))

    data_path = "data/"
    feature = "NetFlow_v2_Features.csv"
    datasets = ["NF-CSE-CIC-IDS2018-v2/NF-CSE-CIC-IDS2018-v2.csv", "NF-UNSW-NB15-v2/NF-UNSW-NB15-v2.csv"]
    
    cache_folder = "tmp/cache"
    if not os.path.exists(cache_folder):
        os.mkdir(cache_folder)

    flow_format = DatasetSpecification(
        include_fields=['NUM_PKTS_UP_TO_128_BYTES', 'SRC_TO_DST_SECOND_BYTES',
                        'OUT_PKTS', 'OUT_BYTES', 'NUM_PKTS_128_TO_256_BYTES',
                        'DST_TO_SRC_AVG_THROUGHPUT', 'DURATION_IN',
                        'L4_SRC_PORT', 'ICMP_TYPE',
                        'PROTOCOL', 'SERVER_TCP_FLAGS',
                        'IN_PKTS', 'NUM_PKTS_512_TO_1024_BYTES',
                        'CLIENT_TCP_FLAGS', 'TCP_WIN_MAX_IN',
                        'NUM_PKTS_256_TO_512_BYTES', 'SHORTEST_FLOW_PKT',
                        'MIN_IP_PKT_LEN', 'LONGEST_FLOW_PKT',
                        'L4_DST_PORT', 'MIN_TTL',
                        'DST_TO_SRC_SECOND_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES',
                        'DURATION_OUT', 'FLOW_DURATION_MILLISECONDS',
                        'TCP_FLAGS', 'MAX_TTL',
                        'SRC_TO_DST_AVG_THROUGHPUT', 'ICMP_IPV4_TYPE',
                        'MAX_IP_PKT_LEN', 'RETRANSMITTED_OUT_BYTES',
                        'IN_BYTES', 'RETRANSMITTED_IN_BYTES',
                        'TCP_WIN_MAX_OUT', 'L7_PROTO',
                        'RETRANSMITTED_OUT_PKTS', 'RETRANSMITTED_IN_PKTS'],
        categorical_fields=['CLIENT_TCP_FLAGS', 'L4_SRC_PORT',
                            'TCP_FLAGS', 'ICMP_IPV4_TYPE',
                            'ICMP_TYPE', 'PROTOCOL',
                            'SERVER_TCP_FLAGS', 'L4_DST_PORT',
                            'L7_PROTO'],
        class_column="Attack",
        benign_label="Benign",
    )

    pre_processing = StandardPreProcessing(n_categorical_levels=32)
    encoding = RecordLevelEmbed(64)
    # transformer = BasicTransformer(n_layers=2, internal_size=128, n_heads=2)
    transformer = BERT(n_layers=2, internal_size=128, n_heads=2)
    classification_head = LastTokenClassificationHead()

    # SIAN
    ft = FlowTransformerMultiClass(pre_processing=pre_processing,
                                   input_encoding=encoding,
                                   sequential_model=transformer,
                                   classification_head=classification_head,
                                   params=FlowTransformerParameters(
                                       window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1))

    df = ft.load_dataset("UNSW-NB15",
                         data_path+datasets[1],
                         specification=flow_format,
                         evaluation_dataset_sampling=EvaluationDatasetSampling.LastRows,
                         # evaluation_percent=0.1,
                         cache_path=cache_folder)
    
    # SIAN
    df["Attack"] = ft.y
    print(df.shape)
    df.to_csv("data/nf-pre/NF-UNSW-NB15-v2-pre.csv", index=False)


# EP
if __name__ == "__main__":
    #sys.exit(main(sys.argv[1:]))
    sys.exit(main())


#fp.close()
#os.close(orig)
