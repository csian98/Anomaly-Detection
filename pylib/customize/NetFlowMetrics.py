#!/usr/bin/env python3
"""NetFlowMetrics.py
Description
Metrics function for NetFlowBertClassifier

Date
Nov 12, 2025
"""

__author__ = "Jeong Hoon Choi"
__version__ = "1.0.0"

# Import #
import sys
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from sklearn.metrics import multilabel_confusion_matrix, f1_score, accuracy_score

# Data Structures define - class #

class PrecisionMetric(tf.keras.metrics.Metric):
    def __init__(self, name="precision", **kwargs):
        super().__init__(name=name, **kwargs)
        self.metric = tf.keras.metrics.Precision()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, tf.int64)
        y_pred = tf.cast(y_pred, tf.int64)
        self.metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.metric.result()

    def reset_state(self):
        self.metric.reset_state()


class RecallMetric(tf.keras.metrics.Metric):
    def __init__(self, name="recall", **kwargs):
        super().__init__(name=name, **kwargs)
        self.metric = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, tf.int64)
        y_pred = tf.cast(y_pred, tf.int64)
        self.metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.metric.result()

    def reset_state(self):
        self.metric.reset_state()


class F1Macro(tf.keras.metrics.Metric):
    def __init__(self, name="f1_macro", **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = PrecisionMetric()
        self.recall = RecallMetric()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * p * r / (p + r + 1e-8)

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

        
# Functions define #

def print_result(y_true, y_pred, le, file=sys.stdout):
    labels = list(le.decoder.keys())
    label_names = [le.decoder[i] for i in labels]
    mcm = multilabel_confusion_matrix(y_true, y_pred)

    micro_f1 = f1_score(y_true, y_pred, average="micro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    overall_acc = accuracy_score(y_true, y_pred)

    print(f"=== Overall Metrics ===\n"
          f"--- Micro F1-score: {micro_f1:.4f}\n"
          f"--- Macro F1-score: {macro_f1:.4f}\n"
          f"--- Overall Accuracy: {overall_acc:.4f}\n", file=file)
    
    print("=== Label Metrics ===", file=file)
    for i, label in enumerate(label_names):
        tn, fp, fn, tp = mcm[i].ravel()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        sensitivity = recall
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        balanced_acc = (sensitivity + specificity) / 2 * 100.0
        print(f"[{label}]\n"
              f"-- TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n"
              f"-- Precision: {precision:.4f} ({tp} / {tp + fp})\n"
              f"-- Recall: {recall:.4f} ({tp} / {tp + fn})\n"
              f"-- Specificity: {specificity:.4f} ({tn} / {tn + fp})\n"
              f"-- Accuracy: {accuracy:.4f}\n"
              f"-- F1-score: {f1:.4f}\n"
              f"-- Balanced Accuracy: {balanced_acc:.4f}\n", file=file)
