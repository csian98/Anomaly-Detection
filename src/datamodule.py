#!/usr/bin/env python3
""" datamodule.py
Description

Date
Sep 16, 2025
"""
__author__ = "Jeong Hoon Choi & Yuxuan Liu"
__version__ = "1.0.0"

# Import #
import os, sys
sys.path.append("src/")
#from datamodule import DataModule

from typing import List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Data Structures define - class #

# assigned: @sian
# imbalance dataset: SMOTE or class weighting require
class DataModule:
    """DataModule Class"""
    def __init__(self, batch_size: int = 64, test_size: float = 0.2,
                 val_size: float = 0.1, random_state: int = 565):
        path = "data/CIC-IDS-2017/MachineLearningCVE/"
        files = [
            "Monday-WorkingHours.pcap_ISCX.csv",
            "Tuesday-WorkingHours.pcap_ISCX.csv",
            "Wednesday-workingHours.pcap_ISCX.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
        ]
        
        data = [pd.read_csv(path + x) for x in files]
        
        self._df = pd.concat(data)
        self.shape = self._df.shape
        self.features = [x.strip() for x in self._df.columns]
        self._df.columns = self.features
        self._df.loc[:, "Label"] = self._df.loc[:, "Label"].str.replace(" � ", "-", regex=False)
        self.labels = self._df.loc[:, "Label"].unique().tolist()

        # Label Encoding
        self._le = LabelEncoder()
        self._df.loc[:, "Label"] = self._le.fit_transform(self._df.loc[:, "Label"])

        # nan, inf Imputation
        self._median = self._df.groupby("Label")[self.features].median()
        self._df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self._df[self.features] = self._df.groupby("Label")[self.features].transform(
            lambda x: x.fillna(x.median()))
        
        X = self._df.drop(columns=["Label"]).values
        y = self._df.loc[:, "Label"].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, stratify=y_train, random_state=random_state)

        # Scale feature
        self._scaler = StandardScaler()
        X_train = self._scaler.fit_transform(X_train)
        X_val = self._scaler.transform(X_val)
        X_test = self._scaler.transform(X_test)

        self._train = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                   torch.tensor(y_train, dtype=torch.long))
        self._val = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                 torch.tensor(y_val, dtype=torch.long))
        self._test = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                  torch.tensor(y_test, dtype=torch.long))
        
        self.batch_size = batch_size

    def __repr__(self) -> str:
        return f"DataModule(shape={self.shape})"

    def __str__(self) -> str:
        return f"{self.shape}"

    def train_dataloader(self):
        return DataLoader(self._train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self._val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self._test, batch_size=self.batch_size, shuffle=False)

    def trnasform(self, df):
        df = df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        for feature in features:
            df[feature] = [self._median.loc[label, feature] if pd.isna(val) else val
                           for val, label in zip(df, df.loc[:, "Label"])]

        X = df.drop(columns=["Label"]).values
        y = df.loc[:, "Label"].values
        
        X = self._scaler.transform(X)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def encode_labels(self, decoded: List) -> List:
        return self._le.transform(decoded)

    def decode_labels(self, encoded: List) -> List:
        return self._le.inverse_transform(encoded)
