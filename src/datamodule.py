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

import numpy as np
import pandas as pd

# Data Structures define - class #

# assigned: @sian
class DataModule:
    """Sample Class"""
    def __init__(self):
        path = "data/CIC-IDS-2017/MachineLearningCVE/"
        packets = [
            "Monday-WorkingHours.pcap_ISCX.csv",
            "Tuesday-WorkingHours.pcap_ISCX.csv",
            "Wednesday-workingHours.pcap_ISCX.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
        ]
        
        data = []
        for packet in packets:
            data.append(pd.read_csv(path + packet))
            
        self.df = pd.concat(data)

    def __repr__(self) -> str:
        return f"DataModule(shape={self.df.shape})"

    def __str__(self) -> str:
        return f"{self.df.shape}"

# Functions define #

def sample_add(a: int, b: int) -> int:
    """sample add functions"""
    return a + b

# Closure & Decorator

def sample_decorator(func):
    local_variable = 0
    def sample_wrapper(*args, **kwargs):
        nonlocal local_variable
        #
        result = func(args, kwargs)
        #
        return result
    return inner_decorator

# Main function define #

def main(*args, **kwargs):
    a = 10
    b = 15
    c = sample_add(a, b)

# EP
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
