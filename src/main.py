#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 12:47:47 2024

@author: wangyifei
"""

from utils import tag_to_word_df
import torch
from transformers import BertTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import contractions
from abte import ABTEModel
import warnings
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(1, '../dataset')

warnings.filterwarnings("ignore")
from pred_ABTE import run_ABTE_test_train


# prepare data
# this part needs to rewrite
# load
# data = pd.read_csv('../dataset/normalized/restaurants_train.csv')
# data_test = pd.read_csv('../dataset/normalized/restaurants_test.csv')



if __name__ == '__main__':
    run_ABTE_test_train(adapter=True, lr_schedule=False)