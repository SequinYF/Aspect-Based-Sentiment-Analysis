#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 19:15:00 2024

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
from consts import *
sys.path.insert(1, '../dataset')

warnings.filterwarnings("ignore")


# prepare data
# this part needs to rewrite

# load
# data = pd.read_csv('../dataset/normalized/restaurants_train.csv')
# data_test = pd.read_csv('../dataset/normalized/restaurants_test.csv')


def run_ABTE_test_train(adapter, lr_schedule):
    if adapter:
        if lr_schedule:
            dir_name = "model_ABTE_adapter_scheduler"
        else:
            dir_name = "model_ABTE_adapter"
    else:
        if lr_schedule:
            dir_name = "model_ABTE_scheduler"
        else:
            dir_name = "model_ABTE"

    # load
    data = pd.read_csv('../dataset/normalized/restaurants_train.csv')
    data_test = pd.read_csv('../dataset/normalized/restaurants_test.csv')

    # define parameters for model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")  # cuda GPU
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")  # check mac mps
    else:
        DEVICE = torch.device("cpu")  # otherwise cpu
    print(f"Test using device: {DEVICE}")

    # define model
    modelABTE = ABTEModel(tokenizer, adapter=adapter)

    # load model and predict
    model_path = dir_name+'/model_lr3.0000000000000004e-05_epochs4_batch8.pkl'
    test_accuracy, test_report = modelABTE.test(
        data_test, load_model=model_path, device=DEVICE)
    test_pred, test_targets = modelABTE.predict_batch(
        data_test, load_model=model_path, device=DEVICE)

    train_accuracy, train_report = modelABTE.test(
        data, load_model=model_path, device=DEVICE)
    train_pred, train_targets = modelABTE.predict_batch(
        data, load_model=model_path, device=DEVICE)

    # save results
    if not os.path.exists('/results'):
        os.makedirs(dir_name+'/results')

    # report
    with open(dir_name+'/results/test_report_lr{}_epochs{}_batch{}.csv'.format(lr, epochs, batch), 'w') as f:
        for r in test_report.split('\n'):
            f.write(r + '\n')

    with open(dir_name+'/results/train_report_lr{}_epochs{}_batch{}.csv'.format(lr, epochs, batch), 'w') as f:
        for r in train_report.split('\n'):
            f.write(r + '\n')

    # predictions
    data_test['Predicted'] = test_pred
    data_test['Actual'] = test_targets
    data_test.to_csv(
        dir_name+'/results/test_pred_lr{}_epochs{}_batch{}.csv'.format(lr, epochs, batch), index=False)

    data['Predicted'] = train_pred
    data['Actual'] = train_targets
    data.to_csv(dir_name+'/results/train_pred_lr{}_epochs{}_batch{}.csv'.format(lr,
                epochs, batch), index=False)

    # accuracy
    test_accuracy = np.array(test_accuracy)
    train_accuracy = np.array(train_accuracy)

    with open(dir_name+'/results/test_accuracy_lr{}_epochs{}_batch{}.csv'.format(lr, epochs, batch), 'w') as f:
        f.write(str(test_accuracy))
    with open(dir_name+'/results/train_accuracy_lr{}_epochs{}_batch{}.csv'.format(lr, epochs, batch), 'w') as f:
        f.write(str(train_accuracy))

