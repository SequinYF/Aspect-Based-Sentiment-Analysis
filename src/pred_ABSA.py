#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:05:14 2024

@author: wangyifei
"""

from pandas import pd
from numpy import np
from absa import ABSAModel
from consts import *
import os
# save results
lr = 1e-5


def run_ABSA_test_train(adapter, lr_schedule):
    if adapter:
        if lr_schedule:
            dir_name_s = "model_ABSA_adapter_scheduler"
        else:
            dir_name_s = "model_ABSA_adapter"
    else:
        if lr_schedule:
            dir_name_s = "model_ABSA_scheduler"
        else:
            dir_name_s = "model_ABSA"

    # load
    data = pd.read_csv(TRAIN_DATA_PATH)
    data_test = pd.read_csv(TEST_DATA_PATH)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    modelABSA = ABSAModel(tokenizer, adapter=adapter)

    model_path = dir_name_s+'/model_lr1e-05_epochs4_batch8.pkl'
    test_accuracy, test_report = modelABSA.test(
        data_test, load_model=model_path, device=DEVICE)
    test_pred, test_pol = modelABSA.predict_batch(
        data_test, load_model=model_path, device=DEVICE)

    train_accuracy, train_report = modelABSA.test(
        data, load_model=model_path, device=DEVICE)
    train_pred, train_pol = modelABSA.predict_batch(
        data, load_model=model_path, device=DEVICE)

    # save results
    if not os.path.exists(dir_name_s+'/results'):
        os.makedirs(dir_name_s+'/results')

    # report
    with open(dir_name_s+'/results/test_report_lr{}_epochs{}_batch{}.csv'.format(lr, epochs, batch), 'w') as f:
        for r in test_report.split('\n'):
            f.write(r + '\n')

    with open(dir_name_s+'/results/train_report_lr{}_epochs{}_batch{}.csv'.format(lr, epochs, batch), 'w') as f:
        for r in train_report.split('\n'):
            f.write(r + '\n')

    # predictions
    data_test['Predicted'] = test_pred
    data_test['Actual'] = test_pol
    data_test.to_csv(
        dir_name_s+'/results/test_pred_lr{}_epochs{}_batch{}.csv'.format(lr, epochs, batch), index=False)

    data['Predicted'] = train_pred
    data['Actual'] = train_pol
    data.to_csv(dir_name_s+'/results/train_pred_lr{}_epochs{}_batch{}.csv'.format(lr,
                epochs, batch), index=False)

    # accuracy
    test_accuracy = np.array(test_accuracy)
    train_accuracy = np.array(train_accuracy)

    with open(dir_name_s+'/results/test_accuracy_lr{}_epochs{}_batch{}.csv'.format(lr, epochs, batch), 'w') as f:
        f.write(str(test_accuracy))
    with open(dir_name_s+'/results/train_accuracy_lr{}_epochs{}_batch{}.csv'.format(lr, epochs, batch), 'w') as f:
        f.write(str(train_accuracy))
