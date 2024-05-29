#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:05:14 2024

@author: wangyifei
"""

import pandas as pd
import numpy as np
from absa import ABSAModel
from abte import ABTEModel
import torch
from transformers import BertTokenizer
import os
import fire
# save results


def run_ABSA_test_train(work_type, adapter, lr_schedule):
    from consts import lr, epochs, batch, TEST_DATA_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH
    
    if adapter:
        if lr_schedule:
            dir_name = "model_{}_adapter_scheduler".format(work_type)
        else:
            dir_name = "model_{}_adapter".format(work_type)
    else:
        if lr_schedule:
            dir_name = "model_{}_scheduler".format(work_type)
        else:
            dir_name = "model_{}".format(work_type)

    # load
    data = pd.read_csv(VAL_DATA_PATH)
    data_test = pd.read_csv(TEST_DATA_PATH)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")  # cuda GPU
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")  # check mac mps
    else:
        DEVICE = torch.device("cpu")  # otherwise cpu
    print(f"Test using device: {DEVICE}")

    # define model
    modelABSA = None
    if work_type == 'ABSA':
        modelABSA = ABSAModel(tokenizer, adapter=adapter)
        model_path = dir_name+'/model_lr3.0000000000000004e-05_epochs2_batch16.pkl'
        lr = 1e-5
    elif work_type == 'ABTE':
        modelABSA = ABTEModel(tokenizer, adapter=adapter)
        model_path = dir_name+'/model_lr3.0000000000000004e-05_epochs2_batch16.pkl'
    else:
        raise Exception('wrong work type, must be ABSA or ABTE')
    
    # save results
    if not os.path.exists(dir_name+'/results'):
        os.makedirs(dir_name+'/results')

    print(model_path)
    # train_accuracy, train_report = modelABSA.test(
    #     data, load_model=model_path, device=DEVICE)
    # test_accuracy, test_report = modelABSA.test(
    #     data_test, load_model=model_path, device=DEVICE)
    # #accuracy
    # test_accuracy = np.array(test_accuracy)
    # train_accuracy = np.array(train_accuracy)

    # with open(dir_name+'/results/test_accuracy_lr{}_epochs{}_batch{}.csv'.format(lr, epochs, batch), 'w') as f:
    #      f.write(str(test_accuracy))

    #     #report
    # with open(dir_name+'/results/test_report_lr{}_epochs{}_batch{}.csv'.format(lr, epochs, batch), 'w') as f:
    #     for r in test_report.split('\n'):
    #         f.write(r + '\n')

    # with open(dir_name+'/results/train_report_lr{}_epochs{}_batch{}.csv'.format(lr, epochs, batch), 'w') as f:
    #     for r in train_report.split('\n'):
    #         f.write(r + '\n')
    # with open(dir_name+'/results/train_accuracy_lr{}_epochs{}_batch{}.csv'.format(lr, epochs, batch), 'w') as f:
    #     f.write(str(train_accuracy))

    # DEVICE = 'cpu' #otherwise 'RuntimeError: Placeholder storage has not been allocated on MPS device!' in mac
    print(adapter, lr_schedule, DEVICE)
    train_pred, train_pol = modelABSA.predict_batch(
        data, load_model=model_path, device=DEVICE)
    test_pred, test_pol = modelABSA.predict_batch(
        data_test, load_model=model_path, device=DEVICE)
    
    # predictions
    data_test['Predicted'] = test_pred
    data_test['Actual'] = test_pol
    data_test.to_csv(
        dir_name+'/results/test_pred_lr{}_epochs{}_batch{}.csv'.format(lr, epochs, batch), index=False)

    data['Predicted'] = train_pred
    data['Actual'] = train_pol
    data.to_csv(dir_name+'/results/train_pred_lr{}_epochs{}_batch{}.csv'.format(lr,
                epochs, batch), index=False)

def prediction(work_type, adapter=True, lr_schedule=False):
    """Predict the model.

     Args:
         work_type (str): Training for which task choices: ['ABTE', 'ABSA']
         lr_schedule (bool): Whether to use learning rate scheduling (default: False).
         adapter (bool): Whether to use Adapter(default: True).
     """
    run_ABSA_test_train(work_type, adapter=adapter, lr_schedule=lr_schedule)
    
    
if __name__ == '__main__':
    fire.Fire(prediction)
    print('Done')
