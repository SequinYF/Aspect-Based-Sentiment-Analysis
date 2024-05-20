#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 20:13:31 2024

@author: wangyifei
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import itertools
from consts import *
from utils import *


def _get_df():
    # adapter training 
    dftest = pd.read_csv('model_ABTE_adapter/results/test_pred_lr3.0000000000000004e-05_epochs5_batch8.csv')
    test_pred = dftest['Predicted']
    
    #load
    data = pd.read_csv(TRAIN_DATA_PATH)
    data_test = pd.read_csv(TEST_DATA_PATH)
    
    test_tags_real = [t.strip('][').split(', ') for t in data_test['Tags']]
    test_tags_real = [[int(i) for i in t] for t in test_tags_real]
    
    test_pred = [t.strip('][').split(', ') for t in test_pred]
    test_pred = [[int(i) for i in t] for t in test_pred]
    
    
    ABTE_data = tag_to_word_df(data_test, 'gold terms', test_tags_real)
    ABTE_data = tag_to_word_df(ABTE_data, 'pred terms', test_pred)
    return ABTE_data

def plot():
    # training loss and the validation loss for the fine-tuning and adapter cases, both with and without scheduler.

    lossABTE = np.loadtxt(
        'model_ABTE/losses_lr3.0000000000000004e-05_epochs5_batch8.txt')
    lossABTE_AS = np.loadtxt(
        'model_ABTE_adapter_scheduler/losses_lr3.0000000000000004e-05_epochs5_batch8.txt')
    lossABTE_S = np.loadtxt(
        'model_ABTE_scheduler/losses_lr3.0000000000000004e-05_epochs5_batch8.txt')
    lossABTE_A = np.loadtxt(
        'model_ABTE_adapter/losses_lr3.0000000000000004e-05_epochs5_batch8.txt')

    sns.set_theme(style="white", rc={
                  "lines.linewidth": 3}, font_scale=1.5, palette="Dark2")
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))

    for i in [0, 1]:
        sns.lineplot(range(len(lossABTE)), lossABTE,
                     ax=ax[i], label='Fine tuning')
        sns.lineplot(range(len(lossABTE_S)), lossABTE_S,
                     ax=ax[i], label='Fine tuning +\nscheduler')
        sns.lineplot(range(len(lossABTE_A)), lossABTE_A,
                     ax=ax[i], label='Adapter')
        sns.lineplot(range(len(lossABTE_AS)), lossABTE_AS,
                     ax=ax[i], label='Adapter +\nscheduler')

        ax[i].set_xlabel('iteration')
        ax[i].set_ylabel('loss')
    ax[1].set_yscale('log')
    ax[1].set_ylabel('log loss')
    ax[0].legend().set_visible(False)
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    if not os.path.isdir('results_ABTE'):
        os.makedirs('results_ABTE')
    fig.savefig('results_ABTE/loss_lr{:.5f}_epochs{}_batch{}.pdf'.format(
        lr, epochs, batch), dpi=300, bbox_inches='tight')
    
def report():
    test_ATE = classification_report_read('model_ABTE/results/test_report_lr3.0000000000000004e-05_epochs5_batch8.csv')
    train_ATE = classification_report_read('model_ABTE/results/train_report_lr3.0000000000000004e-05_epochs5_batch8.csv')
    print_aligned(test_ATE, train_ATE, 'TEST FINE-TUNING', 'TRAIN FINE-TUNING')
    
    test_ATE_S = classification_report_read('model_ABTE_scheduler/results/test_report_lr3.0000000000000004e-05_epochs5_batch8.csv')
    train_ATE_S = classification_report_read('model_ABTE_scheduler/results/train_report_lr3.0000000000000004e-05_epochs5_batch8.csv')
    print_aligned(test_ATE_S, train_ATE_S, 'TEST FINE-TUNING + SCHEDULER', 'TRAIN FINE-TUNING + SCHEDULER')
    
    test_ATE_A = classification_report_read('model_ABTE_adapter/results/test_report_lr3.0000000000000004e-05_epochs5_batch8.csv')
    train_ATE_A = classification_report_read('model_ABTE_adapter/results/train_report_lr3.0000000000000004e-05_epochs5_batch8.csv')
    print_aligned(test_ATE_A, train_ATE_A, 'TEST ADAPTER', 'TRAIN ADAPTER')
    
    test_ATE_AS = classification_report_read('model_ABTE_adapter_scheduler/results/test_report_lr3.0000000000000004e-05_epochs5_batch8.csv')
    train_ATE_AS = classification_report_read('model_ABTE_adapter_scheduler/results/train_report_lr3.0000000000000004e-05_epochs5_batch8.csv')
    print_aligned(test_ATE_AS, train_ATE_AS, 'TEST ADAPTER + SCHEDULER', 'TRAIN ADAPTER + SCHEDULER')
    

    
def word_cloud (data):
    from wordcloud import WordCloud
    wordcloud = WordCloud( collocations=False,
                          background_color="cornflowerblue",
                          colormap="magma",
                          max_words=50).generate(data)

    return wordcloud

def target_predicted_wordcloud(targets, predicted, file_name):
    
    sns.set_theme(style='white', font_scale=2)
    fig, ax = plt.subplots(1, 2, figsize=(22, 6))
    ax[0].imshow(word_cloud(targets))
    ax[0].axis("off")
    ax[0].set_title("Target")
    ax[1].imshow(word_cloud(predicted))
    ax[1].axis("off")
    ax[1].set_title("Predicted")
    fig.savefig(file_name, dpi=300, bbox_inches='tight')
    
def gen_word_cloud():
    ABTE_data = _get_df()
    gold_terms = ABTE_data['gold terms'].values.flatten().tolist()
    gold_terms = list(itertools.chain(*gold_terms))
    
    pred_terms = ABTE_data['pred terms'].values.flatten().tolist()
    pred_terms = list(itertools.chain(*pred_terms))
    target_predicted_wordcloud(' '.join(gold_terms), ' '.join(pred_terms), "results_ABTE/adapter_extracted_terms_wordcloud.pdf")

def get_example(coun=10):
    ABTE_data = _get_df()
    return ABTE_data.head(20)