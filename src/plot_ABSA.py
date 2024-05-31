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
    dataframes = []
    dftest = pd.DataFrame()
    csv_files = ['model_ABSA_adapter/results/test_pred_lr3.0000000000000004e-05_epochs5_batch8.csv', 'model_ABSA_adapter_scheduler/results/test_pred_lr3.0000000000000004e-05_epochs5_batch8.csv','model_ABSA/results/test_pred_lr3.0000000000000004e-05_epochs5_batch8.csv', 'model_ABSA_scheduler/results/test_pred_lr3.0000000000000004e-05_epochs5_batch8.csv']
    for csv_path in csv_files:
        data = pd.read_csv(csv_path)
        dataframes.append(data)
    dftest = pd.concat(dataframes, ignore_index=True)

    # dftest = pd.read_csv('model_ABSA_adapter/results/test_pred_lr3.0000000000000004e-05_epochs5_batch8.csv')
    # dftest = pd.read_csv('model_ABSA_adapter_scheduler/results/test_pred_lr3.0000000000000004e-05_epochs5_batch8.csv')
    # dftest = pd.read_csv('model_ABSA/results/test_pred_lr3.0000000000000004e-05_epochs5_batch8.csv')
    # dftest = pd.read_csv('model_ABSA_scheduler/results/test_pred_lr3.0000000000000004e-05_epochs5_batch8.csv')
    test_pred = dftest['Predicted']
    
    #load
    data_test = pd.read_csv(TEST_DATA_PATH)
    
    test_tags_real = [t.strip('][').split(', ') for t in data_test['Tags']]
    test_tags_real = [[int(i) for i in t] for t in test_tags_real]
    
    test_pred = [t.strip('][').split(', ') for t in test_pred]
    test_pred = [[int(i) for i in t] for t in test_pred]
    
    
    ABSA_data = tag_to_word_df(data_test, 'gold terms', test_tags_real)
    ABSA_data = tag_to_word_df(ABSA_data, 'pred terms', test_pred)
    return ABSA_data

def plot():
    # training loss and the validation loss for the fine-tuning and adapter cases, both with and without scheduler.

    lossABSA = np.loadtxt(
        'model_ABSA/losses_lr3.0000000000000004e-05_epochs3_batch16.txt')
    lossABSA_AS = np.loadtxt(
        'model_ABSA_adapter_scheduler/losses_lr3.0000000000000004e-05_epochs3_batch16.txt')
    lossABSA_S = np.loadtxt(
        'model_ABSA_scheduler/losses_lr3.0000000000000004e-05_epochs3_batch16.txt')
    lossABSA_A = np.loadtxt(
        'model_ABSA_adapter/losses_lr3.0000000000000004e-05_epochs3_batch16.txt')
    
    sns.set_theme (style="white", rc={"lines.linewidth": 3}, font_scale=1.5, palette="Dark2")
    fig, ax = plt.subplots(figsize=(18,5))


    sns.lineplot(x=range(len(lossABSA)), y=lossABSA,
                    ax=ax, label='Fine tuning')
    sns.lineplot(x=range(len(lossABSA_S)), y=lossABSA_S,
                    ax=ax, label='Fine tuning +\nscheduler')
    sns.lineplot(x=range(len(lossABSA_A)), y=lossABSA_A,
                    ax=ax, label='Adapter')
    sns.lineplot(x=range(len(lossABSA_AS)), y=lossABSA_AS,
                    ax=ax, label='Adapter +\nscheduler')


    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    if not os.path.isdir('results_ABSA'):
        os.makedirs('results_ABSA')

    fig.savefig('results_ABSA/loss_lr{:.5f}_epochs{}_batch{}.pdf'.format(lr, epochs, batch), dpi=300, bbox_inches='tight')
    # sns.set_theme(style="white", rc={
    #               "lines.linewidth": 3}, font_scale=1.5, palette="Dark2")
    # fig, ax = plt.subplots(1, 2, figsize=(20, 5))

    # for i in [0, 1]:
    #     sns.lineplot(x=range(len(lossABSA)), y=lossABSA,
    #                  ax=ax[i], label='Fine tuning')
    #     sns.lineplot(x=range(len(lossABSA_S)), y=lossABSA_S,
    #                  ax=ax[i], label='Fine tuning +\nscheduler')
    #     sns.lineplot(x=range(len(lossABSA_A)), y=lossABSA_A,
    #                  ax=ax[i], label='Adapter')
    #     sns.lineplot(x=range(len(lossABSA_AS)), y=lossABSA_AS,
    #                  ax=ax[i], label='Adapter +\nscheduler')

    #     ax[i].set_xlabel('iteration')
    #     ax[i].set_ylabel('loss')
    # ax[1].set_yscale('log')
    # ax[1].set_ylabel('log loss')
    # ax[0].legend().set_visible(False)
    # ax[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # if not os.path.isdir('results_ABSA'):
    #     os.makedirs('results_ABSA')
    # fig.savefig('results_ABSA/loss_lr{:.5f}_epochs{}_batch{}.pdf'.format(
    #     lr, epochs, batch), dpi=300, bbox_inches='tight')
    
def report():
    test_ATE = classification_report_read('model_ABSA/results/test_report_lr3.0000000000000004e-05_epochs5_batch8.csv')
    train_ATE = classification_report_read('model_ABSA/results/train_report_lr3.0000000000000004e-05_epochs5_batch8.csv')
    print_aligned(test_ATE, train_ATE, 'TEST FINE-TUNING', 'TRAIN FINE-TUNING')
    
    test_ATE_S = classification_report_read('model_ABSA_scheduler/results/test_report_lr3.0000000000000004e-05_epochs5_batch8.csv')
    train_ATE_S = classification_report_read('model_ABSA_scheduler/results/train_report_lr3.0000000000000004e-05_epochs5_batch8.csv')
    print_aligned(test_ATE_S, train_ATE_S, 'TEST FINE-TUNING + SCHEDULER', 'TRAIN FINE-TUNING + SCHEDULER')
    
    test_ATE_A = classification_report_read('model_ABSA_adapter/results/test_report_lr3.0000000000000004e-05_epochs5_batch8.csv')
    train_ATE_A = classification_report_read('model_ABSA_adapter/results/train_report_lr3.0000000000000004e-05_epochs5_batch8.csv')
    print_aligned(test_ATE_A, train_ATE_A, 'TEST ADAPTER', 'TRAIN ADAPTER')
    
    test_ATE_AS = classification_report_read('model_ABSA_adapter_scheduler/results/test_report_lr3.0000000000000004e-05_epochs5_batch8.csv')
    train_ATE_AS = classification_report_read('model_ABSA_adapter_scheduler/results/train_report_lr3.0000000000000004e-05_epochs5_batch8.csv')
    print_aligned(test_ATE_AS, train_ATE_AS, 'TEST ADAPTER + SCHEDULER', 'TRAIN ADAPTER + SCHEDULER')
    

    
def word_cloud (data):
    from wordcloud import WordCloud
    wordcloud = WordCloud( collocations=False,
                          background_color="cornflowerblue",
                          colormap="rainbow",
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
    ABSA_data = _get_df()
    gold_terms = ABSA_data['gold terms'].values.flatten().tolist()
    gold_terms = list(itertools.chain(*gold_terms))
    
    pred_terms = ABSA_data['pred terms'].values.flatten().tolist()
    pred_terms = list(itertools.chain(*pred_terms))
    target_predicted_wordcloud(' '.join(gold_terms), ' '.join(pred_terms), "results_ABSA/adapter_extracted_terms_wordcloud_.pdf")

def get_example(coun=10):
    ABSA_data = _get_df()
    return ABSA_data.head(coun)

# confusion matrix 
def plot_confusion_matrix(predictions, labels, title, ax, 
                          cmap='BuPu'):
    """
    This function prints and plots the confusion matrix.
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, predictions, normalize = 'true')
    cm = cm[1:,1:]
    sns.set_theme (style="white", rc={"lines.linewidth": 3}, font_scale=1.5)
    sns.heatmap(cm, annot=True,  cmap=cmap, ax=ax, cbar=False)
    ax.set_title(title)
    ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
    ax.set_yticklabels(['Negative', 'Neutral', 'Positive'])
    ax.set_ylabel('Ground Truth'),ax.set_xlabel('Predicted')
    
def plot_confusion_matrix_df(df_path, title, ax, cmap='BuPu'):

    df = pd.read_csv(df_path)
    pred = df['Predicted'].apply(lambda x: [int (i) if i !='None' else -2 for i in x.strip('][').split(', ')]).to_list()
    gtruth = df['Actual'].apply(lambda x: [int (i) if i !='None' else -2 for i in x.strip('][').split(', ')]).to_list()
    predicted, ground_truth = [], []
    for i in range(len(pred)):
        predicted+=pred[i]
        ground_truth+=gtruth[i]
    plot_confusion_matrix(predicted, ground_truth, title, ax, cmap=cmap)

def compare_confusion_mat():
    df = 'model_ABSA/results/test_pred_lr1e-05_epochs5_batch8.csv'
    dfS = 'model_ABSA_scheduler/results/test_pred_lr1e-05_epochs5_batch8.csv'
    dfA = 'model_ABSA_adapter/results/test_pred_lr1e-05_epochs5_batch8.csv'
    dfAS = 'model_ABSA_adapter_scheduler/results/test_pred_lr1e-05_epochs5_batch8.csv'
    fig, ax = plt.subplots(2,2,figsize=(15,15))
    #set space between subplots
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    fig.suptitle('Confusion matrices')
    plot_confusion_matrix_df(df, r'$\bf{FINE-TUNING}$', ax[0][0])
    plot_confusion_matrix_df(dfS, r'$\bf{FINE-TUNING + SCHEDULING}$', ax[0][1])
    plot_confusion_matrix_df(dfA, r'$\bf{ADAPTER}$', ax[1][0])
    plot_confusion_matrix_df(dfAS, r'$\bf{ADAPTER + SCHEDULING}$', ax[1][1])
    
    fig.savefig('results_ABSA/CMatrix_test.png', dpi=300, bbox_inches='tight')
