#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 20:13:31 2024

@author: wangyifei
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from consts import *


def plot_():
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
