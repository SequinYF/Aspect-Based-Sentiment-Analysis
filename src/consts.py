'''
Author: Yifei Wang
Github: ephiewangyf@gmail.com
Date: 2024-05-25 12:12:07
LastEditors: ephie && ephiewangyf@gmail.com
LastEditTime: 2024-05-29 18:20:53
FilePath: /Aspect-Based-Sentiment-Analysis/src/consts.py
Description: 
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 20:21:23 2024

@author: wangyifei
"""


batch = 8
lr = 3*1e-5
epochs = 5

TRAIN_DATA_PATH = '../IMDB/lda_train_reviews_labelled_pr_bert.csv'
VAL_DATA_PATH = '../IMDB/lda_train_reviews_labelled_pr_bert.csv'
TEST_DATA_PATH = '../IMDB/lda_test_reviews_labelled_pr_bert.csv'