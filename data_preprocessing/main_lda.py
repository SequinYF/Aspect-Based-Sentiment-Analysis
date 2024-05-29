"""
Created on Fri May 24 16:24:14 2024

@author: Kazeem
"""
import csv
from transformers import pipeline

from lda_utils import *
from utils import pre_process, nlp_code, read_file
import os
from tqdm import tqdm
import torch

# model location
model_location = "../saved_models/"
# Data location
data_location = "../dataset/raw/"


if not os.path.exists(model_location):
    os.makedirs(model_location)
    print(f"'{model_location}' created.")
else:
    print(f"'{model_location}' exist")

sentiment_analyzer = pipeline('sentiment-analysis')
class DetectAspect:

    def __init__(self, review_document):
        self.document = review_document
        self.start = 2
        self.processed_doc = get_processed_doc(self.document)
        self.dictionary, self.corpus = get_dictionary_corpus(
            self.processed_doc)
        self.coherence_values = compute_coherence_values(
            self.dictionary, self.corpus, self.processed_doc, self.start)
        self.optimal_model, self.optimal_num_topics = get_model(
            self.coherence_values, self.start)
        self.top_words = get_top_words(self.optimal_model)

    def train(self, text):
        top_words = self.top_words
        aspect_sentences = find_aspects(text, top_words)
        # Analyze sentiment for each aspect
        aspect_sentiments = {}
        for idx, word_sentences in aspect_sentences.items():
            for word, sentence in word_sentences:
                if len(sentence) > 512:
                    sentence = sentence[:512]
                sentiment = sentiment_analyzer(sentence)[0]
                if word not in aspect_sentiments:
                    aspect_sentiments[word] = []
                aspect_sentiments[word].append(sentiment)
        aggregated_sentiments = {aspect: aggregate_sentiments(sentiments) for aspect, sentiments in
                                 aspect_sentiments.items()}
        return aggregated_sentiments


def label():
    # sample review or use the commented code below to load file
    #output_list = ['1.csv', '2.csv']
    positive_in_file_path = f"{data_location}train_positive_reviews.txt"
    negative_in_file_path = f"{data_location}train_negative_reviews.txt"
    pos_test_in_file_path = f"{data_location}test_positive_reviews.txt"
    neg_test_in_file_path = f"{data_location}test_negative_reviews.txt"
    pos_val_in_file_path = f"{data_location}val_positive_reviews.txt"
    neg_val_in_file_path = f"{data_location}val_negative_reviews.txt"

    positive_out_file_path = f"{data_location}lda_train_positive_reviews_labelled_pr_bert.csv"
    negative_out_file_path = f"{data_location}lda_train_negative_reviews_labelled_pr_bert.csv"
    pos_test_out_file_path = f"{data_location}lda_test_positive_reviews_labelled_pr_bert.csv"
    neg_test_out_file_path = f"{data_location}lda_test_negative_reviews_labelled_pr_bert.csv"
    pos_val_out_file_path = f"{data_location}lda_val_positive_reviews_labelled_pr_bert.csv"
    neg_val_out_file_path = f"{data_location}lda_val_negative_reviews_labelled_pr_bert.csv"



    positive_reviews = read_file(positive_in_file_path)
    negative_reviews = read_file(negative_in_file_path)
    pos_test = read_file(pos_test_in_file_path)
    neg_test = read_file(neg_test_in_file_path)
    pos_val = read_file(pos_val_in_file_path)
    neg_val = read_file(neg_val_in_file_path)

    # reviews_list = [positive_reviews, negative_reviews,pos_test, neg_test,pos_val, neg_val]
    # output_list = [positive_out_file_path, negative_out_file_path,pos_test_out_file_path, neg_test_out_file_path,pos_val_out_file_path, neg_val_out_file_path]
    reviews_list = [neg_val]
    output_list = [neg_val_out_file_path]

    print('start.......')

    # Initialize class
    for reviews, output_file_path in zip(reviews_list, output_list):
        with open(output_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Review', 'Aspects', 'Sentiments'])
            detector = DetectAspect(reviews)
            for review in tqdm(reviews):
                detection = detector.train(review)
                aspects = []
                sentiments = []
                data = []
                for aspect, sentiment in detection.items():
                    aspects.append(aspect)
                    sentiments.append(sentiment['label'])
                data.append([review, aspects, sentiments])
                csv_writer.writerows(data)
            csv_file.close()

