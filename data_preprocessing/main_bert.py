"""
Created on Fri May 24 16:24:14 2024

@author: Kazeem
"""


from tqdm import tqdm
import concurrent.futures
import csv
import time
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset
from transformers import BertTokenizer
from bert.bert_model import BertAspectExtraction, BertAspectSentimentAnalysis
from bert.dataset import DatasetAspectExtraction, DatasetAspectSentientAnalysis
from bert.bert_utils import get_classification_report, get_key_polarity, create_mini_batch_ate, create_mini_batch_absa
from utils import save_model, load_model, pre_process, nlp_code, evl_time, read_file

# model location
model_location = "../saved_models/"
# Data location
data_location = "../dataset/raw/"

if not os.path.exists(model_location):
    os.makedirs(model_location)
    print(f"'{model_location}' created.")
else:
    print(f"'{model_location}' exist")

# Device and Model setup
DEVICE = 'mps'
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")  # cuda GPU
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # check mac mps
else:
    DEVICE = torch.device("cpu")  # otherwise cpu
print(f"Using device: {DEVICE}")

pretrain_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(
    pretrain_model_name, max_length=1024, truncation=True)
lr = 2e-5
model_ATE = BertAspectExtraction(pretrain_model_name).to(DEVICE)
optimizer_ATE = torch.optim.Adam(model_ATE.parameters(), lr=lr)
model_ABSA = BertAspectSentimentAnalysis(pretrain_model_name).to(DEVICE)
optimizer_ABSA = torch.optim.Adam(model_ABSA.parameters(), lr=lr)

# Load data for ATE
laptops_train_ds_ate = DatasetAspectExtraction(
    pd.read_csv(f"{data_location}laptops_train.csv"), tokenizer)
laptops_test_ds_ate = DatasetAspectExtraction(
    pd.read_csv(f"{data_location}laptops_test.csv"), tokenizer)
restaurants_train_ds_ate = DatasetAspectExtraction(
    pd.read_csv(f"{data_location}restaurants_train.csv"), tokenizer)
restaurants_test_ds_ate = DatasetAspectExtraction(
    pd.read_csv(f"{data_location}restaurants_test.csv"), tokenizer)
twitter_train_ds_ate = DatasetAspectExtraction(
    pd.read_csv(f"{data_location}twitter_train.csv"), tokenizer)
twitter_test_ds_ate = DatasetAspectExtraction(
    pd.read_csv(f"{data_location}twitter_test.csv"), tokenizer)

# Load data for ABSA
laptops_train_ds_absa = DatasetAspectSentientAnalysis(
    pd.read_csv(f"{data_location}laptops_train.csv"), tokenizer)
laptops_test_ds_absa = DatasetAspectSentientAnalysis(
    pd.read_csv(f"{data_location}laptops_test.csv"), tokenizer)
restaurants_train_ds_absa = DatasetAspectSentientAnalysis(pd.read_csv(f"{data_location}restaurants_train.csv"),
                                                          tokenizer)
restaurants_test_ds_absa = DatasetAspectSentientAnalysis(
    pd.read_csv(f"{data_location}restaurants_test.csv"), tokenizer)
twitter_train_ds_absa = DatasetAspectSentientAnalysis(
    pd.read_csv(f"{data_location}twitter_train.csv"), tokenizer)
twitter_test_ds_absa = DatasetAspectSentientAnalysis(
    pd.read_csv(f"{data_location}twitter_test.csv"), tokenizer)

# Combine the dataset for ate
train_ds_ate = ConcatDataset(
    [laptops_train_ds_ate, restaurants_train_ds_ate, twitter_train_ds_ate])
test_ds_ate = ConcatDataset(
    [laptops_test_ds_ate, restaurants_test_ds_ate, twitter_train_ds_ate])

# Combine the dataset for absa
train_ds_absa = ConcatDataset(
    [laptops_train_ds_absa, restaurants_train_ds_absa, twitter_train_ds_absa])
test_ds_absa = ConcatDataset(
    [laptops_test_ds_absa, restaurants_test_ds_absa, twitter_test_ds_absa])

# load all dataset
train_loader_ate = DataLoader(
    train_ds_ate, batch_size=32, collate_fn=create_mini_batch_ate, shuffle=True)
test_loader_ate = DataLoader(
    test_ds_ate, batch_size=64, collate_fn=create_mini_batch_ate, shuffle=True)
train_loader_absa = DataLoader(
    train_ds_absa, batch_size=32, collate_fn=create_mini_batch_absa, shuffle=True)
test_loader_absa = DataLoader(
    test_ds_absa, batch_size=64, collate_fn=create_mini_batch_absa, shuffle=True)


def train_model_aspect_extraction(loader, epochs):
    all_data = len(loader)
    for epoch in range(epochs):
        finish_data = 0
        losses = []
        current_times = []

        for data in loader:
            t0 = time.time()
            ids_tensors, tags_tensors, _, masks_tensors = data
            ids_tensors = ids_tensors.to(DEVICE)
            tags_tensors = tags_tensors.to(DEVICE)
            masks_tensors = masks_tensors.to(DEVICE)
            loss = model_ATE(
                ids_tensors=ids_tensors, tags_tensors=tags_tensors, masks_tensors=masks_tensors)
            losses.append(loss.item())
            loss.backward()
            optimizer_ATE.step()
            optimizer_ATE.zero_grad()

            finish_data += 1
            current_times.append(round(time.time() - t0, 3))
            current = np.mean(current_times)
            hr, min, sec = evl_time(
                current * (all_data - finish_data) + current * all_data * (epochs - epoch - 1))
            print('epoch:', epoch, " batch:", finish_data, "/", all_data, " loss:", np.mean(losses), " hr:", hr,
                  " min:", min, " sec:", sec)

        save_model(model_ATE, f'{model_location}bert_aspect_extraction.pkl')


def test_model_aspect_extraction(loader):
    pred = []
    trueth = []
    with torch.no_grad():
        for data in loader:
            ids_tensors, tags_tensors, _, masks_tensors = data
            ids_tensors = ids_tensors.to(DEVICE)
            tags_tensors = tags_tensors.to(DEVICE)
            masks_tensors = masks_tensors.to(DEVICE)

            outputs = model_ATE(ids_tensors=ids_tensors,
                                tags_tensors=None, masks_tensors=masks_tensors)

            _, predictions = torch.max(outputs, dim=2)

            pred += list([int(j) for i in predictions for j in i])
            trueth += list([int(j) for i in tags_tensors for j in i])

    return trueth, pred


def predict_model_aspect_extraction(sentence, tokenizer):
    word_pieces = []

    tokens = tokenizer.tokenize(sentence) 
    word_pieces += tokens
    
    #https://stackoverflow.com/questions/70520725/runtimeerror-the-expanded-size-of-the-tensor-585-must-match-the-existing-size
    max_length = 512
    if len(word_pieces) > max_length - 2:
        word_pieces = word_pieces[:max_length - 2]

    ids = tokenizer.convert_tokens_to_ids(word_pieces)
    input_tensor = torch.tensor([ids]).to(DEVICE)
    
    with torch.no_grad():
        outputs = model_ATE(input_tensor, None, None)
        _, predictions = torch.max(outputs, dim=2)

        
    predictions = predictions[0].tolist()

    return word_pieces, predictions, outputs


# Training the aspect extraction
#train_model_aspect_extraction(train_loader_ate, 3)

# Loading the saved model for future use
model_ATE = load_model(
    model_ATE, f'{model_location}bert_aspect_extraction.pkl')


#########################################################
# get classification report, uncomment the below code
# x, y = test_model_aspect_extraction(test_loader_ate)
# classification_report = get_classification_report(x, y)
# print(classification_report)
#########################################################


def train_model_aspect_sentimental_analysis(loader, epochs):
    all_data = len(loader)
    for epoch in range(epochs):
        finish_data = 0
        losses = []
        current_times = []

        for data in loader:
            t0 = time.time()
            ids_tensors, segments_tensors, masks_tensors, label_ids = data
            ids_tensors = ids_tensors.to(DEVICE)
            segments_tensors = segments_tensors.to(DEVICE)
            label_ids = label_ids.to(DEVICE)
            masks_tensors = masks_tensors.to(DEVICE)
            loss = model_ABSA(ids_tensors=ids_tensors, lable_tensors=label_ids, masks_tensors=masks_tensors,
                              segments_tensors=segments_tensors)
            losses.append(loss.item())
            loss.backward()
            optimizer_ABSA.step()
            optimizer_ABSA.zero_grad()

            finish_data += 1
            current_times.append(round(time.time() - t0, 3))
            current = np.mean(current_times)
            hr, min, sec = evl_time(
                current * (all_data - finish_data) + current * all_data * (epochs - epoch - 1))
            print('epoch:', epoch, " batch:", finish_data, "/", all_data, " loss:", np.mean(losses), " hr:", hr,
                  " min:", min, " sec:", sec)

        save_model(
            model_ABSA, f'{model_location}bert_aspect_sentiment_analysis.pkl')


def test_model_aspect_sentimental_analysis(loader):
    pred = []
    trueth = []
    with torch.no_grad():
        for data in loader:
            ids_tensors, segments_tensors, masks_tensors, label_ids = data
            ids_tensors = ids_tensors.to(DEVICE)
            segments_tensors = segments_tensors.to(DEVICE)
            masks_tensors = masks_tensors.to(DEVICE)

            outputs = model_ABSA(
                ids_tensors, None, masks_tensors=masks_tensors, segments_tensors=segments_tensors)

            _, predictions = torch.max(outputs, dim=1)

            pred += list([int(i) for i in predictions])
            trueth += list([int(i) for i in label_ids])

    return trueth, pred


def predict_model_aspect_sentimental_analysis(sentence, aspect, tokenizer):
    t1 = tokenizer.tokenize(sentence)
    t2 = tokenizer.tokenize(aspect)
    
    word_pieces = ['[cls]']
    word_pieces += t1
    word_pieces += ['[sep]']
    word_pieces += t2

    segment_tensor = [0] + [0] * len(t1) + [0] + [1] * len(t2)

    ids = tokenizer.convert_tokens_to_ids(word_pieces)
    input_tensor = torch.tensor([ids]).to(DEVICE)
    segment_tensor = torch.tensor(segment_tensor).to(DEVICE)

    with torch.no_grad():
        outputs = model_ABSA(input_tensor, None, None,
                             segments_tensors=segment_tensor)
        _, predictions = torch.max(outputs, dim=1)

    return word_pieces, predictions, outputs


# Training the aspect based sentiment analysis
#train_model_aspect_sentimental_analysis(train_loader_absa, 6)

# Loading the saved model for future use
model_ABSA = load_model(
    model_ABSA, f'{model_location}bert_aspect_sentiment_analysis.pkl')


# def AspectExtractionSentimentAnalysis(text):
#     terms = []
#     word = ""
#     x, y, z = predict_model_aspect_extraction(text, tokenizer)
#     for i in range(len(y)):
#         if y[i] == 1:
#             if len(word) != 0:
#                 terms.append(word.replace(" ##", ""))
#             word = x[i]
#         if y[i] == 2:
#             word += (" " + x[i])

#     if len(word) != 0:
#         terms.append(word.replace(" ##", ""))

#     sentiment = []
#     if len(terms) != 0:
#         for i in terms:
#             _, c, p = predict_model_aspect_sentimental_analysis(
#                 text, i, tokenizer)
#             sentiment.append(int(c))

#     aspect_key, polarity_key = get_key_polarity(x, terms, sentiment)
#     return x, terms if terms else [], sentiment if sentiment else [], aspect_key, polarity_key


import re

def split_text_into_segments(text, max_segment_length=100):
    # Split long text into sentences or paragraphs
    sentences = re.split(r'(?<=[^A-Z].[.?!])\s+(?=[A-Z])', text)
    
    segments = []
    current_segment = ""
    
    for sentence in sentences:
        if len(current_segment) + len(sentence) <= max_segment_length:
            current_segment += sentence + " "
        else:
            segments.append(current_segment.strip())
            current_segment = sentence + " "
    
    if current_segment:
        segments.append(current_segment.strip())
    
    return segments


def AspectExtractionSentimentAnalysis(text):
    segments = split_text_into_segments(text)
    
    all_terms = []
    all_sentiments = []
    all_aspect_keys = []
    all_polarity_keys = []
    X = []
    
    for segment in segments:
        if not segment:
            continue
        terms = []
        word = ""
        x, y, z = predict_model_aspect_extraction(segment, tokenizer)
        
        for i in range(len(y)):
            if y[i] == 1:
                if len(word) != 0:
                    terms.append(word.replace(" ##", ""))
                word = x[i]
            if y[i] == 2:
                word += (" " + x[i])
        
        if len(word) != 0:
            terms.append(word.replace(" ##", ""))
        
        sentiment = []
        if len(terms) != 0:
            for i in terms:
                _, c, p = predict_model_aspect_sentimental_analysis(segment, i, tokenizer)
                sentiment.append(int(c))
        
        aspect_key, polarity_key = get_key_polarity(x, terms, sentiment)
        
        all_terms.extend(terms)
        all_sentiments.extend(sentiment)
        all_aspect_keys.append(aspect_key)
        all_polarity_keys.append(polarity_key)
        X.append(x)
    
    return X, all_terms, all_sentiments, all_aspect_keys, all_polarity_keys


# Loading the models again for use
# model_ABSA = load_model(
#     model_ABSA, f'{model_location}bert_aspect_sentiment_analysis.pkl')
# model_ATE = load_model(
#     model_ATE, f'{model_location}bert_aspect_extraction.pkl')


positive_in_file_path = f"{data_location}train_positive_reviews.txt"
negative_in_file_path = f"{data_location}train_negative_reviews.txt"
pos_test_in_file_path = f"{data_location}test_positive_reviews.txt"
neg_test_in_file_path = f"{data_location}test_negative_reviews.txt"
pos_val_in_file_path = f"{data_location}val_positive_reviews.txt"
neg_val_in_file_path = f"{data_location}val_negative_reviews.txt"

positive_out_file_path = f"{data_location}train_negative_reviews_labelled_pr_bert.csv"
negative_out_file_path = f"{data_location}train_negative_reviews_labelled_pr_bert.csv"
pos_test_out_file_path = f"{data_location}test_negative_reviews_labelled_pr_bert.csv"
neg_test_out_file_path = f"{data_location}test_negative_reviews_labelled_pr_bert.csv"
pos_val_out_file_path = f"{data_location}val_negative_reviews_labelled_pr_bert.csv"
neg_val_out_file_path = f"{data_location}val_negative_reviews_labelled_pr_bert.csv"



positive_reviews = read_file(positive_in_file_path)
negative_reviews = read_file(negative_in_file_path)
pos_test = read_file(pos_test_in_file_path)
neg_test = read_file(neg_test_in_file_path)
pos_val = read_file(pos_val_in_file_path)
neg_val = read_file(neg_val_in_file_path)

reviews_list = [positive_reviews, negative_reviews,pos_test, neg_test,pos_val_in_file_path, neg_val_in_file_path]
output_list = [positive_out_file_path, negative_out_file_path,pos_test_out_file_path, neg_test_out_file_path,pos_val_out_file_path, neg_val_out_file_path]
# process imdb
print('Starting process imdb')


def process_review(review):
    review = pre_process(nlp_code(review))
    tokens, aspects, sentiments, aspect_key, polarity_key = AspectExtractionSentimentAnalysis(
        review)
    return [tokens, aspects, aspect_key, polarity_key]


def write_to_csv(output_file_path, data):
    with open(output_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Review', 'Aspects', 'Aspect Key', 'Sentiments'])
        csv_writer.writerows(data)


# def process_reviews(reviews, output_file_path):
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = []
#         for review in reviews:
#             future = executor.submit(process_review, review)
#             futures.append(future)

#         data = []
#         for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
#             result = future.result()
#             data.append(result)

#     write_to_csv(output_file_path, data)

def process_reviews(reviews, output_file_path, batch_size=30):
    with open(output_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Review', 'Aspects', 'Aspect Key', 'Sentiments'])

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        data = []
        for review in reviews:
            future = executor.submit(process_review, review)
            futures.append(future)

            if len(futures) == batch_size:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    data.append(result)
                write_to_csv(output_file_path, data)
                futures = []
                data = []

        if futures:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                data.append(result)
            write_to_csv(output_file_path, data)

# total
# train = pos + neg = 20k
# test = pos + neg = 25k
# val = pos + neg = 5k
# Total = 50k

for reviews, output_file_path in zip(reviews_list, output_list):
    process_reviews(reviews, output_file_path)

# for reviews, output_file_path in zip(reviews_list, output_list):

#     with open(output_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
#         csv_writer = csv.writer(csv_file)
#         csv_writer.writerow(['Review', 'Aspects', 'Aspect Key', 'Sentiments'])
#         data = []
#         for review in tqdm(reviews):
#             review = pre_process(nlp_code(review))
#             tokens, aspects, sentiments, aspect_key, polarity_key = AspectExtractionSentimentAnalysis(
#                 review)
#             data.append([tokens, aspects, aspect_key, polarity_key])
#             csv_writer.writerows(data)
#         csv_file.close()
