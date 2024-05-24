"""
Created on Fri May 24 16:24:14 2024

@author: Kazeem
"""
import csv

import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

nlp = spacy.load('en_core_web_sm')
# Preprocess the documents
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def nlp_code(text):
    nlp_text = nlp(text)
    return str(nlp_text)


def pre_process(text):
    tokens = [word for word in nltk.word_tokenize(text.lower()) if word.isalpha()]
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def evl_time(t):
    min, sec = divmod(t, 60)
    hr, min = divmod(min, 60)
    return int(hr), int(min), int(sec)


def load_model(model, path):
    model.load_state_dict(torch.load(path), strict=False)
    return model


def save_model(model, name):
    torch.save(model.state_dict(), name)


# Read the review file
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as txt_file:
        review_document = [line.strip() for line in txt_file if line.strip()]
        txt_file.close()
    return review_document
