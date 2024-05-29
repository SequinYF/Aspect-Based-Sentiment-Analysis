"""
Created on Fri May 24 16:24:14 2024

@author: Kazeem
"""

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
from gensim.models import CoherenceModel
from gensim import corpora, models
import spacy


# Download necessary NLTK data files
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

# Preprocess the documents
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def pre_process(text):
    tokens = [word for word in nltk.word_tokenize(text.lower()) if word.isalpha()]
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    #return ' '.join(tokens)
    return tokens

def get_processed_doc(full_document):
    processed_docs = [pre_process(doc) for doc in full_document]
    return processed_docs


def get_dictionary_corpus(processed_doc):
    # Create a dictionary representation of the documents
    dictionary = corpora.Dictionary(processed_doc)

    # Convert document into the bag-of-words format
    corpus = [dictionary.doc2bow(doc) for doc in processed_doc]
    return dictionary, corpus


def compute_coherence_values(dictionary, corpus, processed_doc, start):
    coherence_values = []
    model_list = []
    start, limit, step = start, 10, 1
    for num_topics in range(start, limit, step):
        model = gensim.models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=15
        )
        model_list.append(model)
        coherencemodel = CoherenceModel(
            model=model,
            texts=processed_doc,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


def get_model(coherence_value, start):
    # Get optimal model
    model_list, coherence_values = coherence_value
    optimal_model = model_list[coherence_values.index(max(coherence_values))]
    optimal_num_topics = coherence_values.index(max(coherence_values)) + start
    return optimal_model, optimal_num_topics


def get_top_words(model):
    optimal_model = model
    topics = optimal_model.print_topics(-1)
    top_words = {}
    for idx, topic in topics:
        top_words[idx] = [word.split('"')[1] for word in topic.split(' + ')]
    return top_words


def extract_aspects(text):
    doc = nlp(text)
    aspects = []
    for token in doc:
        if token.pos_ in ("NOUN") and token.dep_ in ("nsubj", "dobj", "attr", "pobj"):
            aspects.append(token.lemma_)
    return aspects


def find_aspects1(text, top_words):
    aspect_sentences = {idx: [] for idx in top_words.keys()}
    for idx, words in top_words.items():
        for word in words:
            if word in text.lower():
                aspect_sentences[idx].append((word, text))
    return aspect_sentences


def find_aspects(text, top_words):
    aspect_sentences = {idx: [] for idx in top_words.keys()}
    text_aspects = extract_aspects(text)
    for idx, words in top_words.items():
        for word in words:
            if word in text_aspects:
                aspect_sentences[idx].append((word, text))
    return aspect_sentences


def aggregate_sentiments(sentiments):
    # Calculate average score for positive and negative sentiments
    positive_scores = [s['score'] for s in sentiments if s['label'] == 'POSITIVE']
    negative_scores = [s['score'] for s in sentiments if s['label'] == 'NEGATIVE']

    if not positive_scores and not negative_scores:
        return {'label': 'NEUTRAL', 'score': 1.0}

    avg_positive = sum(positive_scores) / len(positive_scores) if positive_scores else 0
    avg_negative = sum(negative_scores) / len(negative_scores) if negative_scores else 0

    # Determine the dominant sentiment
    if avg_positive >= avg_negative:
        return {'label': 'POSITIVE', 'score': avg_positive}
    else:
        return {'label': 'NEGATIVE', 'score': avg_negative}
