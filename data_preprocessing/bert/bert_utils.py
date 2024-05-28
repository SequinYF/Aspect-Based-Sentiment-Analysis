"""
Created on Fri May 24 16:24:14 2024

@author: Kazeem
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import classification_report


def create_mini_batch_ate(samples):
    ids_tensors = [s[1] for s in samples]
    ids_tensors = pad_sequence(ids_tensors, batch_first=True)

    tags_tensors = [s[2] for s in samples]
    tags_tensors = pad_sequence(tags_tensors, batch_first=True)

    pols_tensors = [s[3] for s in samples]
    pols_tensors = pad_sequence(pols_tensors, batch_first=True)

    masks_tensors = torch.zeros(ids_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(ids_tensors != 0, 1)

    return ids_tensors, tags_tensors, pols_tensors, masks_tensors


def create_mini_batch_absa(samples):
    ids_tensors = [s[1] for s in samples]
    ids_tensors = pad_sequence(ids_tensors, batch_first=True)

    segments_tensors = [s[2] for s in samples]
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)

    label_ids = torch.stack([s[3] for s in samples])

    masks_tensors = torch.zeros(ids_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(ids_tensors != 0, 1)

    return ids_tensors, segments_tensors, masks_tensors, label_ids


def get_key_polarity(tokens, terms, sentiment):
    list_len = len(tokens)
    aspect_list = [0 for aspect in range(list_len)]
    sentiment_list = [-1 for aspect in range(list_len)]

    sentiment_use = 0
    for member in terms:
        if member in tokens:
            member_split = member.strip().split(' ')
            split_len = len(member_split)
            sentiment_to_use = sentiment[sentiment_use]
            if split_len > 1:
                for m in member_split:
                    if m == member_split[-1]:
                        ind = tokens.index(m)
                        aspect_list[ind] = 2
                        sentiment_list[ind] = sentiment_to_use
                    else:
                        ind = tokens.index(m)
                        aspect_list[ind] = 1
                        sentiment_list[ind] = sentiment_to_use
            else:
                if member_split[0] in tokens:
                    ind = tokens.index(member_split[0])
                    aspect_list[ind] = 1
                    sentiment_list[ind] = sentiment_to_use
            sentiment_use = sentiment_use + 1
    return aspect_list, sentiment_list


def get_classification_report(x, y):
    target_names = [str(i) for i in range(3)]
    report = classification_report(x, y, target_names)
    return report
