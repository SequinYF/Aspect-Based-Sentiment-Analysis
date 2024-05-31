"""
Created on Fri May 24 16:24:14 2024

@author: Kazeem
"""
import torch
from transformers import BertModel, BertTokenizer


# Bert Module
class BertAspectExtraction(torch.nn.Module):
    def __init__(self, pretrain_model):
        super(BertAspectExtraction, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model, max_length=1024)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 3)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, ids_tensors, tags_tensors, masks_tensors):
        bert_outputs = self.bert(input_ids=ids_tensors, attention_mask=masks_tensors).to_tuple()
        linear_outputs = self.linear(bert_outputs[0])
        if tags_tensors is not None:
            tags_tensors = tags_tensors.view(-1)
            linear_outputs = linear_outputs.view(-1,3)
            loss = self.loss_fn(linear_outputs, tags_tensors)
            return loss
        else:
            return linear_outputs


class BertAspectSentimentAnalysis(torch.nn.Module):
    def __init__(self, pretrain_model):
        super(BertAspectSentimentAnalysis, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model, max_length=1024)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 3)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, ids_tensors, lable_tensors, masks_tensors, segments_tensors):
        _, pooled_outputs = self.bert(input_ids=ids_tensors, attention_mask=masks_tensors, token_type_ids=segments_tensors).to_tuple()
        linear_outputs = self.linear(pooled_outputs)
        if lable_tensors is not None:
            loss = self.loss_fn(linear_outputs, lable_tensors)
            return loss
        else:
            return linear_outputs
