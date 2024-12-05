from transformers import BertTokenizer, BertModel
import torch.nn as nn

def pretrain_bert_wwm_token():
    tokenizer = BertTokenizer.from_pretrained("model/pretrain/roberta_wwm")
    return tokenizer

def pretrain_bert_wwm_model():
    model = BertModel.from_pretrained("model/pretrain/roberta_wwm").cuda()
    for param in model.parameters():
        param.requires_grad = False
    return model

def pretrain_bert_uncased_token():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer

def pretrain_bert_uncased_model():
    model = BertModel.from_pretrained("bert-base-uncased").cuda()
    for param in model.parameters():
        param.requires_grad = False
    return model

