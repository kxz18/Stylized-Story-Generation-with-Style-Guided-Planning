#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class BertClassify(nn.Module):

    def __init__(self, class_num):
        super(BertClassify, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden_size = 768
        ff_size = 4 * hidden_size
        self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, ff_size),
                    nn.ReLU(),
                    nn.Linear(ff_size, class_num))

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        hiddens = output.last_hidden_state  # [batch_size, seq_len, hidden_size]
        hidden = hiddens[:, 0]  # [batch_size, hidden_size]
        res = self.classifier(hidden)
        return res


TOKENIZER = None
MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def classify(texts):
    global TOKENIZER, MODEL
    if TOKENIZER is None:
        TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
    if MODEL is None:
        path = __file__.replace('bert.py', 'bert.pth')
        prefix = __file__.replace('bert.py', '')
        sys.path.append(prefix)
        MODEL = torch.load(path, map_location='cpu').to(DEVICE)
        sys.path.remove(prefix)
        MODEL.eval()
    
    input_dict = TOKENIZER(texts, truncation=True, padding=True)
    with torch.no_grad():
        output = MODEL(torch.tensor(input_dict['input_ids']).to(DEVICE),
                       torch.tensor(input_dict['attention_mask']).to(DEVICE))
        output = F.softmax(output, dim=-1)

    return output.detach().cpu().numpy()
