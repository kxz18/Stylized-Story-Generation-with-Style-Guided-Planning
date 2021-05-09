#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

from pathlib import Path
import pickle
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer


def flush_print(s):
    print(s)
    sys.stdout.flush()


def read_roc_split(file_dir):
    file_dir = Path(file_dir)
    texts = []
    labels = []
    df = pd.read_csv(file_dir)
    keys = [f'sentence{i+1}' for i in range(5)]
    for idx in range(len(df)):
        story = []
        for key in keys:
            story.append(df[key][idx])
        texts.append(' '.join(story))
        labels.append(df['label'][idx])
    return texts, labels


class ROCDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def get_dataloader(file_path, batch_size, shuffle=True):
    cached_path = file_path + '.cached.pkl'
    try:
        flush_print('Cache found, directly loading data...')
        with open(cached_path, 'rb') as fin:
            encodings, labels = pickle.load(fin)
    except FileNotFoundError:
        flush_print('Cache not found, processing from ground up...')
        texts, labels = read_roc_split(file_path)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encodings = tokenizer(texts, truncation=True, padding=True)
        with open(cached_path, 'wb') as fout:
            pickle.dump((encodings, labels), fout)
    dataset = ROCDataset(encodings, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    file_dir = '../../data/ROCStories_dev.csv'
    dev_loader = get_dataloader(file_dir, batch_size=16, shuffle=True)
    for batch in dev_loader:
        print(batch)
        break