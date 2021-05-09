#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer


STYLE_TOKENS = ['<emo>', '<eve>', '<norm>']
MASK_TOKENS = ['[MALE]', '[FEMALE]', '[NEUTRAL]']
genre = ['emotional', 'event', 'normal']  # column names of corresponding style

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
tokenizer.add_tokens(STYLE_TOKENS, special_tokens=True)
tokenizer.add_tokens(MASK_TOKENS, special_tokens=True)


def flush_print(s):
    print(s)
    sys.stdout.flush()

def label_to_idx(label, word_to_idx):
    '''transform label to idx in vocabulary'''
    if label == 0:
        return word_to_idx[EMO]
    elif label == 1:
        return word_to_idx[EVE]
    elif label == 2:
        return word_to_idx[NORM]
    else:
        flush_print("label out of range")
        return word_to_idx[UNK]


def get_style_token(label):
    return STYLE_TOKENS[label]


def get_style_token_id(style_token):
    return tokenizer.get_added_vocab()[style_token]


def get_keywords(data_frame, idx, label):
    '''return string of keywords splitted by space'''
    global genre
    keywords_str = data_frame[genre[label]][idx]
    if not isinstance(keywords_str, str):
        keywords_str = ''
    else:
        # cut-off to maximum of 5 words
        keywords_str = keywords_str.replace('"', '').replace(',', ' ')
        kl = keywords_str.split()
        kl = kl[0:min(5, len(kl))]
        keywords_str = ' '.join(kl)
    return keywords_str


def read_roc(file_path):
    df = pd.read_csv(file_path)
    contexts = []  # first sentence
    stories = []   # left 4 sentences
    keywords = []
    labels = []
    story_keys = [f'sentence{i}' for i in range(2, 6)]
    for idx in range(len(df)):
        story = []
        for key in story_keys:
            story.append(df[key][idx])
        story = ' '.join(story)
        stories.append(story)
        label = df['label'][idx]
        labels.append(label)
        contexts.append(f'{get_style_token(label)} ' + df['sentence1'][idx])
        keywords.append(get_keywords(df, idx, label))
    return contexts, stories, keywords, labels


class ROCDataset(torch.utils.data.Dataset):
    def __init__(self, ctx_encs, story_encs, keywords_encs, labels):
        self.ctx_encs = ctx_encs  # encoding(ids, mask) of context
        self.story_encs = story_encs
        self.keywords_encs = keywords_encs
        self.labels = labels

    def __getitem__(self, idx):
        item = {}
        item['enc_in_ids'] = self.ctx_encs.data['input_ids'][idx]
        item['enc_att_mask'] = self.ctx_encs['attention_mask'][idx]
        item['dec_in_ids'] = self.story_encs['input_ids'][idx]
        item['dec_att_mask'] = self.story_encs['attention_mask'][idx]
        item['labels'] = self.labels[idx]
        for key in item:
            item[key] = torch.tensor(item[key])

        keywords_ids = self.keywords_encs['input_ids'][idx]
        dist = torch.zeros(len(tokenizer))
        dist[keywords_ids[1:-1]] = 1  # start and end are <s> </s>
        # dist sum
        dist_sum = dist.sum()
        if len(keywords_ids[1:-1]) != 0:
            dist = dist / dist_sum
        item['keywords_dist'] = dist

        return item

    def __len__(self):
        return len(self.labels)


def get_dataloader_and_tokenizer(file_path, batch_size, shuffle=True, num_workers=4):
    cached_path = file_path + '.ours.cached.pkl'
    
    try:
        flush_print('Try to locate cache ...')
        with open(cached_path, 'rb') as fin:
            contexts, stories, keywords, labels = pickle.load(fin)
        flush_print('Cache found, directly loading data...')
    except (FileNotFoundError, EOFError):
        flush_print('Cache not found, processing from ground up...')
        contexts, stories, keywords, labels = read_roc(file_path)
        contexts = tokenizer(contexts, truncation=True, padding=True)
        stories = tokenizer(stories, truncation=True, padding=True)
        keywords = tokenizer(keywords)
        with open(cached_path, 'wb') as fout:
            pickle.dump((contexts, stories, keywords, labels), fout)
    dataset = ROCDataset(contexts, stories, keywords, labels)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers), tokenizer


if __name__ == '__main__':
    file_dir = './dataset/ROCStories_dev.csv'
    dev_loader, tokenizer = get_dataloader_and_tokenizer(file_dir, batch_size=2, shuffle=True)
    print(len(dev_loader))
    for batch in dev_loader:
        for key in batch:
            val = batch[key]
            if key == 'enc_in_ids':
                print(tokenizer.decode(val[0]))
                print(tokenizer.decode(val[1]))
            elif key == 'keywords_dist':
                print(tokenizer.decode(torch.nonzero(val[0]).squeeze(1)))
                print(tokenizer.decode(torch.nonzero(val[1]).squeeze(1)))
            print(key, val)
        break
    print('original special tokens:')
    for key in tokenizer.special_tokens_map:
        print(key, tokenizer.special_tokens_map[key])
    print('added tokens')
    print(tokenizer.get_added_vocab())
