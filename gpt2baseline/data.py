#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer


STYLE_TOKENS = ['<emo>', '<eve>', '<norm>']
MASK_TOKENS = ['[MALE]', '[FEMALE]', '[NEUTRAL]']
genre = ['emotional', 'event', 'normal']  # column names of corresponding style

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
tokenizer.add_tokens(STYLE_TOKENS, special_tokens=True)
tokenizer.add_tokens(MASK_TOKENS, special_tokens=True)
tokenizer.add_special_tokens({'pad_token':'<PAD>'}) # TODO: use <bos> as pad_token instead of add a new one


def flush_print(s):
    print(s)
    sys.stdout.flush()

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

    global tokenizer

    df = pd.read_csv(file_path)
    contexts = []  # first sentence
    stories = []   # left 4 sentences
    keywords = []
    labels = []
    story_keys = [f'sentence{i}' for i in range(1, 6)]  # whole story instead of range(2, 6)
    for idx in range(len(df)):
        label = df['label'][idx]
        story = [get_style_token(label)]
        for key in story_keys:
            story.append(df[key][idx])  # add bos token at the end of story
        story = ' '.join(story) + tokenizer.eos_token
        stories.append(story)
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
        item['context'] = self.ctx_encs[idx]
        item['story'] = self.story_encs[idx]
        # item['labels'] = torch.tensor(self.labels[idx])

        # keywords_ids = self.keywords_encs['input_ids'][idx]
        # dist = torch.zeros(len(tokenizer))
        # dist[keywords_ids[1:-1]] = 1
        # item['keywords_dist'] = dist

        return item

    def __len__(self):
        return len(self.labels)


def get_dataloader_and_tokenizer(file_path, batch_size, shuffle=True):
    # cached_path = file_path + '.gpt2.cached.pkl'
    
    # try:
    #     flush_print('Cache found, directly loading data...')
    #     with open(cached_path, 'rb') as fin:
    #         contexts_length, stories, keywords, labels = pickle.load(fin)
    # except (FileNotFoundError, EOFError):
    #     flush_print('Cache not found, processing from ground up...')
    contexts, stories, keywords, labels = read_roc(file_path)
        # contexts = tokenizer(contexts)
        # contexts_length = [len(item) for item in contexts['input_ids']]  
        # stories = tokenizer(stories, truncation=True, padding='longest')
        # keywords = tokenizer(keywords)
        # with open(cached_path, 'wb') as fout:
        #     pickle.dump((contexts_length, stories, keywords, labels), fout)
    dataset = ROCDataset(contexts, stories, keywords, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), tokenizer


if __name__ == '__main__':
    file_dir = '../data/ROCStories_dev.csv'
    dev_loader, tokenizer = get_dataloader_and_tokenizer(file_dir, batch_size=2, shuffle=True)
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
