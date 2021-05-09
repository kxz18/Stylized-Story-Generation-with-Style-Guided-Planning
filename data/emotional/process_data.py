#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from emotional import get_emotional_words


file_path = sys.argv[1]
df = pd.read_csv(file_path)
sent_keys = [f'sentence{i+1}' for i in range(1, 5)]
print(df.columns)
example_num = len(df)
total_len = 0
max_len = 0
min_len = 1000
num = 0
total = 0
kl_lens = []

for idx in tqdm(range(example_num)):
    story = []
    total += 1
    for key in sent_keys:
        story.append(df[key][idx].replace('.', ' .').replace(',', ' ,'))
    story = ' '.join(story)
    word_list, pos_list = get_emotional_words(story, pos=True)
    df.loc[idx, 'emotional'] = ', '.join(list(word_list))
    df.loc[idx, 'emotional_pos'] = ', '.join(list(pos_list))
    total_len += len(word_list)
    kl_lens.append(len(word_list))
    if len(word_list) > max_len:
        max_len = len(word_list)
    if len(word_list) < min_len:
        min_len = len(word_list)
    if len(word_list) > 3:
        num += 1
df.to_csv('ROCStories_add_emotional.csv', index=False)
print(f'number of entries: {example_num}, emotional: {num}')
print(f'average: {total_len / example_num}, min: {min_len}, max: {max_len}')
mean = np.mean(kl_lens)
sigma = np.sqrt(np.var(kl_lens))
print(f'mean: {mean}, sigma: {sigma}')
with open('stats.data', 'wb') as fout:
    pickle.dump((mean, sigma), fout)
