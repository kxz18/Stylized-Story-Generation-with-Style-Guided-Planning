#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import pickle
import nltk
import pandas as pd
from tqdm import tqdm
from protagonist import extract_protagonist


file_path = sys.argv[1]
df = pd.read_csv(file_path)
sent_keys = [f'sentence{i+1}' for i in range(5)]
print(df.columns)
example_num = len(df)
num = 0

for idx in tqdm(range(example_num)):
    story = []
    for key in sent_keys:
        story.append(df[key][idx])
    story = ' '.join(story)
    protagonist = extract_protagonist(story)
    df.loc[idx, 'protagonist'] = protagonist
    if protagonist is None:
        print(story)
        story = nltk.word_tokenize(story)
        print(nltk.pos_tag(story))
        num += 1
df.to_csv('ROCStories_add_protagonist.csv')
print(f'number of entries: {example_num}, protagonist not found: {num}')
