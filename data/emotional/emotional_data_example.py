#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import pandas as pd
from emotional import get_emotional_words, judge_emotional


file_path = sys.argv[1]
df = pd.read_csv(file_path)
sent_keys = [f'sentence{i+1}' for i in range(5)]
print(df.columns)
example_num = 20
total_len = 0
max_len = 0
min_len = 1000
num = 0
emotional_words = []
for idx in range(example_num):
    story = []
    for key in sent_keys:
        story.append(df[key][idx].replace('.', ' .').replace(',', ' ,'))
    # story_title = df['storytitle'][idx]
    story_title = ''
    cut_story = ' '.join(story[1:])
    story = ' '.join(story)
    print('='*10 + story_title + '='*10)
    print(story)
    print('emotional words:')
    print(get_emotional_words(cut_story, pos=True))
    print(judge_emotional(cut_story))
    print()
    word_list = get_emotional_words(story)
    emotional_words.append(word_list)
    total_len += len(word_list)
    if len(word_list) > max_len:
        max_len = len(word_list)
    if len(word_list) < min_len:
        min_len = len(word_list)
    if len(word_list) > 3:
        num += 1
# df.to_csv('ROCStories_add_emotional.csv')
print(f'number of entries: {example_num}, emotional: {num}')
print(f'average: {total_len / example_num}, min: {min_len}, max: {max_len}')
