#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import pandas as pd
from protagonist import extract_protagonist


file_path = sys.argv[1]
df = pd.read_csv(file_path)
sent_keys = [f'sentence{i+1}' for i in range(5)]
print(df.columns)
example_num = 20
total_len = 0
num = 0
emotional_words = []
for idx in range(example_num):
    story = []
    for key in sent_keys:
        story.append(df[key][idx])
    story_title = df['storytitle'][idx]
    story = ' '.join(story)
    print('='*10 + story_title + '='*10)
    print(story)
    pro = extract_protagonist(story)
    print(f'protagnist: {pro}')
    print()
    if pro is None:
        num += 1
print(f'number of entries: {example_num}, has a protagnist: {num}')
