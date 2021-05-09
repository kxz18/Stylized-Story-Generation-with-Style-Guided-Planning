#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import numpy as np
sys.path.append('../..')
from data.emotional.emotional import judge_emotional
from data.event.event import judge_event
sys.path.remove('../..')


def judge(text):
    emo = judge_emotional(text)
    eve = judge_event(text)
    if (emo < 0.7 and eve < 0.7) or abs(emo - eve) < 0.3:
        # normal type, no specific style
        return 2, emo, eve, 0
    return np.argmax([emo, eve]), emo, eve, 0


if __name__ == '__main__':
    import pandas as pd

    path = sys.argv[1]
    data = pd.read_csv(path)
    example_num = 10
    sent_keys = [f'sentence{i+1}' for i in range(5)]
    for i in range(example_num):
        story = []
        for key in sent_keys:
            story.append(data[key][i])
        res = judge(' '.join(story[1:]))
        print('='*50)
        print(' '.join(story))
        items = ['label', 'emotional score', 'event score']
        prompt = ''
        for key, val in zip(items, res):
            prompt += f'{key}: {val}, '
        print(prompt[:-2])

