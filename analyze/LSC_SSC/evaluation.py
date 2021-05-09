#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import re
import argparse
import numpy as np
from tqdm import tqdm
from distinctn import distinctn
from judge_genre import judge
from classification.bert import classify


FILE = None
DISCARDED = ['[MALE]', '[FEMALE]', '[NEUTRAL]',
             '<eos>', '<unk>', '<pad>',
             '<emo>', '<eve>']

def print_save(s):
    print(s)
    FILE.write(s + '\n')


def parse():
    parser = argparse.ArgumentParser(description='calculate distinc1/2/3 and '
                                                 'style score of given generated texts')
    parser.add_argument('--src', type=str, required=True,
                        help='source of texts')
    parser.add_argument('--output', type=str, required=True,
                        help='path of file to save results')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='size of a batch for bert classification')
    return parser.parse_args()


def average_distinctn(texts, ns):
    '''texts: list of texts
       ns: list of values of n
       return: average distinctn'''
    for n in ns:
        print_save(f'calculating average distinct-{n}')
        scores = []
        for text in tqdm(texts):
            scores.append(distinctn(text, n))
        score = np.mean(scores)
        print_save(f'average distinct-{n}: {score}')
    

def average_style_eval(texts):
    '''texts: list of texts
       return: accu, average score'''
    print_save(f'calculating style scores')
    style_num = None
    style_scores = None
    for text in tqdm(texts):
        res = judge(text)
        label = res[0]
        scores = res[1:]

        if style_num is None:
            # initialize
            style_num = [0 for i in range(len(scores))]
            style_scores = [[] for i in range(len(scores))]

        style_num[label] += 1
        for idx, score in enumerate(scores):
            style_scores[idx].append(score)

    avg_score = np.mean(style_scores, axis=1)
    style_accu = np.array(style_num) / len(texts)
    for style, accu, score in zip([i for i in range(len(style_num))], style_accu, avg_score):
        print_save(f'style {style} accu: {accu}, score: {score}')


def average_bert_eval(texts, batch_size):
    print_save(f'calculating style scores by bert')
    style_num = None
    style_scores = None
    batch_num = int((len(texts) - 1) / batch_size) + 1
    for batchIdx in tqdm(range(batch_num)):
        begin = batchIdx * batch_size
        end = min(begin + batch_size, len(texts))
        inputs = texts[begin:end]
        res = classify(inputs)  # [batch_size, class_num]
        
        for scores in res:
            label = np.argmax(scores)
            if style_num is None:
                # initialize
                style_num = [0 for i in range(len(scores))]
                style_scores = [[] for i in range(len(scores))]

            style_num[label] += 1
            for idx, score in enumerate(scores):
                style_scores[idx].append(score)

    avg_score = np.mean(style_scores, axis=1)
    style_accu = np.array(style_num) / len(texts)
    for style, accu, score in zip([i for i in range(len(style_num))], style_accu, avg_score):
        print_save(f'style {style} accu: {accu}, score: {score}')


if __name__ == '__main__':
    args = parse()
    FILE = open(args.output, 'w')
    print_save('=' * 10 + f'analyzing texts from {args.src}' + '=' * 10)
    with open(args.src, 'r') as fin:
        texts = fin.read()
    texts = re.sub(r'\n+', '\n', texts).rstrip('\n').split('\n')
    for idx, text in enumerate(texts):
        for word in DISCARDED:
            text = text.replace(word, '')
        texts[idx] = text.strip('\n').strip(' ')
    average_distinctn(texts, [1, 2, 3])
    average_style_eval(texts)
    average_bert_eval(texts, args.batch_size)
    FILE.close()
