import torch
import os
import sys
import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('../../')
from bartbaseline.data import get_style_token as bart_style_token
from gpt2baseline.data import get_style_token as gpt_style_token
from ours.data import get_style_token as ours_style_token
sys.path.remove('../../')

sys.path.append('../../bartbaseline')
import bartGen
sys.path.remove('../../bartbaseline')

sys.path.append('../../gpt2baseline')
import GPT
sys.path.remove('../../gpt2baseline')

sys.path.append('../../ours')
import model
sys.path.remove('../../ours')

def parse():
    parser = argparse.ArgumentParser(description="Generate story on test set")
    parser.add_argument('--model', type=str, required=True, help='model saved path')
    parser.add_argument('--data', type=str, required=True, help='Test set path, supposed to be csv')
    parser.add_argument('--batch_size', type=int, help='batch size when model generate story')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='device to run model')
    parser.add_argument('--output', type=str, help='output file path')
    parser.add_argument('-t', '--temperature', type=float, default=0.8, help='temperature for sampling')
    parser.add_argument('--model-type', type=str, choices=['ours', 'bart', 'gpt2'], required=True,
                        help='type of loaded model')
    return parser.parse_args()

def BartBaseline_gen(model, contexts, label, device, temperature):
    '''generation function of models based on bart
       contexts: list of raw texts
       label: labels of style'''
    contexts = list(contexts)
    labels = list(label)
    for idx in range(len(contexts)):
        contexts[idx] = bart_style_token(labels[idx]) + ' ' + contexts[idx]
    return model.inference(contexts, device, temperature=temperature)   # 1 and -1 are </s>


def GPT2_gen(model, contexts, label, device, temperature):
    '''generation function of models based on gpt
       contexts: list of raw texts
       label: labels of style'''
    contexts = list(contexts)
    labels = list(label)
    for idx in range(len(contexts)):
        contexts[idx] = gpt_style_token(labels[idx]) + ' ' + contexts[idx]
    whole_stories = model.inference(contexts, device, temperature=temperature)   # whole story, including leading context
    for i, story in enumerate(whole_stories):
        whole_stories[i] = story[story.find('.')+1:]
    return whole_stories  # get rid of the first sentence

def Ours_gen(model, contexts, label, device, temperature):
    '''generation function of models based on bart
       contexts: list of raw texts
       label: labels of style'''
    contexts = list(contexts)
    labels = list(label)
    for idx in range(len(contexts)):
        contexts[idx] = ours_style_token(labels[idx]) + ' ' + contexts[idx]
    return model.inference(contexts, device, temperature=temperature)   # 1 and -1 are </s>

def get_gen(model_type):
    '''return the corresponding function
    '''
    if model_type == 'bart':
        gen = BartBaseline_gen
    elif model_type == 'gpt2':
        gen = GPT2_gen
    elif model_type == 'ours':
        gen = Ours_gen
    return gen


def run(args):
    if (args.device == 'cpu') or (not torch.cuda.is_available()):
        DEVICE = torch.device('cpu')
    elif (args.device == 'cuda'):
        print('device is cuda')
        torch.cuda.set_device(0)
        DEVICE = torch.device('cuda')

    args.output += f'_{args.model_type}'

    f0 = open(args.output+'_0.txt', 'w')
    f1 = open(args.output+'_1.txt', 'w')
    f2 = open(args.output+'_2.txt', 'w')

    # load data
    df = pd.read_csv(args.data)

    # load model
    model = torch.load(args.model, map_location='cpu')
    model = model.to(DEVICE)

    start = 0
    batch_num = len(df) / args.batch_size
    counter = 0
    while start < len(df):
        counter += 1
        print(f'batch {counter}/{batch_num}', end=', ')
        start_time = time.time()
        end = min(len(df), start + args.batch_size)
        contexts = df[start:end]['sentence1']
        batch_size = len(contexts)

        gen = get_gen(args.model_type)

        story_0 = gen(model, contexts, [0]*batch_size, DEVICE, args.temperature)
        story_1 = gen(model, contexts, [1]*batch_size, DEVICE, args.temperature)
        story_2 = gen(model, contexts, [2]*batch_size, DEVICE, args.temperature)
        
        # write style 0 story
        for story in story_0:
            print(story, file=f0)
        # write style 1 story
        for story in story_1:
            print(story, file=f1)
        # write style 2 story
        for story in story_2:
            print(story, file=f2)
        start = end
        print('elapsed time = {:.2f} s'.format(time.time()-start_time), end=' '*10 + '\r')
    
    f0.close()
    f1.close()
    f2.close()



if __name__ == '__main__':
    args = parse()
    run(args)
