import torch
import sys
import argparse
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk import sent_tokenize, word_tokenize


sys.path.append('../..')
from bartbaseline.data import get_style_token as bart_style_token
from gpt2baseline.data import get_style_token as gpt_style_token
from ours.data import get_style_token as ours_style_token
sys.path.remove('../..')


sys.path.append('../../bartbaseline')
import bartGen
sys.path.remove('../../bartbaseline')

sys.path.append('../../gpt2baseline')
import GPT
sys.path.remove('../../gpt2baseline')

sys.path.append('../../ours')
import model
sys.path.remove('../../ours')


def get_device(device):
    if not device or device == 'cpu':
        DEVICE = torch.device('cpu')
    elif device == 'cuda':
        torch.cuda.set_device(0)
        DEVICE = torch.device('cuda')
    return DEVICE

def parse():
    parser = argparse.ArgumentParser(description="Generate story on test set")
    parser.add_argument('--model', type=str, required=True, help='model saved path')
    parser.add_argument('--data', type=str, required=True, help='Test set path')
    parser.add_argument('--batch_size', type=int, help='batch size when model generate story')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='device to run model')
    parser.add_argument('--model-type', type=str, choices=['bart', 'gpt2', 'ours'], required=True,
                        help='type of loaded model')
    parser.add_argument('--save_bleu', action='store_true', help='Save bleu to file or not')
    parser.add_argument('-t', '--temperature', type=float, default=0.8, help='Temperature in generation')
    parser.add_argument("--mode", type=str, choices=['avg', 'max'], required=True,
            help="Use average of three type of the max bleu")
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
    '''return the corresponding generate function
    '''
    if model_type == 'bart':
        gen = BartBaseline_gen
    elif model_type == 'gpt2':
        gen = GPT2_gen
    elif model_type == 'ours':
        gen = Ours_gen
    return gen


def gen_step(args):
    '''generate stories before calculating BLEU
    return stories of style emo | eve | norm , respectively
    '''
    DEVICE = get_device(args.device)
    T = args.temperature
    model_type = args.model_type 
    # 读入测试集
    df = pd.read_csv(args.data)
    model = torch.load(args.model, map_location='cpu')
    model = model.to(DEVICE)

    start = 0
    batch_num = len(df) / args.batch_size
    counter = 0
    gen = get_gen(model_type)
    emo_stories = [] 
    eve_stories = [] 
    while start < len(df):
        counter += 1
        print(f'batch {counter}/{batch_num}', end=', ')
        start_time = time.time()
        end = min(len(df), start + args.batch_size)
        contexts = df[start:end]['sentence1']

        emo_st = gen(model, contexts, [0]*(end-start), DEVICE, temperature=T)
        eve_st = gen(model, contexts, [1]*(end-start), DEVICE, temperature=T)
        emo_stories.extend(emo_st)
        eve_stories.extend(eve_st)
        
        start = end
        print('elapsed time = {:.2f} s'.format(time.time()-start_time), end=' '*10 + '\r')
    return emo_stories, eve_stories

def sent_level_bleu(df:pd.DataFrame, raw_candidates):
    '''
       @param df: the golden truth read from the test set
       @param raw_candidates: the generated stories from gen_step
    '''
    bleu_1, bleu_2, bleu_3, bleu_4 = [], [], [], []
    for i, can in tqdm(enumerate(raw_candidates)):
        ref = ' '.join([df.iloc[i][f'sentence{x}'] for x in range(2, 6)])
        ref = [word_tokenize(ref)]
        can = word_tokenize(can)
        # calculating bleu
        bleu_1.append(sentence_bleu(ref, can, weights=[1]))
        bleu_2.append(sentence_bleu(ref, can, weights=[0.5, 0.5]))
        bleu_3.append(sentence_bleu(ref, can, weights=[1/3, 1/3, 1/3]))
        bleu_4.append(sentence_bleu(ref, can, weights=[0.25, 0.25, 0.25, 0.25]))
    return bleu_1, bleu_2, bleu_3, bleu_4


def bleu(emo_st, eve_st, args):
    # read test set
    df = pd.read_csv(args.data)
    assert(len(emo_st)==len(eve_st)==len(df))
    # averaging the sentence level BLEU
    print('*'*15 + 'MICRO BLEU' + '*'*15)
    emo_bleu_1, emo_bleu_2, emo_bleu_3, emo_bleu_4 = sent_level_bleu(df, emo_st)
    eve_bleu_1, eve_bleu_2, eve_bleu_3, eve_bleu_4 = sent_level_bleu(df, eve_st)
    if args.mode == 'max':
        bleu_1 = np.maximum(emo_bleu_1, eve_bleu_1).mean()
        bleu_2 = np.maximum(emo_bleu_2, eve_bleu_2).mean()
        bleu_3 = np.maximum(emo_bleu_3, eve_bleu_3).mean()
        bleu_4 = np.maximum(emo_bleu_4, eve_bleu_4).mean()
        result_path = f'{args.model_type}_{args.temperature}_max_bleu_micro.txt'
    elif args.mode == 'avg':
        bleu_1 = np.mean([emo_bleu_1, eve_bleu_1])
        bleu_2 = np.mean([emo_bleu_2, eve_bleu_2])
        bleu_3 = np.mean([emo_bleu_3, eve_bleu_3])
        bleu_4 = np.mean([emo_bleu_4, eve_bleu_4])
        result_path = f'{args.model_type}_{args.temperature}_avg_bleu_micro.txt'
    if args.save_bleu:
        with open(result_path, 'w') as f:
            for i, bleu in enumerate([bleu_1,bleu_2,bleu_3,bleu_4 ]):
                print(f'bleu{i+1} = {bleu}', file=f)
    else:
        for i, bleu in enumerate([bleu_1,bleu_2,bleu_3,bleu_4 ]):
            print(f'bleu{i+1} = {bleu}')


if __name__ == '__main__':
    args = parse()
    print(args)
    emo_st, eve_st = gen_step(args)
    bleu(emo_st, eve_st, args)
