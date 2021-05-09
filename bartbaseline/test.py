import argparse
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AdamW
from torch.utils.data import DataLoader

from bartGen import BartGen
from data import get_dataloader_and_tokenizer as get_tokenizer, flush_print, get_style_token

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)

setup_seed(6)

def parse():
    parser = argparse.ArgumentParser(description="train or test classifier")
    parser.add_argument("--test_set", type=str, required=True, help="Test set data path")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--mode", type=str, choices=['avg', 'min'], required=True,
            help="Use average of three type of the min ppl")
    parser.add_argument("--batch_size", type=int, default=32,\
            help="Batch size (how many sentence one round)")
    return parser.parse_args()

def read_roc(file_path):
    df = pd.read_csv(file_path)
    contexts = []  # first sentence
    stories = []   # left 4 sentences
    story_keys = [f'sentence{i}' for i in range(2, 6)]
    for idx in range(len(df)):
        story = []
        for key in story_keys:
            story.append(df[key][idx])
        story = ' '.join(story)
        stories.append(story)
        contexts.append(df['sentence1'][idx])
    return contexts, stories

class ROCDataset(torch.utils.data.Dataset):
    def __init__(self, ctx, story):
        self.ctx = ctx  # encoding(ids, mask) of context
        self.story = story

    def __getitem__(self, idx):
        item = {}
        item['ctx'] = self.ctx[idx]
        item['story'] = self.story[idx]
        return item

    def __len__(self):
        return len(self.ctx)

def get_dataloader_and_tokenizer(args, shuffle=True, num_workers=4):
    _, tokenizer = get_tokenizer(file_path=args.test_set,
                                batch_size=args.batch_size)
    contexts, stories = read_roc(args.test_set)
    dataset = ROCDataset(contexts, stories)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                      shuffle=shuffle, num_workers=num_workers)
    return dataloader, tokenizer

def evaluate(model, dataloader, tokenizer, mode, DEVICE):
    '''evaluate on test set'''
    print('*'*10 + f'eval mode = {mode}' + '*'*10)
    model.eval()
    story_loss_fn = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id, reduction='none')
    story_losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            emo_ctx = [f'{get_style_token(0)} {item}' for item in batch['ctx']]
            eve_ctx = [f'{get_style_token(1)} {item}' for item in batch['ctx']]
            story = batch['story']
            batch_size = len(story)
            emo_ctx = tokenizer(emo_ctx, truncation=True, padding=True)
            eve_ctx = tokenizer(eve_ctx, truncation=True, padding=True)
            story = tokenizer(story, truncation=True, padding=True)
            length = (torch.tensor(story['input_ids']) != tokenizer.pad_token_id).sum(dim=-1).to(DEVICE)
            dec_in_ids = torch.tensor(story['input_ids']).to(DEVICE)
            dec_att_mask = torch.tensor(story['attention_mask']).to(DEVICE)

            loss = []
            for ctx in [emo_ctx, eve_ctx]:
                enc_in_ids = torch.tensor(ctx['input_ids']).to(DEVICE)
                enc_att_mask = torch.tensor(ctx['attention_mask']).to(DEVICE)
            
                outputs = model(enc_in_ids, enc_att_mask,
                                dec_in_ids[:, :-1], dec_att_mask[:, :-1])  # [batch_size, seq_len]
                outputs = outputs.permute(0, 2, 1)
                story_loss = story_loss_fn(outputs, dec_in_ids[:, 1:])
                story_loss = story_loss.sum(dim=-1) / length
                loss.append(story_loss)     # each element size in `loss` is [batch_size, ]
            story_loss = torch.stack(loss, dim=0) # [3, batch_size]
            assert(story_loss.size() == (2, batch_size))
            if mode == 'min':
                story_loss = torch.amin(story_loss, dim=0)
            elif mode == 'avg':
                story_loss = story_loss.mean(dim=0)
            assert(story_loss.size() == (batch_size, ))
            story_loss = story_loss.mean()
            story_losses.append(story_loss.detach().item())

    val_story_loss = np.mean(story_losses)
    return val_story_loss


if __name__ == '__main__':
    print('='*10 + 'Configuration' + '='*10)
    args = parse()
    print(args)
    flush_print('='*10 + 'Loading test set' + '='*10)
    test_loader, tokenizer = get_dataloader_and_tokenizer(args)
    flush_print('='*10 + 'Loading model' + '='*10)
    model = torch.load(args.model)
    flush_print(str(model))
    DEVICE = torch.device('cuda')
    flush_print('='*10 + 'Begin test' + '='*10)
    test_loss = evaluate(model, test_loader, tokenizer, mode=args.mode, DEVICE=DEVICE)
    flush_print(f'test result: loss {test_loss}, ppl: {pow(np.e, test_loss)}')
