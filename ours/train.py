#!/usr/bin/python
# -*- coding:utf-8 -*-
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW
from model import OursGen
from data import get_dataloader_and_tokenizer, flush_print



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)

setup_seed(6)


def parse():
    '''parse commands'''
    parser = argparse.ArgumentParser(description="train or test classifier")
    parser.add_argument("--train_set", type=str, required="--test" not in sys.argv,
            help="train set data path")
    parser.add_argument("--dev_set", type=str, required="--test" not in sys.argv,
            help="dev set data path")
    parser.add_argument("--test_set", type=str, required="--test" in sys.argv,
            help="test set data path")
    parser.add_argument("-lr", type=float, required="--test" not in sys.argv,\
            help="learning rate")
    parser.add_argument("-e", type=int, required="--test" not in sys.argv,
            help="times of epoch to run")
    parser.add_argument("--model", type=str, required=True,\
            help="weights storage of the model")
    parser.add_argument("--test", action="store_true", help="if in test mode")
    parser.add_argument("--batch-size", type=int, default=100,\
            help="batch size (how many sentence one round)")
    parser.add_argument("--step", type=int, default=2000,\
            help="number of data of a step")
    parser.add_argument("--device", type=int, default=0,\
            help="device number of cuda")
    parser.add_argument("-c", action="store_true",
            help="continue training")
    parser.add_argument("--alpha", action="store", type=float,default=0.2,
            help="keywords loss factor")
    return parser.parse_args()

args = parse()

USE_CUDA = torch.cuda.is_available()
if USE_CUDA and args.device >= 0:
    torch.cuda.set_device(args.device)
if args.device >= 0:
    print('='*10+f'Using gpu {args.device}'+'='*10)
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
else:
    print('='*10+'Using cpu'+'='*10)
    DEVICE = torch.device('cpu')

ALPHA = args.alpha     # factor of keyword loss

def evaluate(model, dataloader):
    '''evaluate on dev set'''
    model.eval()
    story_loss_fn = nn.NLLLoss(ignore_index=model.tokenizer.pad_token_id)
    keyword_loss_fn = nn.BCELoss(reduction='sum')
    losses = []
    story_losses = []
    keyword_losses = []

    with torch.no_grad():
        for batch in dataloader:

            enc_in_ids = batch['enc_in_ids'].to(DEVICE)
            enc_att_mask = batch['enc_att_mask'].to(DEVICE)
            dec_in_ids = batch['dec_in_ids'].to(DEVICE)
            dec_att_mask = batch['dec_att_mask'].to(DEVICE)
            keyword_dist = batch['keywords_dist'].to(DEVICE)
            
            outputs = model(input_ids=enc_in_ids, attention_mask=enc_att_mask,
                            decoder_input_ids=dec_in_ids[:, :-1], use_cache=False, 
                            decoder_attention_mask=dec_att_mask[:, :-1])  # [batch_size, seq_len]
            story_loss = story_loss_fn(outputs['story'].view(-1, len(model.tokenizer)),
                           dec_in_ids[:, 1:].reshape(1, -1).squeeze(0))

            # print(f'origin golden shape = {keyword_dist.size()}, min of keyword = {torch.min(outputs["keyword"])}, max of keyword = {torch.max(outputs["keyword"])}')
            # print(f'min of golden = {torch.min(golden_keyword)}, max of gloden = {torch.max(golden_keyword)}\n'
            #         f'min of max origin gloden = {torch.min(torch.max(keyword_dist, dim=-1)[0])}')
            keyword_loss = keyword_loss_fn(outputs['keyword'], keyword_dist) / (torch.sum(keyword_dist) + 1)

            loss = story_loss + ALPHA * keyword_loss
            story_losses.append(story_loss.detach().item())
            keyword_losses.append(keyword_loss.detach().item())
            losses.append(loss.detach().item())
            
    model.train()
    val_loss = np.mean(losses)
    val_story_loss = np.mean(story_losses)
    val_keyword_loss = np.mean(keyword_losses)
    return val_loss, val_story_loss, val_keyword_loss


def train(model, dataloader, dev_dataloader):
    global args
    flush_print('='*10 + 'start training' + '='*10)
    model = model.to(DEVICE)
    story_loss_fn = nn.NLLLoss(ignore_index=model.tokenizer.pad_token_id)
    keyword_loss_fn = nn.BCELoss(reduction='sum')

    optimizer = AdamW(model.parameters(), lr=args.lr)

    step_data_num = 0  # record data num within the step
    step = 0
    step_start = time.time()
    best_val_loss = 100000
    best_pos = (0, 0, 0)  # (epoch, batch, step)
    early_stop = False

    for epoch in range(args.e):
        flush_print(f'epoch {epoch} start')
        start = time.time()
        model.train()

        epoch_losses = []
        step_losses = []
        story_losses = []
        keyword_losses = []

        for idx, batch in enumerate(dataloader):

            enc_in_ids = batch['enc_in_ids'].to(DEVICE)
            enc_att_mask = batch['enc_att_mask'].to(DEVICE)
            dec_in_ids = batch['dec_in_ids'].to(DEVICE)
            dec_att_mask = batch['dec_att_mask'].to(DEVICE)
            keyword_dist = batch['keywords_dist'].to(DEVICE)
            batch_size = len(enc_in_ids)
            
            outputs = model(input_ids=enc_in_ids, attention_mask=enc_att_mask,
                            decoder_input_ids=dec_in_ids[:, :-1], use_cache=False, 
                            decoder_attention_mask=dec_att_mask[:, :-1])  # [batch_size, seq_len]
            story_loss = story_loss_fn(outputs['story'].view(-1, len(model.tokenizer)),
                           dec_in_ids[:, 1:].reshape(1, -1).squeeze(0))

            # debug
            # print(f'story loss: {story_loss}')
            if torch.isnan(outputs['story']).sum() > 0:
                print('story nan: {outputs["story"]}')
                break

            # print(f'origin golden shape = {keyword_dist.size()}, min of keyword = {torch.min(outputs["keyword"])}, max of keyword = {torch.max(outputs["keyword"])}')
            # print(f'min of golden = {torch.min(golden_keyword)}, max of gloden = {torch.max(golden_keyword)}\n'
            #         f'min of max origin gloden = {torch.min(torch.max(keyword_dist, dim=-1)[0])}')
            
            # +1 in case keyword_dist sum to 0
            try:
                keyword_loss = keyword_loss_fn(outputs['keyword'], keyword_dist) / (torch.sum(keyword_dist) + 1)
                if torch.isnan(keyword_loss).any():
                    print('outputs : \n', outputs['keyword'])
                    print('keyword dist : \n', keyword_dist)
                    print(f'keyword loss: {keyword_loss}')
            except Exception:
                print('keyword nan')
                print(outputs['keyword'])
                continue

            loss = story_loss + ALPHA * keyword_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            story_losses.append(story_loss.detach().item())
            keyword_losses.append(keyword_loss.detach().item())
            epoch_losses.append(loss.detach().item())
            step_losses.append(loss.detach().item())

            # step check
            step_data_num += batch_size
            if step_data_num >= args.step:
                step += 1
                eval_time_start = time.time()
                val_loss, val_story_loss, val_keyword_loss = evaluate(model, dev_dataloader)
                flush_print(f'\n\tstep {step}, batch {idx}, elapsed {time.time() - step_start} s:\n'
                            f'\t\tstory train set loss: {np.mean(story_losses)}\n'
                            f'\t\tkeyword train set loss: {np.mean(keyword_losses)}\n'
                            f'\t\ttrain set loss: {np.mean(step_losses)}\n'
							f'\t\tdev story loss: {val_story_loss}\n'
                            f'\t\tdev keyword loss: {val_keyword_loss}\n'
                            f'\t\tdev set loss: {val_loss}\n'
                            f'\t\teval time elapsed: {time.time() - eval_time_start} s')
                if val_loss < best_val_loss:
                    # save model
                    with open(args.model, 'wb') as fout:
                        torch.save(model, fout)
                    best_val_loss = val_loss
                    best_pos = (epoch, idx, step)
                    flush_print('\n\t\tbest model saved')
                flush_print(f'\t\tBest val loss: {best_val_loss}\n'
                            f'\t\tAt Epoch {best_pos[0]}, batch {best_pos[1]}, step {best_pos[2]}')
                if step - best_pos[2] >= 10:
                    # last update step is 10 earlier than current step, thus likely to be overfitting
                    early_stop = True
                    break
                # reset
                step_data_num = step_data_num % args.step
                step_losses = []
                keyword_losses = []
                step_start = time.time()
                story_losses = []
        # finish epoch
        flush_print(f'epoch {epoch} finished, elapsed {time.time() - start}')
        if early_stop:
            break
    flush_print('='*10 + 'Finished Training' + '='*10)
    flush_print(f'Best val loss: {best_val_loss}\n'
                f'At Epoch {best_pos[0]}, batch {best_pos[1]}, step {best_pos[2]}\n')



if __name__ == '__main__':
    print('='*10 + 'Configuration' + '='*10)
    print(args)
    if args.test:
        flush_print('='*10 + 'Loading test set' + '='*10)
        test_loader, _ = get_dataloader_and_tokenizer(
                                    file_path=args.test_set,
                                    batch_size=args.batch_size)
        flush_print('='*10 + 'Loading model' + '='*10)
        model = torch.load(args.model)
        flush_print(str(model))
        flush_print('='*10 + 'Begin test' + '='*10)
        _, test_loss, __ = evaluate(model, test_loader)
        flush_print(f'test result: loss {test_loss}, ppl: {pow(np.e, test_loss)}')
    else:
        if args.lr > 5e-5:
            flush_print('WARNING: optimal learning rate for fine-tuning is 5e-5')
        flush_print('='*10 + 'Loading Training set' + '='*10)
        train_loader, tokenizer = get_dataloader_and_tokenizer(
                                    file_path=args.train_set,
                                    batch_size=args.batch_size)
        flush_print('='*10 + 'Loading dev set' + '='*10)
        dev_loader, _ = get_dataloader_and_tokenizer(
                                file_path=args.dev_set,
                                batch_size=args.batch_size,
                                shuffle=False)
        if args.c:
            flush_print('='*10 + 'Loading model' + '='*10)
            model = torch.load(args.model)
            flush_print(f"Model loaded from {args.model}")
        else:
            flush_print('Training from ground up')
            flush_print(f'Model path: {args.model}')
            flush_print('='*10 + 'Creating model' +'='*10)
            model_kwargs = {'tokenizer': tokenizer}
            model = OursGen.from_pretrained('facebook/bart-base', **model_kwargs)
            model.resize_token_embeddings(len(tokenizer))
        train(model, train_loader, dev_loader)
