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
from bert import BertClassify
from data import get_dataloader, flush_print



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
    return parser.parse_args()

args = parse()

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.set_device(args.device)
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

def evaluate(model, dataloader):
    '''evaluate on dev set'''
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)  # [batch_size]
            outputs = model(input_ids, attention_mask)  # [batch_size, class_num]
            loss = loss_fn(outputs, labels)

            losses.append(loss.detach().item())
            
            prediction = torch.argmax(outputs, -1)
            correct += (prediction == labels).sum().detach().item()
            total += len(prediction)

    model.train()
    val_loss = np.mean(losses)
    val_accu = correct / total
    # return val_loss, val_loss_story, val_loss_class, val_accu
    return val_loss, val_accu


def train(model, dataloader, dev_dataloader):
    global args
    flush_print('='*10 + 'start training' + '='*10)
    model = model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()

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

        for idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)  # [batch_size]
            outputs = model(input_ids, attention_mask)  # [batch_size, class_num]
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.detach().item())
            step_losses.append(loss.detach().item())

            # step check
            step_data_num += len(labels)
            if step_data_num >= args.step:
                step += 1
                eval_time_start = time.time()
                val_loss, val_acc = evaluate(model, dev_dataloader)
                flush_print(f'\n\tstep {step}, batch {idx}, elapsed {time.time() - step_start} s:\n'
                            f'\t\ttrain set loss: {np.mean(step_losses)}\n'
                            f'\t\tdev set loss: {val_loss}\n'
                            f'\t\tclassification acc: {val_acc}\n'
                            f'\t\teval time elapsed: {time.time() - eval_time_start} s')
                if val_loss < best_val_loss:
                    # save model
                    with open(args.model, 'wb') as fout:
                        torch.save(model, fout)
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    best_pos = (epoch, idx, step)
                    flush_print('\n\t\tbest model saved')
                flush_print(f'\t\tBest val loss: {best_val_loss}\n'
                            f'\t\tBest classification acc: {best_val_acc}\n'
                            f'\t\tAt Epoch {best_pos[0]}, batch {best_pos[1]}, step {best_pos[2]}')
                if step - best_pos[2] >= 10:
                    # last update step is 10 earlier than current step, thus likely to be overfitting
                    early_stop = True
                    break
                # reset
                step_data_num = step_data_num % args.step
                step_losses = []
                step_start = time.time()
        # finish epoch
        flush_print(f'epoch {epoch} finished, elapsed {time.time() - start}')
        if early_stop:
            break
    flush_print('='*10 + 'Finished Training' + '='*10)
    flush_print(f'Best val loss: {best_val_loss}\n'
                f'Best classification acc: {best_val_acc}\n'
                f'At Epoch {best_pos[0]}, batch {best_pos[1]}, step {best_pos[2]}\n')



if __name__ == '__main__':
    print('='*10 + 'Configuration' + '='*10)
    print(args)
    if args.test:
        flush_print('='*10 + 'Loading test set' + '='*10)
        test_loader = get_dataloader(file_path=args.test_set,
                                 batch_size=args.batch_size)
        flush_print('='*10 + 'Loading model' + '='*10)
        model = torch.load(args.model)
        flush_print(str(model))
        flush_print('='*10 + 'Begin test' + '='*10)
        test_loss, test_acc = evaluate(model, test_loader)
        flush_print(f'test result: loss {test_loss}, acc: {test_acc}')
    else:
        if args.lr > 5e-5:
            flush_print('WARNING: optimal learning rate for fine-tuning is 5e-5')
        flush_print('='*10 + 'Loading Training set' + '='*10)
        train_loader = get_dataloader(file_path=args.train_set,
                                  batch_size=args.batch_size)
        flush_print('='*10 + 'Loading dev set' + '='*10)
        dev_loader = get_dataloader(file_path=args.dev_set,
                                batch_size=args.batch_size)
        if args.c:
            flush_print('='*10 + 'Loading model' + '='*10)
            model = torch.load(args.model)
            flush_print(f"Model loaded from {args.model}")
        else:
            flush_print('Training from ground up')
            flush_print(f'Model path: {args.model}')
            flush_print('='*10 + 'Creating model' +'='*10)
            model = BertClassify(3)
        train(model, train_loader, dev_loader)
