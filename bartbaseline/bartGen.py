#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration


class BartGen(nn.Module):
    def __init__(self, tokenizer):
        super(BartGen, self).__init__()
        # self.bart = BartModel.from_pretrained('facebook/bart-large')
        self.bart = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        self.bart.resize_token_embeddings(len(tokenizer))
        bart_last_hidden_size = 1024
        self.classifier = nn.Linear(bart_last_hidden_size, len(tokenizer))
        self.tokenizer = tokenizer
    
    def forward(self, enc_in_ids, enc_att_mask, dec_in_ids, dec_att_mask):
        '''
        enc_in_ids: [batch_size, enc_seq_len]
        enc_att_mask: [batch_size, enc_seq_len], mask paddings
        dec_in_ids: [batch_size, dec_seq_len]
        dec_att_mask: [batch_size, dec_seq_len], mask paddings
        '''
        # hidden: [batch_size, dec_seq_len, bart_last_hidden_size]
        seq2seq_output = self.bart(input_ids=enc_in_ids,
                                   attention_mask=enc_att_mask,
                                   decoder_input_ids=dec_in_ids,
                                   decoder_attention_mask=dec_att_mask,
                                   use_cache=False)
        output = seq2seq_output.logits
        return output
    
    def inference(self, contexts, device, top_k=50, temperature=1, max_length=120, num_beams=1, do_sample=True):
        '''contexts: list of contexts in raw text form, with style token already inserted'''
        output_texts = []
        encs = self.tokenizer(contexts, truncation=True, padding=True, return_tensors='pt')
        encs = encs.to(device)
        story_ids = self.bart.generate(encs['input_ids'],
                                       attention_mask=encs['attention_mask'],
                                       num_beams=num_beams,
                                       max_length=max_length,
                                       temperature=temperature,
                                       top_k=top_k,
                                       do_sample=do_sample)
        raw_stories = [self.tokenizer.decode(story) for story in story_ids]
        output_texts = list(map(self.format_out_texts, raw_stories))
        return output_texts
    
    def format_out_texts(self, text):
        t_map = self.tokenizer.special_tokens_map
        for key in t_map:
            text = text.replace(t_map[key], '')
        return text
