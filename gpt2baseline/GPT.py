#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import transformers.tokenization_utils

class GPT2Gen(nn.Module):
    def __init__(self, tokenizer:GPT2Tokenizer):
        super().__init__()
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = tokenizer

        self.gpt.resize_token_embeddings(len(self.tokenizer))
    
    def forward(self, **tokenizer_input):
        '''
        @param labels: The id of golden truth, which usually the same as input_ids
        '''
        output = self.gpt(**tokenizer_input)
        return output

    def inference(self, context, device, top_k=50, temperature=1.0, do_sample=True, num_return_sequences=1):
        self.tokenizer.padding_side = 'left'    # Adding padding in the left
        inputs = self.tokenizer(context, padding=True, truncation=True, return_tensors='pt')
        inputs = inputs.to(device)
        output_sequences = self.gpt.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],    # Absolute pos embedding
            max_length=120,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=do_sample,
            num_beams=1
        )
        stories = [self.tokenizer.decode(sequence) for sequence in output_sequences]
        return list(map(self.format_out_texts, stories))

    def format_out_texts(self, text):
        t_map = self.tokenizer.special_tokens_map
        for key in t_map:
            text = text.replace(t_map[key], '')
        return text

