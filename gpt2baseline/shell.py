import torch
import argparse
import sys

import GPT
from data import get_style_token as gpt_style_token


def parse():
    parser = argparse.ArgumentParser(description="Generate story on test set")
    parser.add_argument('--model', type=str, required=True, help='Model saved path')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Device to run model')
    parser.add_argument('-t', '--temperature', type=float, default=1.0, help='Temperature when generation')
    parser.add_argument('--num', type=int, default=1, help='Number of stories when generating')

    return parser.parse_args()

def generate(model, context, label, device, args):
    style_token = gpt_style_token(label)
    context = [style_token + ' ' + context] * args.num
    return model.inference(context, device, temperature=args.temperature, do_sample=True)

def run(args):
    DEVICE = torch.device(args.device)
    model = torch.load(args.model, map_location='cpu')
    model = model.to(DEVICE)
    while True:
        context = input('>>> ').split(' ')
        if 'exit' in context[-1]:
            break
        try:
            label = int(context[-1])
        except ValueError:
            print('Please input an integer as label.')
            continue
        title = ' '.join(context[:-1])
        stories = generate(model, title, label, DEVICE, args)
        print('\n\n'.join(stories))

if __name__ == '__main__':
    args = parse()
    run(args)
