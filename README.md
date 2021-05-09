# Stylized Story Generation with Style-Guided Planning

This repository contains the pytorch implementation of "Stylized Story Generation with Style-Guided Planning".

Xiangzhe Kong∗, Jialiang Huang∗, Ziquan Tung, Jian Guan and Minlie Huang†

## Code description

- Code for extracting keywords and annotating the dataset are in `data` .
- Code for GPT2/Bart/Our models are in `gpt2baseline` / `bartbaseline` / `ours` respectively.
- Code for calculating BLEU / SSC / LSC are in `analyze`.

## Run
We provide a shell script that incorporate both training and testing procedure.

- To train four models  (GPT2, Bart, Ours, Bert for SSC) at the same time (4 GPUs are needed)

```
bash run.sh parallel
```

- To train four models sequentially, using only one GPU

```
bash run.sh serial
```

Logs about the training and testing will be saved in the directory of each model respectively.

Besides, BLEU scores are saved in `analyze/bleu` and LSC/SSC scores are saved in `analyze/LSC_SSC`.