#!/bin/bash
python train.py --train_set ../../../data/ROCStories_train.csv --dev_set ../../../data/ROCStories_dev.csv \
    -lr 5e-5 -e 10 --batch-size 32 --model bert.pth > classifier_train_log.txt

echo "Classifier training over: $(date +'%F %H:%M:%S')"
