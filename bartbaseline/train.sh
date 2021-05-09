#!/bin/bash
start_time=$(date +%s)

python train.py --train_set ../data/ROCStories_train.csv --dev_set ../data/ROCStories_dev.csv \
    -lr 5e-5 -e 10 --model bart_baseline.pth \
    --batch-size 32 > bart_train_log.txt

end_time=$(date +%s)
elpased=$((${end_time}-${start_time}))
echo "Bart training time = $(($elpased/60)) min $(($elpased%60)) sec" > bart_train_time.txt

python test.py --test_set ../data/ROCStories_test.csv --model ./bart_baseline.pth --mode min\
    --batch_size 64 > bart_test_log.txt

