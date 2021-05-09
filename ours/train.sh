#!/bin/bash
factor=0.2

start_time=$(date +%s)

stdbuf -o 0 python train.py --model ours_model.pth --train_set ../data/ROCStories_train.csv\
    --dev_set ../data/ROCStories_dev.csv -lr 5e-5 -e 10 --batch-size 32 --device 0 --alpha ${factor} | tee ours_train_log.txt

end_time=$(date +%s)
elpased=$((${end_time}-${start_time}))
echo "Our model training time = $(($elpased/60)) min $(($elpased%60)) sec" > ours_train_time.txt

stdbuf -o 0 python test.py --model ours_model.pth --mode min --batch_size 64\
    --test_set ../data/ROCStories_test.csv | tee ours_test_log.txt
