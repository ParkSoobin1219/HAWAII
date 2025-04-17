#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

python test.py \
 --dataroot ./datasets/horse2zebra \
 --results_dir ./results/horse2zebra/db/295 \
 --name db \
 --batch_size_dec 30 \
 --use_sa_layers True \
 --sa_blocks 3,5 \
 --epoch 295 \
 --model hawaii \
 --netG resnet_9blocks \
 --dataset_mode unaligned \
 --freq_separation True \
 --num_test 550
