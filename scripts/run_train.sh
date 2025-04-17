#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

python train.py \
 --name db3 \
 --logdir ./logdir/db3 \
 --model hawaii \
 --netG resnet_9blocks \
 --dataroot ./datasets/horse2zebra \
 --dataset_mode unaligned \
 --freq_separation True \
 --lr 0.0002 \
 --n_epochs 200 \
 --n_epochs_decay 200 \
 --nce_layers 0,4,8,12,16 \
 --use_sa_layers True \
 --sa_blocks 3,5 \
 --lambda_NCE 1.5 \
 --lambda_idt 1.0 \
 --lambda_NCE_D 0.5 \
 --batch_size_dec 30 \
 --D_feat_layers 1
