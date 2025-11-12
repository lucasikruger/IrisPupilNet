#!/bin/bash

# First training run on MOBIUS dataset
# Using small parameters for testing

source ~/.workvenv/bin/activate

cd /home/agot-lkruger/tesis/IrisPupilNet

python irispupilnet/train.py \
  --dataset csv_seg \
  --data-root "/media/agot-lkruger/X9 Pro/facu/facu/tesis/MOBIUS" \
  --csv /home/agot-lkruger/tesis/IrisPupilNet/dataset/mobius_output/mobius_dataset.csv \
  --default-format mobius_3c \
  --model unet_se_small \
  --img-size 160 \
  --num-classes 3 \
  --base 32 \
  --batch-size 16 \
  --epochs 5 \
  --lr 1e-3 \
  --workers 2 \
  --out runs/first_test
