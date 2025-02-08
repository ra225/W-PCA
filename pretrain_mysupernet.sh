#!/bin/bash

NUM_GPU=4
GPU_DEVICES='2,3,4,5'

VOCAB_PATH='./pretrained_ckpt/bert-base-uncased-vocab.txt'
WIKI_DIR='./dataset/pretrain_data/wikipedia_nomask'
BOOK_DIR='./dataset/pretrain_data/bookcorpus_nomask'
CONCATE_DATA_DIR='./dataset/pretrain_data/wiki_book_nomask'

STUDENT_MODEL='mysupernet'
TEACHER_PRETRAIN_PATH='./pretrained_ckpt/bert-base-uncased-pytorch_model.bin'

PRETRAIN_LR=1e-4
PRETRAIN_TRAIN_RATIO=1
PRETRAIN_EPOCHS=10
PRETRAIN_BS=$((256 / NUM_GPU))

if [ $((256 % NUM_GPU)) -ne 0 ]; then
  echo "NUM_GPU must be a factor of 256."
  exit 1
fi

PRETRAIN_EXP_PATH='./exp/pretrain/mysupernet/'

#STUDENT_RESUME_PATH='/efficient-bert/exp/pretrain/mysupernet/pretrain_only0_1/ckpt_ep4.bin'


python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_addr 127.0.0.9 --master_port 29509 \
pretrain.py --distributed --test 1  --dynamic_heads 0 --min_pca 0 --new_loss 0 --range1 0 --range2 1  --type_each_block 3 --dynamic 0 --gpu_devices $GPU_DEVICES --lowercase --student_model $STUDENT_MODEL \
--train_ratio $PRETRAIN_TRAIN_RATIO --total_epochs $PRETRAIN_EPOCHS --batch_size $PRETRAIN_BS \
--lr $PRETRAIN_LR --teacher_pretrain_path $TEACHER_PRETRAIN_PATH --vocab_path $VOCAB_PATH \
--wiki_dir $WIKI_DIR --book_dir $BOOK_DIR --concate_data_dir $CONCATE_DATA_DIR --exp_dir $PRETRAIN_EXP_PATH