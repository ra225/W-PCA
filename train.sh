#!/bin/bash

NUM_GPU=4
GPU_DEVICES='4,5,6,7'
MODEL_NAME='roberta_base'

GLUE_SEQ_LEN=128
GLUE_EPOCHS=3
GLUE_BS=8
GLUE_LR=3e-5
SEED=1

SQUAD_SEQ_LEN=128
SQUAD_QLEN=64
SQUAD_TRUNCATE_STRIDE=32
SQUAD_EPOCH=3
SQUAD_BS=8
SQUAD_LR=3e-5

#VOCAB_PATH='./pretrained_ckpt/bert-base-uncased-vocab.txt'
VOCAB_PATH='./pretrained_ckpt/roberta-base-vocab.json'
#PRETRAIN_PATH='./pretrained_ckpt/bert-base-uncased-pytorch_model.bin'
PRETRAIN_PATH='./pretrained_ckpt/roberta-base-pytorch_model.bin'
MERGE_PATH='./pretrained_ckpt/merges.txt'
#PRETRAIN_PATH_MNLI='./exp/train/bert_base/mnli/best_model.bin'
PRETRAIN_PATH_MNLI='./exp/train/roberta_base/mnli/best_model.bin'
GLUE_DIR='./dataset/glue'
SQUAD_DIR='./dataset/squad'
#EXP_DIR='./exp/train/bert_base/'
EXP_DIR='./exp/train/roberta_base/'


# GLUE
bash dist_train.sh $NUM_GPU --gpu_devices $GPU_DEVICES --seed $SEED --val_freq 50  --model_name $MODEL_NAME --lowercase --task mnli  --lr $GLUE_LR --batch_size $GLUE_BS --total_epochs $GLUE_EPOCHS --data_dir $GLUE_DIR --max_seq_len $GLUE_SEQ_LEN --vocab_path $VOCAB_PATH --merge_path $MERGE_PATH --pretrain_path $PRETRAIN_PATH --exp_dir $EXP_DIR
bash dist_train.sh $NUM_GPU --gpu_devices $GPU_DEVICES --seed $SEED --val_freq 50  --model_name $MODEL_NAME --lowercase --task qnli  --lr $GLUE_LR --batch_size $GLUE_BS --total_epochs $GLUE_EPOCHS --data_dir $GLUE_DIR --max_seq_len $GLUE_SEQ_LEN --vocab_path $VOCAB_PATH --merge_path $MERGE_PATH --pretrain_path $PRETRAIN_PATH_MNLI --exp_dir $EXP_DIR
bash dist_train.sh $NUM_GPU --gpu_devices $GPU_DEVICES --seed $SEED --val_freq 50  --model_name $MODEL_NAME --lowercase --task sst-2 --lr $GLUE_LR --batch_size $GLUE_BS --total_epochs $GLUE_EPOCHS --data_dir $GLUE_DIR --max_seq_len $GLUE_SEQ_LEN --vocab_path $VOCAB_PATH --merge_path $MERGE_PATH --pretrain_path $PRETRAIN_PATH_MNLI --exp_dir $EXP_DIR
bash dist_train.sh $NUM_GPU --gpu_devices $GPU_DEVICES --seed $SEED --val_freq 50  --model_name $MODEL_NAME --lowercase --task cola  --lr $GLUE_LR --batch_size $GLUE_BS --total_epochs $GLUE_EPOCHS --data_dir $GLUE_DIR --max_seq_len $GLUE_SEQ_LEN --vocab_path $VOCAB_PATH --merge_path $MERGE_PATH --pretrain_path $PRETRAIN_PATH_MNLI --exp_dir $EXP_DIR
bash dist_train.sh $NUM_GPU --gpu_devices $GPU_DEVICES --seed $SEED --val_freq 50  --model_name $MODEL_NAME --lowercase --task sts-b --lr $GLUE_LR --batch_size $GLUE_BS --total_epochs $GLUE_EPOCHS --data_dir $GLUE_DIR --max_seq_len $GLUE_SEQ_LEN --vocab_path $VOCAB_PATH --merge_path $MERGE_PATH --pretrain_path $PRETRAIN_PATH_MNLI --exp_dir $EXP_DIR
bash dist_train.sh $NUM_GPU --gpu_devices $GPU_DEVICES --seed $SEED --val_freq 50  --model_name $MODEL_NAME --lowercase --task mrpc  --lr $GLUE_LR --batch_size $GLUE_BS --total_epochs $GLUE_EPOCHS --data_dir $GLUE_DIR --max_seq_len $GLUE_SEQ_LEN --vocab_path $VOCAB_PATH --merge_path $MERGE_PATH --pretrain_path $PRETRAIN_PATH_MNLI --exp_dir $EXP_DIR
bash dist_train.sh $NUM_GPU --gpu_devices $GPU_DEVICES --seed $SEED --val_freq 50  --model_name $MODEL_NAME --lowercase --task rte   --lr $GLUE_LR --batch_size $GLUE_BS --total_epochs $GLUE_EPOCHS --data_dir $GLUE_DIR --max_seq_len $GLUE_SEQ_LEN --vocab_path $VOCAB_PATH --merge_path $MERGE_PATH --pretrain_path $PRETRAIN_PATH_MNLI --exp_dir $EXP_DIR
bash dist_train.sh $NUM_GPU --gpu_devices $GPU_DEVICES --seed $SEED --val_freq 50  --model_name $MODEL_NAME --lowercase --task wnli  --lr $GLUE_LR --batch_size $GLUE_BS --total_epochs $GLUE_EPOCHS --data_dir $GLUE_DIR --max_seq_len $GLUE_SEQ_LEN --vocab_path $VOCAB_PATH --merge_path $MERGE_PATH --pretrain_path $PRETRAIN_PATH_MNLI --exp_dir $EXP_DIR
bash dist_train.sh $NUM_GPU --gpu_devices $GPU_DEVICES --seed $SEED --val_freq 250 --model_name $MODEL_NAME --lowercase --task qqp   --lr $GLUE_LR --batch_size $GLUE_BS --total_epochs $GLUE_EPOCHS --data_dir $GLUE_DIR --max_seq_len $GLUE_SEQ_LEN --vocab_path $VOCAB_PATH --merge_path $MERGE_PATH --pretrain_path $PRETRAIN_PATH_MNLI --exp_dir $EXP_DIR
a = '''
'''
# SQuAD
bash dist_train.sh $NUM_GPU --gpu_devices $GPU_DEVICES --seed $SEED --val_freq 500 --model_name $MODEL_NAME --lowercase --task squad1.1 --lr $SQUAD_LR --batch_size $SQUAD_BS --total_epochs $SQUAD_EPOCH --data_dir $SQUAD_DIR --max_seq_len $SQUAD_SEQ_LEN --max_query_len $SQUAD_QLEN --trunc_stride $SQUAD_TRUNCATE_STRIDE --vocab_path $VOCAB_PATH --merge_path $MERGE_PATH --pretrain_path $PRETRAIN_PATH --exp_dir $EXP_DIR
bash dist_train.sh $NUM_GPU --gpu_devices $GPU_DEVICES --seed $SEED --val_freq 500 --model_name $MODEL_NAME --lowercase --task squad2.0 --lr $SQUAD_LR --batch_size $SQUAD_BS --total_epochs $SQUAD_EPOCH --data_dir $SQUAD_DIR --max_seq_len $SQUAD_SEQ_LEN --max_query_len $SQUAD_QLEN --trunc_stride $SQUAD_TRUNCATE_STRIDE --vocab_path $VOCAB_PATH --merge_path $MERGE_PATH --pretrain_path $PRETRAIN_PATH_MNLI --exp_dir $EXP_DIR
