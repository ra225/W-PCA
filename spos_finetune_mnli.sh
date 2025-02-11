#!/bin/bash

NUM_GPU=4
GPU_DEVICES='4,5,6,7'

VOCAB_PATH='./pretrained_ckpt/bert-base-uncased-vocab.txt'
WIKI_DIR='./dataset/pretrain_data/wikipedia_nomask'
BOOK_DIR='./dataset/pretrain_data/bookcorpus_nomask'
CONCATE_DATA_DIR='./dataset/pretrain_data/wiki_book_nomask'
GLUE_DIR='./dataset/glue'
SQUAD_DIR='./dataset/squad'

TEACHER_PRETRAIN_PATH='./pretrained_ckpt/bert-base-uncased-pytorch_model.bin'
TEACHER_PRETRAIN_PATH_MNLI='./exp/train/bert_base/mnli/best_model.bin'
TEACHER_PRETRAIN_PATH_QQP='./exp/train/bert_base/qqp/best_model.bin'
TEACHER_PRETRAIN_PATH_QNLI='./exp/train/bert_base/qnli/best_model.bin'
TEACHER_PRETRAIN_PATH_SST2='./exp/train/bert_base/sst-2/best_model.bin'
TEACHER_PRETRAIN_PATH_COLA='./exp/train/bert_base/cola/best_model.bin'
TEACHER_PRETRAIN_PATH_STSB='./exp/train/bert_base/sts-b/best_model.bin'
TEACHER_PRETRAIN_PATH_MRPC='./exp/train/bert_base/mrpc/best_model.bin'
TEACHER_PRETRAIN_PATH_RTE='./exp/train/bert_base/rte/best_model.bin'
TEACHER_PRETRAIN_PATH_WNLI='./exp/train/bert_base/wnli/best_model.bin'
TEACHER_PRETRAIN_PATH_SQUAD1='./exp/train/bert_base/squad1.1/best_model.bin'
TEACHER_PRETRAIN_PATH_SQUAD2='./exp/train/bert_base/squad2.0/best_model.bin'

#STUDENT_PRETRAIN_PATH='./exp/pretrain/mysupernet/stage3/ckpt_ep10.bin'  # Pre-trained supernet checkpoint in stage 3
STUDENT_PRETRAIN_PATH='./exp/finetune_search/s_6blocks/downstream_ckpt_ep10.bin'


PRETRAIN_LR=1e-4
PRETRAIN_TRAIN_RATIO=1
PRETRAIN_EPOCHS=10
PRETRAIN_BS=320
PRETRAIN_EXP_PATH='./exp/pretrain/auto_bert/'
DOWNSTREAM_EXP_PATH='./exp/downstream/auto_bert/'

GLUE_SEQ_LEN=128
GLUE_BS=8
#GLUE_BS=120
#GLUE_EPOCHS=10
#GLUE_EPOCHS_COLA=50
GLUE_EPOCHS=0
GLUE_EPOCHS_COLA=0
GLUE_LR=5e-5

SQUAD_SEQ_LEN=128
SQUAD_QLEN=64
SQUAD_TRUNCATE_STRIDE=32
SQUAD_BS=8
#SQUAD_BS=120
SQUAD_EPOCHS=10
SQUAD_LR=1e-4

HIDDEN_RATIO=1
PRED_RATIO=1



FFN_EXPR='[1,4,5,11,3,5,1,9,11,1,11,3]'


## EfficientBERT
STUDENT_MODEL='mysupernet'

STUDENT_PRETRAIN_PATH2='./exp/pretrain/auto_bert/params_68M/ckpt_ep10.bin'
STUDENT_PRETRAIN_PATH2='./exp/finetune_search/epoch0/downstream_ckpt_ep10.bin'
STUDENT_PRETRAIN_PATH3='./exp/finetune_search/epoch0/downstream_ckpt_ep10.bin'

## GLUE

bash dist_spos_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --new_loss 0 --type_blocks 12 --type_each_block 6 --val_freq 50  --exp_dir $DOWNSTREAM_EXP_PATH'/mnli/'  --ffn_expr $FFN_EXPR --student_pretrain_path $STUDENT_PRETRAIN_PATH2 --student_model $STUDENT_MODEL --task mnli  --lr $GLUE_LR --batch_size $GLUE_BS --total_epochs $GLUE_EPOCHS      --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR --max_seq_len $GLUE_SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_MNLI

bash dist_spos_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --new_loss 0 --type_blocks 12 --type_each_block 6 --val_freq 50  --exp_dir $DOWNSTREAM_EXP_PATH'/qnli/'  --ffn_expr $FFN_EXPR --student_pretrain_path $STUDENT_PRETRAIN_PATH3 --student_model $STUDENT_MODEL --task qnli  --lr $GLUE_LR --batch_size $GLUE_BS --total_epochs $GLUE_EPOCHS      --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR --max_seq_len $GLUE_SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_QNLI
bash dist_spos_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --new_loss 0 --type_blocks 12 --type_each_block 6 --val_freq 50  --exp_dir $DOWNSTREAM_EXP_PATH'/sst-2/' --ffn_expr $FFN_EXPR --student_pretrain_path $STUDENT_PRETRAIN_PATH3 --student_model $STUDENT_MODEL --task sst-2 --lr $GLUE_LR --batch_size $GLUE_BS --total_epochs $GLUE_EPOCHS      --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR --max_seq_len $GLUE_SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_SST2
bash dist_spos_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --new_loss 0 --type_blocks 12 --type_each_block 6 --val_freq 50  --exp_dir $DOWNSTREAM_EXP_PATH'/cola/'  --ffn_expr $FFN_EXPR --student_pretrain_path $STUDENT_PRETRAIN_PATH3 --student_model $STUDENT_MODEL --task cola  --lr $GLUE_LR --batch_size $GLUE_BS --total_epochs $GLUE_EPOCHS_COLA --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR --max_seq_len $GLUE_SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_COLA
bash dist_spos_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --new_loss 0 --type_blocks 12 --type_each_block 6 --val_freq 50  --exp_dir $DOWNSTREAM_EXP_PATH'/sts-b/' --ffn_expr $FFN_EXPR --student_pretrain_path $STUDENT_PRETRAIN_PATH3 --student_model $STUDENT_MODEL --task sts-b --lr $GLUE_LR --batch_size $GLUE_BS --total_epochs $GLUE_EPOCHS      --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR --max_seq_len $GLUE_SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_STSB
bash dist_spos_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --new_loss 0 --type_blocks 12 --type_each_block 6 --val_freq 50  --exp_dir $DOWNSTREAM_EXP_PATH'/mrpc/'  --ffn_expr $FFN_EXPR --student_pretrain_path $STUDENT_PRETRAIN_PATH3 --student_model $STUDENT_MODEL --task mrpc  --lr $GLUE_LR --batch_size $GLUE_BS --total_epochs $GLUE_EPOCHS      --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR --max_seq_len $GLUE_SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_MRPC
bash dist_spos_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --new_loss 0 --type_blocks 12 --type_each_block 6 --val_freq 50  --exp_dir $DOWNSTREAM_EXP_PATH'/rte/'   --ffn_expr $FFN_EXPR --student_pretrain_path $STUDENT_PRETRAIN_PATH3 --student_model $STUDENT_MODEL --task rte   --lr $GLUE_LR --batch_size $GLUE_BS --total_epochs $GLUE_EPOCHS      --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR --max_seq_len $GLUE_SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_RTE
bash dist_spos_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --new_loss 0 --type_blocks 12 --type_each_block 6 --val_freq 50  --exp_dir $DOWNSTREAM_EXP_PATH'/wnli/'  --ffn_expr $FFN_EXPR --student_pretrain_path $STUDENT_PRETRAIN_PATH3 --student_model $STUDENT_MODEL --task wnli  --lr $GLUE_LR --batch_size $GLUE_BS --total_epochs $GLUE_EPOCHS      --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR --max_seq_len $GLUE_SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_WNLI
bash dist_spos_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --new_loss 0 --type_blocks 12 --type_each_block 6 --val_freq 250 --exp_dir $DOWNSTREAM_EXP_PATH'/qqp/'   --ffn_expr $FFN_EXPR --student_pretrain_path $STUDENT_PRETRAIN_PATH3 --student_model $STUDENT_MODEL --task qqp   --lr $GLUE_LR --batch_size $GLUE_BS --total_epochs $GLUE_EPOCHS      --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR --max_seq_len $GLUE_SEQ_LEN --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_QQP

## SQuAD
bash dist_spos_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --new_loss 0 --type_blocks 12 --type_each_block 6 --val_freq 500 --exp_dir $DOWNSTREAM_EXP_PATH'/squad1.1/' --ffn_expr $FFN_EXPR --student_pretrain_path $STUDENT_PRETRAIN_PATH2 --student_model $STUDENT_MODEL --task squad1.1  --lr $SQUAD_LR --batch_size $SQUAD_BS --total_epochs $SQUAD_EPOCHS --vocab_path $VOCAB_PATH --data_dir $SQUAD_DIR --max_seq_len $SQUAD_SEQ_LEN --max_query_len $SQUAD_QLEN --trunc_stride $SQUAD_TRUNCATE_STRIDE --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_SQUAD1
bash dist_spos_finetune.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --new_loss 0 --type_blocks 12 --type_each_block 6 --val_freq 500 --exp_dir $DOWNSTREAM_EXP_PATH'/squad2.0/' --ffn_expr $FFN_EXPR --student_pretrain_path $STUDENT_PRETRAIN_PATH3 --student_model $STUDENT_MODEL --task squad2.0  --lr $SQUAD_LR --batch_size $SQUAD_BS --total_epochs $SQUAD_EPOCHS --vocab_path $VOCAB_PATH --data_dir $SQUAD_DIR --max_seq_len $SQUAD_SEQ_LEN --max_query_len $SQUAD_QLEN --trunc_stride $SQUAD_TRUNCATE_STRIDE --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --teacher_pretrain_path $TEACHER_PRETRAIN_PATH_SQUAD2
