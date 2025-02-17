#!/bin/bash

NUM_GPU=4
#GPU_DEVICES='0,1,2,6'
GPU_DEVICES='0,1,2,3'

EXP_DIR='./exp/finetune_search/'
VOCAB_PATH='./pretrained_ckpt/bert-base-uncased-vocab.txt'
GLUE_DIR='./dataset/glue'

STUDENT_MODEL2='mt_mysupernet'

PRETRAIN_DIR1='./dataset/pretrain_data/wikipedia_nomask'
PRETRAIN_DIR2='./dataset/pretrain_data/wiki_book_nomask'
WIKI_DIR='./dataset/pretrain_data/wikipedia_nomask'
BOOK_DIR='./dataset/pretrain_data/bookcorpus_nomask'

#FFN_EXPR='[11,0,0,0,0,0,10,11,0,10,11,11]'
#FFN_EXPR='[11,11,0,11,0,0,0,11,11,0,11,0]'
FFN_EXPR='[3,3,3,3,3,3,3,3,3,3,3,3]'

FINETUNE_BS=$((256 / NUM_GPU))

if [ $((256 % NUM_GPU)) -ne 0 ]; then
  echo "NUM_GPU must be a factor of 256."
  exit 1
fi


TEACHER_PRETRAIN_PATH='./pretrained_ckpt/bert-base-uncased-pytorch_model.bin'
TEACHER_DOWNSTREAM_PATH1='./exp/train/bert_base/mnli/best_model.bin'
STUDENT_PRETRAIN_PATH1='./exp/pretrain/mysupernet/epoch0/ckpt_ep1.bin'  # Pre-trained mysupernet checkpoint for search stage 1, 2 (Wikipedia only)
#STUDENT_PRETRAIN_PATH2='./exp/pretrain/mysupernet/stage3/ckpt_ep10.bin'  # Pre-trained mysupernet checkpoint for search stage 3 (entirely pre-trained w/o weight sharing, Wikipedia + BooksCorpus)
#STUDENT_PRETRAIN_PATH2='./exp/pretrain/mysupernet/only0_0.25size_attnsupervise/ckpt_ep10.bin'  # Pre-trained mysupernet checkpoint for search stage 3 (with weight sharing, Wikipedia + BooksCorpus)
STUDENT_PRETRAIN_PATH2='./exp/pretrain/auto_bert/15.7M/ckpt_ep10.bin'
STUDENT_PRETRAIN_PATH2='./exp/pretrain/auto_bert/15.7M/ckpt_ep10.bin'
STUDENT_DOWNSTREAM_PATH='./exp/finetune_search/downstream/downstream_ckpt_ep10.bin'  # Fine-tuned mysupernet checkpoint for search stage 3 (with weight sharing)

# Search stage 3
# Pre-train with weight sharing (modify the sub-dir name into $STUDENT_PRETRAIN_PATH3 after training)
#bash dist_finetune_search.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --stage3 --exp_dir $EXP_DIR --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR --pretrain_dir1 $PRETRAIN_DIR1 --pretrain_dir2 $PRETRAIN_DIR2 --wiki_dir $WIKI_DIR --book_dir $BOOK_DIR --teacher_pretrain_path $TEACHER_PRETRAIN_PATH --teacher_downstream_path1 $TEACHER_DOWNSTREAM_PATH1 --student_pretrain_path2 $STUDENT_PRETRAIN_PATH2

# Fine-tune with weight sharing (modify the sub-dir name into $STUDENT_DOWNSTREAM_PATH after training)
#bash dist_finetune_search.sh $NUM_GPU --test 0 --dynamic 0 --gpu_devices $GPU_DEVICES --lowercase --stage3 --exp_dir $EXP_DIR --batch_size4 $FINETUNE_BS --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR --pretrain_dir1 $PRETRAIN_DIR1 --pretrain_dir2 $PRETRAIN_DIR2 --wiki_dir $WIKI_DIR --book_dir $BOOK_DIR --teacher_pretrain_path $TEACHER_PRETRAIN_PATH --teacher_downstream_path1 $TEACHER_DOWNSTREAM_PATH1 --student_pretrain_path2 $STUDENT_PRETRAIN_PATH2 --student_pretrain_path3 $STUDENT_PRETRAIN_PATH2
#python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_addr 127.0.0.36 --master_port 29537 \
#finetune_search.py --distributed --test 0 --min_pca 0 --student_model2 $STUDENT_MODEL2 --new_loss 1 --range1 0 --range2 1  --type_each_block 3 --dynamic 0 --gpu_devices $GPU_DEVICES --lowercase --stage3 \
#--exp_dir $EXP_DIR --batch_size4 $FINETUNE_BS --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR \
#--pretrain_dir1 $PRETRAIN_DIR1 --pretrain_dir2 $PRETRAIN_DIR2 --wiki_dir $WIKI_DIR --book_dir $BOOK_DIR \
#--teacher_pretrain_path $TEACHER_PRETRAIN_PATH --teacher_downstream_path1 $TEACHER_DOWNSTREAM_PATH1 \
#--student_pretrain_path2 $STUDENT_PRETRAIN_PATH2 --student_pretrain_path3 $STUDENT_PRETRAIN_PATH2

#python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_addr 127.0.0.36 --master_port 29537 \
#finetune_search.py --distributed --test 1 --min_pca 0 --student_model2 $STUDENT_MODEL2 --new_loss 1 --range1 0 --range2 1  --type_each_block 3 --dynamic 0 --gpu_devices $GPU_DEVICES --lowercase --ffn_expr $FFN_EXPR --stage3 \
#--exp_dir $EXP_DIR --batch_size4 $FINETUNE_BS --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR \
#--pretrain_dir1 $PRETRAIN_DIR1 --pretrain_dir2 $PRETRAIN_DIR2 --wiki_dir $WIKI_DIR --book_dir $BOOK_DIR \
#--teacher_pretrain_path $TEACHER_PRETRAIN_PATH --teacher_downstream_path1 $TEACHER_DOWNSTREAM_PATH1 \
#--student_pretrain_path2 $STUDENT_PRETRAIN_PATH2 --student_pretrain_path3 $STUDENT_PRETRAIN_PATH2

python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_addr 127.0.0.36 --master_port 29537 \
finetune_search.py --distributed --test 1 --min_pca 0 --student_model2 $STUDENT_MODEL2 --new_loss 0 --range1 0 --range2 1  --type_each_block 6 --dynamic 0 --gpu_devices $GPU_DEVICES --lowercase --ffn_expr $FFN_EXPR --stage3 \
--exp_dir $EXP_DIR --batch_size4 $FINETUNE_BS --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR \
--pretrain_dir1 $PRETRAIN_DIR1 --pretrain_dir2 $PRETRAIN_DIR2 --wiki_dir $WIKI_DIR --book_dir $BOOK_DIR \
--teacher_pretrain_path $TEACHER_PRETRAIN_PATH --teacher_downstream_path1 $TEACHER_DOWNSTREAM_PATH1 \
--student_pretrain_path2 $STUDENT_PRETRAIN_PATH1 --student_pretrain_path3 $STUDENT_PRETRAIN_PATH1

# Architecture search
#bash dist_finetune_search.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --stage3 --exp_dir $EXP_DIR --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR --pretrain_dir1 $PRETRAIN_DIR1 --pretrain_dir2 $PRETRAIN_DIR2 --wiki_dir $WIKI_DIR --book_dir $BOOK_DIR --teacher_pretrain_path $TEACHER_PRETRAIN_PATH --teacher_downstream_path1 $TEACHER_DOWNSTREAM_PATH1 --student_pretrain_path2 $STUDENT_PRETRAIN_PATH2 --student_pretrain_path3 $STUDENT_PRETRAIN_PATH3 --student_downstream_path $STUDENT_DOWNSTREAM_PATH
