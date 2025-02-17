
#!/bin/bash

NUM_GPU=2
GPU_DEVICES='1,2'

EXP_DIR='./exp/spos/'
VOCAB_PATH='./pretrained_ckpt/bert-base-uncased-vocab.txt'
GLUE_DIR='./dataset/glue'

PRETRAIN_DIR2='./dataset/pretrain_data/wiki_book_nomask'
WIKI_DIR='./dataset/pretrain_data/wikipedia_nomask'
BOOK_DIR='./dataset/pretrain_data/bookcorpus_nomask'
CONCATE_DATA_DIR='./dataset/pretrain_data/wiki_book_nomask'


BASE_SUPERNET_PATH='./exp/finetune_search/s_6blocks_noqk_10M/downstream_ckpt_ep10.bin'

# spos
#bash dist_spos.sh $NUM_GPU --gpu_devices $GPU_DEVICES --max_params 10000000 --base_model_name 'mt_mysupernet_10M'  --num_layers 6 --range1 0 --range2 1  --type_each_block 3 --lowercase --exp_dir $EXP_DIR --vocab_path $VOCAB_PATH --data_dir $GLUE_DIR --pretrain_dir2 $PRETRAIN_DIR2 --wiki_dir $WIKI_DIR --book_dir $BOOK_DIR --base_supernet_path $BASE_SUPERNET_PATH

#python -m torch.distributed.launch --seed 8888 --nproc_per_node=$NUM_GPU --master_addr 127.0.0.38 --master_port 29539 \
#spos.py --distributed --gpu_devices $GPU_DEVICES --max_params 10000000 --base_model_name 'mt_mysupernet_10M' \
# --num_layers 6 --range1 0 --range2 1  --type_each_block 3 --lowercase --exp_dir $EXP_DIR --vocab_path $VOCAB_PATH \
# --data_dir $GLUE_DIR --pretrain_dir2 $PRETRAIN_DIR2 --wiki_dir $WIKI_DIR --book_dir $BOOK_DIR \
# --base_supernet_path $BASE_SUPERNET_PATH

python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_addr 127.0.0.38 --master_port 29539 \
spos.py --distributed --gpu_devices $GPU_DEVICES --max_params 15700000 --base_model_name 'mt_mysupernet' \
 --num_layers 12 --range1 0 --range2 1  --type_each_block 6 --lowercase --exp_dir $EXP_DIR --vocab_path $VOCAB_PATH \
 --data_dir $GLUE_DIR --pretrain_dir2 $PRETRAIN_DIR2 --wiki_dir $WIKI_DIR --book_dir $BOOK_DIR --concate_data_dir $CONCATE_DATA_DIR \
  --base_supernet_path $BASE_SUPERNET_PATH --proxy_type 0 \
 #--ffn_expr $FFN_EXPR
