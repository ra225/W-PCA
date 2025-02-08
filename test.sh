#!/bin/bash

GPU_DEVICES='7'
DATA_DIR='./dataset/glue'
VOCAB_DIR='./pretrained_ckpt/bert-base-uncased-vocab.txt'

MODEL_NAME='mysupernet'
CKPT_PATH='./exp/downstream/auto_bert/'
SAVE_DIR='test_results/params_68M/'
#FFN_EXPR='[1,0,0,2,0,0,1,0,1,0,2,1]'
#FFN_EXPR='[1,0,0,0,0,1,0,1,0,1,0,1]'
#FFN_EXPR='[16,16,1,7,1,6,0,10,9,12,16,13]'
#FFN_EXPR='[3,4,0,3,0,0,4,3,2,2,5,4]'
FFN_EXPR='[11,0,0,0,0,0,10,11,0,10,11,11]'
FFN_EXPR='[9,3,3,10,11,10,5,11,9,11,1,8]'
#15.7M synaptic_diversity
FFN_EXPR='[1,2,0,1,3,11,6,9,9,10,6,7]'
#68M
FFN_EXPR='[1,4,5,11,3,5,1,9,11,1,11,3]'
FINAL_HEADS='[12,4,6,6,6,4,12,12,6,8,11,12]'
EXP_DIR=$CKPT_PATH'/test_logs/'


python test.py --task mnli    --gpu_devices $GPU_DEVICES --lowercase --type_blocks 12 --type_each_block 6 --model_name $MODEL_NAME --data_dir $DATA_DIR --vocab_path $VOCAB_DIR --result_dir $SAVE_DIR --exp_dir $EXP_DIR --ffn_expr $FFN_EXPR --resume_path $CKPT_PATH'/mnli/params_68M/best_model.bin'
python test.py --task mnli-mm --gpu_devices $GPU_DEVICES --lowercase --type_blocks 12 --type_each_block 6 --model_name $MODEL_NAME --data_dir $DATA_DIR --vocab_path $VOCAB_DIR --result_dir $SAVE_DIR --exp_dir $EXP_DIR --ffn_expr $FFN_EXPR --resume_path $CKPT_PATH'/mnli/params_68M/best_model.bin'
python test.py --task ax      --gpu_devices $GPU_DEVICES --lowercase --type_blocks 12 --type_each_block 6 --model_name $MODEL_NAME --data_dir $DATA_DIR --vocab_path $VOCAB_DIR --result_dir $SAVE_DIR --exp_dir $EXP_DIR --ffn_expr $FFN_EXPR --resume_path $CKPT_PATH'/mnli/params_68M/best_model.bin'
python test.py --task qnli    --gpu_devices $GPU_DEVICES --lowercase --type_blocks 12 --type_each_block 6 --model_name $MODEL_NAME --data_dir $DATA_DIR --vocab_path $VOCAB_DIR --result_dir $SAVE_DIR --exp_dir $EXP_DIR --ffn_expr $FFN_EXPR --resume_path $CKPT_PATH'/qnli/params_68M/best_model.bin'
python test.py --task sst-2   --gpu_devices $GPU_DEVICES --lowercase --type_blocks 12 --type_each_block 6 --model_name $MODEL_NAME --data_dir $DATA_DIR --vocab_path $VOCAB_DIR --result_dir $SAVE_DIR --exp_dir $EXP_DIR --ffn_expr $FFN_EXPR --resume_path $CKPT_PATH'/sst-2/params_68M/best_model.bin'
python test.py --task cola    --gpu_devices $GPU_DEVICES --lowercase --type_blocks 12 --type_each_block 6 --model_name $MODEL_NAME --data_dir $DATA_DIR --vocab_path $VOCAB_DIR --result_dir $SAVE_DIR --exp_dir $EXP_DIR --ffn_expr $FFN_EXPR --resume_path $CKPT_PATH'/cola/params_68M/best_model.bin'
python test.py --task sts-b   --gpu_devices $GPU_DEVICES --lowercase --type_blocks 12 --type_each_block 6 --model_name $MODEL_NAME --data_dir $DATA_DIR --vocab_path $VOCAB_DIR --result_dir $SAVE_DIR --exp_dir $EXP_DIR --ffn_expr $FFN_EXPR --resume_path $CKPT_PATH'/sts-b/params_68M/best_model.bin'
python test.py --task mrpc    --gpu_devices $GPU_DEVICES --lowercase --type_blocks 12 --type_each_block 6 --model_name $MODEL_NAME --data_dir $DATA_DIR --vocab_path $VOCAB_DIR --result_dir $SAVE_DIR --exp_dir $EXP_DIR --ffn_expr $FFN_EXPR --resume_path $CKPT_PATH'/mrpc/params_68M/best_model.bin'
python test.py --task rte     --gpu_devices $GPU_DEVICES --lowercase --type_blocks 12 --type_each_block 6 --model_name $MODEL_NAME --data_dir $DATA_DIR --vocab_path $VOCAB_DIR --result_dir $SAVE_DIR --exp_dir $EXP_DIR --ffn_expr $FFN_EXPR --resume_path $CKPT_PATH'/rte/params_68M/best_model.bin'
python test.py --task wnli    --gpu_devices $GPU_DEVICES --lowercase --type_blocks 12 --type_each_block 6 --model_name $MODEL_NAME --data_dir $DATA_DIR --vocab_path $VOCAB_DIR --result_dir $SAVE_DIR --exp_dir $EXP_DIR --ffn_expr $FFN_EXPR --resume_path $CKPT_PATH'/wnli/params_68M/best_model.bin'
python test.py --task qqp     --gpu_devices $GPU_DEVICES --lowercase --type_blocks 12 --type_each_block 6 --model_name $MODEL_NAME --data_dir $DATA_DIR --vocab_path $VOCAB_DIR --result_dir $SAVE_DIR --exp_dir $EXP_DIR --ffn_expr $FFN_EXPR --resume_path $CKPT_PATH'/qqp/params_68M/best_model.bin'

