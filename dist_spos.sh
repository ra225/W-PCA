#!/bin/bash
NUM_PROC=$1
shift
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_addr 127.0.0.38 --master_port 29539 \
spos.py --distributed "$@"
