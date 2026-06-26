#!/usr/bin/env bash
###
 # @Author: EASON XU
 # @Date: 2023-12-07 07:39:57
 # @LastEditors: EASON XU
 # @Version: Do not edit
 # @LastEditTime: 2025-11-13 13:02:55
 # @Description: 头部注释
 # @FilePath: /UniLiDAR/tools/dist_test.sh
### 

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29501}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --deterministic --eval bbox
