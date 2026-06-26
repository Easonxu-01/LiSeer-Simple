cd $(readlink -f `dirname $0`)
###
 # @Author: EASON XU
 # @Date: 2023-12-07 01:49:10
 # @LastEditors: EASON XU
 # @Version: Do not edit
 # @LastEditTime: 2024-11-25 07:37:11
 # @Description: 头部注释
 # @FilePath: /UniLiDAR/run_eval.sh
### 
conda activate OpenOccupancy

echo $1
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

export PYTHONPATH="."

ckpt=$2
gpu=$3

# # 设置CUDA_VISIBLE_DEVICES环境变量使只有前两个GPU对当前进程可见
# export CUDA_VISIBLE_DEVICES=0,1,2,3

bash tools/dist_test.sh $config $ckpt $gpu ${@:4}