cd $(readlink -f `dirname $0`)
###
 # @Author: EASON XU
 # @Date: 2023-12-07 01:49:12
 # @LastEditors: EASON XU
 # @Version: Do not edit
 # @LastEditTime: 2024-11-25 07:36:58
 # @Description: 头部注释
 # @FilePath: /UniLiDAR/run.sh
### 
conda activate OpenOccupancy
export PYTHONPATH="."

echo $1
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

# # 设置CUDA_VISIBLE_DEVICES环境变量使只有前两个GPU对当前进程可见
# export CUDA_VISIBLE_DEVICES=4,5,6,7

bash tools/dist_train.sh $config $2 ${@:3}

