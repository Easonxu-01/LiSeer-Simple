cd $(readlink -f `dirname $0`)
conda activate OpenOccupancy
export PYTHONPATH="."

echo $1
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

# Restrict the run to a subset of GPUs, if needed:
# export CUDA_VISIBLE_DEVICES=0,1,2,3

bash tools/dist_train.sh $config $2 ${@:3}

