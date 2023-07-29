#!/bin/sh

export PYTHONPATH=./
# eval "$(conda shell.bash hook)"
PYTHON=python

TRAIN_CODE=train_vkitti_sp.py
# TEST_CODE=test.py

dataset=$1
exp_name=$2

# result_dir=${exp_dir}/result
# config=config/${dataset}/${dataset}.yaml
config=$3
cvfold=$4
fold=cv0${cvfold}

exp_dir=exp/${dataset}/${fold}/${exp_name}
model_dir=${exp_dir}/model
visual_dir=${exp_dir}/visual

mkdir -p ${model_dir}
mkdir -p ${visual_dir}
# mkdir -p ${result_dir}/last
# mkdir -p ${result_dir}/best
cp tool/train_vkitti.sh tool/${TRAIN_CODE} ${config} ${exp_dir}


# now=$(date +"%Y%m%d_%H%M%S")
$PYTHON ${exp_dir}/${TRAIN_CODE} \
  --config=${config} \
  save_path ${exp_dir} \
  cvfold ${cvfold}
  # 2>&1 | tee ${exp_dir}/train-$now.log