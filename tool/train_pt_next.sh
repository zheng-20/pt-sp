#!/bin/sh

export PYTHONPATH=./
# eval "$(conda shell.bash hook)"
PYTHON=python

TRAIN_CODE=train_pt_next.py
# TEST_CODE=test.py

dataset=$1
exp_name=$2
exp_dir=exp/${dataset}_prim_boundary_aggre/${exp_name}
model_dir=${exp_dir}/model
# result_dir=${exp_dir}/result
# config=config/${dataset}/${dataset}.yaml
config=$3

mkdir -p ${model_dir}
# mkdir -p ${result_dir}/last
# mkdir -p ${result_dir}/best
cp tool/train_pt_next.sh tool/${TRAIN_CODE} ${config} ${exp_dir}


# now=$(date +"%Y%m%d_%H%M%S")
$PYTHON ${exp_dir}/${TRAIN_CODE} \
  --config=${config} \
  save_path ${exp_dir}
  # 2>&1 | tee ${exp_dir}/train-$now.log
