#!/bin/bash

# TODO do qm9, pcba, muv, hiv, chembl separately later because they're bigger
datasets=('delaney' 'lipo' 'freesolv' 'pdbbind_full' 'pdbbind_core' 'pdbbind_refined' 'qm7' 'qm8' 'bace' 'bbbp' 'sider' 'clintox' 'tox21' 'toxcast')
dataset_type=('regression' 'regression' 'regression' 'regression' 'regression' 'regression' 'regression' 'regression' 'classification' 'classification' 'classification' 'classification' 'classification' 'classification' 'classification')
metrics=('rmse' 'rmse' 'rmse' 'rmse' 'rmse' 'rmse' 'mae' 'mae' 'auc' 'auc' 'auc' 'auc' 'auc' 'auc')

folds=(0 1 2 3 4 5 6 7 8 9)
gpu=0
num_gpus=4

for i in ${!datasets[@]}; do
    echo ${datasets[$i]}
    for fold in ${!folds[@]}; do
        echo ${folds[$fold]}
        CUDA_VISIBLE_DEVICES=$gpu python hyperparameter_optimization.py --data_path data/${datasets[$i]}.csv --dataset_type ${dataset_type[$i]} --split_type crossval --crossval_index_file crossval_index_files/${folds[$fold]}.pkl --crossval_index_dir crossval_folds/${datasets[$i]}/random --features_path /data/rsg/chemistry/yangk/saved_features/${datasets[$i]}.pckl --no_features_scaling --num_iters 50 --config_save_path ../ckpt/417_hyperopt/${datasets[$i]}/random/${folds[$fold]}/config.json --log_dir ../ckpt/417_hyperopt/${datasets[$i]}/random/${folds[$fold]}/logdir --quiet --metric ${metrics[$i]} &
        gpu=$((($gpu + 1) % $num_gpus))
    done
    wait
done