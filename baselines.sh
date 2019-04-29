#!/bin/bash

# TODO do qm9, pcba, muv, hiv, chembl separately later because they're bigger
datasets=('delaney' 'lipo' 'freesolv' 'pdbbind_full' 'pdbbind_core' 'pdbbind_refined' 'qm7' 'qm8' 'bace' 'bbbp' 'sider' 'clintox' 'tox21' 'toxcast' 'qm9' 'pcba' 'muv' 'hiv' 'chembl')
dataset_type=('regression' 'regression' 'regression' 'regression' 'regression' 'regression' 'regression' 'regression' 'classification' 'classification' 'classification' 'classification' 'classification' 'classification' 'regression' 'classification' 'classification' 'classification' 'classification')
metrics=('rmse' 'rmse' 'rmse' 'rmse' 'rmse' 'rmse' 'mae' 'mae' 'auc' 'auc' 'auc' 'auc' 'auc' 'auc' 'mae' 'prc-auc' 'prc-auc' 'auc' 'auc')
sizes=('one' 'one' 'one' 'one' 'one' 'one' 'one' 'one' 'one' 'one' 'one' 'one' 'one' 'one' 'big' 'big' 'big' 'one' 'big')
# datasets=('delaney')
# dataset_type=('regression')
# metrics=('rmse')
# sizes=('one')

folds=(0 1 2 3 4 5 6 7 8 9)
gpus=(0, 1)
num_gpus=${#gpus[@]}
gpu_index=0

# FFN baselines
for i in ${!datasets[@]}; do
    echo ${datasets[$i]}
    for fold in ${!folds[@]}; do
        echo ${folds[$fold]}
        file=./crossval_index_files/${sizes[$i]}/${folds[$fold]}_test.pkl
        if [[ ! -e "$file" ]]; then
            echo "Fold indices do not exist" # you should expect this to happen when not testing on all 10 folds
        else
            CUDA_VISIBLE_DEVICES=${gpus[${gpu_index}]} python train.py --data_path data/${datasets[$i]}.csv --dataset_type ${dataset_type[$i]} --save_dir ../ckpt/417_ffn_morgan/${datasets[$i]}/random/${folds[$fold]} --split_type crossval --crossval_index_file crossval_index_files/one/${folds[$fold]}_test.pkl --crossval_index_dir crossval_folds/${datasets[$i]}/random --features_only --features_generator morgan --quiet --metric ${metrics[$i]} &
            CUDA_VISIBLE_DEVICES=${gpus[${gpu_index}]} python train.py --data_path data/${datasets[$i]}.csv --dataset_type ${dataset_type[$i]} --save_dir ../ckpt/417_ffn_morgan_count/${datasets[$i]}/random/${folds[$fold]} --split_type crossval --crossval_index_file crossval_index_files/one/${folds[$fold]}_test.pkl --crossval_index_dir crossval_folds/${datasets[$i]}/random --features_only --features_generator morgan_count --quiet --metric ${metrics[$i]} &
            CUDA_VISIBLE_DEVICES=${gpus[${gpu_index}]} python train.py --data_path data/${datasets[$i]}.csv --dataset_type ${dataset_type[$i]} --save_dir ../ckpt/417_ffn_rdkit/${datasets[$i]}/random/${folds[$fold]} --split_type crossval --crossval_index_file crossval_index_files/one/${folds[$fold]}_test.pkl --crossval_index_dir crossval_folds/${datasets[$i]}/random --features_only --features_path /data/rsg/chemistry/yangk/saved_features/${datasets[$i]}.pckl --no_features_scaling --quiet --metric ${metrics[$i]} &
        fi
        gpu_index=$(($((${gpu_index} + 1)) % ${num_gpus}))
    done
    wait
done
