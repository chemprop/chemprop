#!/bin/bash -x

# TODO do pcba, chembl separately later because they're bigger
datasets=('delaney' 'lipo' 'freesolv' 'pdbbind_full' 'pdbbind_core' 'pdbbind_refined' 'qm7' 'qm8' 'bace' 'bbbp' 'sider' 'clintox' 'tox21' 'toxcast' 'hiv' 'muv' 'qm9') # 'pcba' 'chembl')
dataset_type=('regression' 'regression' 'regression' 'regression' 'regression' 'regression' 'regression' 'regression' 'classification' 'classification' 'classification' 'classification' 'classification' 'classification' 'classification' 'classification' 'regression') # 'classification' 'classification')
metrics=('rmse' 'rmse' 'rmse' 'rmse' 'rmse' 'rmse' 'mae' 'mae' 'auc' 'auc' 'auc' 'auc' 'auc' 'auc' 'auc' 'prc-auc' 'mae') # 'auc' 'auc')
sizes=('one' 'one' 'one' 'one' 'one' 'one' 'one' 'one' 'one' 'one' 'one' 'one' 'one' 'one' 'big' 'big' 'big') # 'big' 'big')
# datasets=('delaney')
# dataset_type=('regression')
# metrics=('rmse')
# sizes=('one')

split_type="scaffold"

folds=(0 1 2 3 4 5 6 7 8 9)
gpus=(0 1 2)
num_gpus=${#gpus[@]}
gpu_index=0

# no ensemble or hyperopt
for i in ${!datasets[@]}; do
    echo ${datasets[$i]}
    for fold in ${!folds[@]}; do
        echo ${folds[$fold]}
        if [[ "${split_type}" == "random" ]]; then
            file="/data/rsg/chemistry/swansonk/chemprop/crossval_index_files/${sizes[$i]}/${folds[$fold]}_test.pkl"
            split_info="--split_type crossval --crossval_index_file $file --crossval_index_dir crossval_folds/${datasets[$i]}/random"
        else
            file="/data/rsg/chemistry/yangk/lsc_experiments_dump_splits/data/${datasets[$i]}/scaffold/fold_$fold/0/split_indices.pckl"
            split_info="--split_type predetermined --folds_file $file --val_fold_index 1 --test_fold_index 2"
        fi
        if [[ ! -e "$file" ]]; then
            echo "Fold indices do not exist" # you should expect this to happen when not testing on all 10 folds
        else 
            CUDA_VISIBLE_DEVICES=${gpus[${gpu_index}]} python train.py --data_path data/${datasets[$i]}.csv --dataset_type ${dataset_type[$i]} --save_dir ../ckpt/417_default/${datasets[$i]}/${split_type}/${folds[$fold]} ${split_info} --quiet --metric ${metrics[$i]} &
            CUDA_VISIBLE_DEVICES=${gpus[${gpu_index}]} python train.py --data_path data/${datasets[$i]}.csv --dataset_type ${dataset_type[$i]} --save_dir ../ckpt/417_features_no_opt/${datasets[$i]}/${split_type}/${folds[$fold]} ${split_info} --features_path /data/rsg/chemistry/yangk/saved_features/${datasets[$i]}.pckl --no_features_scaling --quiet --metric ${metrics[$i]} &
            gpu_index=$(($((${gpu_index} + 1)) % ${num_gpus}))
        fi
    done
    wait
done
