#!/bin/bash -x

datasets=('delaney' 'lipo' 'freesolv' 'pdbbind_full' 'pdbbind_core' 'pdbbind_refined' 'qm7' 'qm8' 'bace' 'bbbp' 'sider' 'clintox' 'hiv' 'qm9' 'tox21')
dataset_type=('regression' 'regression' 'regression' 'regression' 'regression' 'regression' 'regression' 'regression' 'classification' 'classification' 'classification' 'classification' 'classification' 'regression' 'classification')
metrics=('rmse' 'rmse' 'rmse' 'rmse' 'rmse' 'rmse' 'mae' 'mae' 'auc' 'auc' 'auc' 'auc' 'auc' 'mae' 'auc')
sizes=('one' 'one' 'one' 'one' 'one' 'one' 'one' 'one' 'one' 'one' 'one' 'one' 'big' 'big' 'one')
# datasets=('delaney')
# dataset_type=('regression')
# metrics=('rmse')
# sizes=('one')

split_type="scaffold"

folds=(0 1 2 3 4 5 6 7 8 9)

# FFN baselines
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
        if [[ "${datasets[$i]}" == "tox21" ]]; then
            single_task="--single_task"
        else
            single_task=""
        fi
        if [[ ! -e "$file" ]]; then
            echo "Fold indices do not exist" # you should expect this to happen when not testing on all 10 folds
        else
            python random_forest.py ${single_task} --data_path data/${datasets[$i]}.csv --dataset_type ${dataset_type[$i]} --save_dir ../ckpt/417_random_forest/${datasets[$i]}/${split_type}/${folds[$fold]} ${split_info} --quiet --metric ${metrics[$i]}
        fi
    done
done
