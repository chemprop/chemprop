#!/bin/bash

datasets=('delaney' 'lipo' 'freesolv' 'pdbbind_full' 'pdbbind_core' 'pdbbind_refined' 'qm7' 'qm8' 'qm9' 'bace' 'bbbp' 'sider' 'clintox' 'tox21' 'toxcast' 'muv' 'hiv' 'pcba' 'chembl')

for i in ${!datasets[@]}; do
    echo ${datasets[$i]}
    python scripts/create_crossval_splits.py --data_path data/${datasets[$i]}.csv --save_dir crossval_folds/${datasets[$i]} --split_type random
done