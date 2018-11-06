#!/bin/bash

SEED=$1
HIDDEN=300
BASE=MPN-s"$SEED"h"$HIDDEN"
echo "random seed $SEED"
mkdir -p $BASE

gunzip data/*.gz

#Small datasets
echo "training delaney"
mkdir -p $BASE/delaney
python ../main.py --data_path data/delaney.csv --dataset_type regression --hidden_size $HIDDEN --save_dir $BASE/delaney --seed $SEED --batch_size 10 | tee $BASE/delaney/LOG

echo "training freesolv"
mkdir -p $BASE/freesolv
python ../main.py --data_path data/freesolv.csv --dataset_type regression --hidden_size $HIDDEN --save_dir $BASE/freesolv --seed $SEED --batch_size 10 | tee $BASE/freesolv/LOG

echo "training lipo"
mkdir -p $BASE/lipo
python ../main.py --data_path data/lipo.csv --dataset_type regression --hidden_size $HIDDEN --save_dir $BASE/lipo --seed $SEED --batch_size 10 | tee $BASE/lipo/LOG

echo "training qm8"
mkdir -p $BASE/qm8
python ../main.py --data_path data/qm8.csv --dataset_type regression --metric mae --hidden_size $HIDDEN --save_dir $BASE/qm8 --seed $SEED | tee $BASE/qm8/LOG

echo "training qm9"
mkdir -p $BASE/qm9
python ../main.py --data_path data/qm9.csv --dataset_type regression --metric mae --hidden_size 1200 --save_dir $BASE/qm9 --seed $SEED | tee $BASE/qm9/LOG

echo "training bace"
mkdir -p $BASE/bace
python ../main.py --data_path data/bace.csv --dataset_type classification --hidden_size $HIDDEN --save_dir $BASE/bace --seed $SEED --batch_size 10 --split scaffold | tee $BASE/bace/LOG

echo "training BBBP"
mkdir -p $BASE/BBBP
python ../main.py --data_path data/BBBP.csv --dataset_type classification --hidden_size $HIDDEN --save_dir $BASE/BBBP --seed $SEED --batch_size 10 --split scaffold | tee $BASE/BBBP/LOG

echo "training tox21"
mkdir -p $BASE/tox21
python ../main.py --data_path data/tox21.csv --dataset_type classification --hidden_size $HIDDEN --save_dir $BASE/tox21 --seed $SEED | tee $BASE/tox21/LOG

echo "training toxcast"
mkdir -p $BASE/toxcast
python ../main.py --data_path data/toxcast.csv --dataset_type classification --hidden_size $HIDDEN --save_dir $BASE/toxcast --seed $SEED | tee $BASE/toxcast/LOG

echo "training sider"
mkdir -p $BASE/sider
python ../main.py --data_path data/sider.csv --dataset_type classification --hidden_size $HIDDEN --save_dir $BASE/sider --seed $SEED --batch_size 10 | tee $BASE/sider/LOG

echo "training clintox"
mkdir -p $BASE/clintox
python ../main.py --data_path data/clintox.csv --dataset_type classification --hidden_size $HIDDEN --save_dir $BASE/clintox --seed $SEED --batch_size 10 | tee $BASE/clintox/LOG

#Large datasets
echo "training MUV"
mkdir -p $BASE/muv
python ../main.py --data_path data/muv.csv --dataset_type classification --hidden_size $HIDDEN --save_dir $BASE/muv --seed $SEED --metric prc | tee $BASE/muv/LOG

echo "training HIV"
mkdir -p $BASE/HIV
python ../main.py --data_path data/HIV.csv --dataset_type classification --hidden_size $HIDDEN --save_dir $BASE/HIV --seed $SEED --split scaffold | tee $BASE/HIV/LOG

echo "training pcba"
mkdir -p $BASE/pcba
python ../main.py --data_path data/pcba.csv --dataset_type classification --hidden_size $HIDDEN --save_dir $BASE/pcba --seed $SEED --metric prc --anneal 50000 | tee $BASE/pcba/LOG
