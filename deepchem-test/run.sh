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
python nnregress.py --train data/delaney.csv --hidden $HIDDEN --save_dir $BASE/delaney --seed $SEED --batch 10 | tee $BASE/delaney/LOG

echo "training freesolv"
mkdir -p $BASE/freesolv
python nnregress.py --train data/freesolv.csv --hidden $HIDDEN --save_dir $BASE/freesolv --seed $SEED --batch 10 | tee $BASE/freesolv/LOG

echo "training lipo"
mkdir -p $BASE/lipo
python nnregress.py --train data/lipo.csv --hidden $HIDDEN --save_dir $BASE/lipo --seed $SEED --batch 10 | tee $BASE/lipo/LOG

echo "training qm8"
mkdir -p $BASE/qm8
python nnregress_qm.py --train data/qm8.csv --hidden $HIDDEN --save_dir $BASE/qm8 --seed $SEED | tee $BASE/qm8/LOG

echo "training qm9"
mkdir -p $BASE/qm9
python nnregress_qm.py --train data/qm9.csv --hidden 1200 --save_dir $BASE/qm9 --seed $SEED | tee $BASE/qm9/LOG

echo "training bace"
mkdir -p $BASE/bace
python nnclassify.py --train data/bace.csv --hidden $HIDDEN --save_dir $BASE/bace --seed $SEED --batch 10 --split scaffold | tee $BASE/bace/LOG

echo "training BBBP"
mkdir -p $BASE/BBBP
python nnclassify.py --train data/BBBP.csv --hidden $HIDDEN --save_dir $BASE/BBBP --seed $SEED --batch 10 --split scaffold | tee $BASE/BBBP/LOG

echo "training tox21"
mkdir -p $BASE/tox21
python nnclassify.py --train data/tox21.csv --hidden $HIDDEN --save_dir $BASE/tox21 --seed $SEED | tee $BASE/tox21/LOG

echo "training toxcast"
mkdir -p $BASE/toxcast
python nnclassify.py --train data/toxcast.csv --hidden $HIDDEN --save_dir $BASE/toxcast --seed $SEED | tee $BASE/toxcast/LOG

echo "training sider"
mkdir -p $BASE/sider
python nnclassify.py --train data/sider.csv --hidden $HIDDEN --save_dir $BASE/sider --seed $SEED --batch 10 | tee $BASE/sider/LOG

echo "training clintox"
mkdir -p $BASE/clintox
python nnclassify.py --train data/clintox.csv --hidden $HIDDEN --save_dir $BASE/clintox --seed $SEED --batch 10 | tee $BASE/clintox/LOG

#Large datasets
echo "training MUV"
mkdir -p $BASE/muv
python nnclassify.py --train data/muv.csv --hidden $HIDDEN --save_dir $BASE/muv --seed $SEED --metric prc | tee $BASE/muv/LOG

echo "training HIV"
mkdir -p $BASE/HIV
python nnclassify.py --train data/HIV.csv --hidden $HIDDEN --save_dir $BASE/HIV --seed $SEED --split scaffold | tee $BASE/HIV/LOG

echo "training pcba"
mkdir -p $BASE/pcba
python nnclassify.py --train data/pcba.csv --hidden $HIDDEN --save_dir $BASE/pcba --seed $SEED --metric prc --anneal 50000 | tee $BASE/pcba/LOG
