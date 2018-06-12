#!/bin/bash

SEED=1
HIDDEN=300
BASE=MPNs"$SEED"h"$HIDDEN"
echo "random seed $SEED"

#Small datasets
echo "training delaney"
python nnregress.py --train data/delaney.txt --hidden $HIDDEN --save_dir $BASE/delaney --seed $SEED --batch 10 | tee $BASE/delaney/LOG

echo "training freesolv"
python nnregress.py --train data/freesolv.txt --hidden $HIDDEN --save_dir $BASE/freesolv --seed $SEED --batch 10 | tee $BASE/freesolv/LOG

echo "training lipo"
python nnregress.py --train data/lipo.txt --hidden $HIDDEN --save_dir $BASE/lipo --seed $SEED --batch 10 | tee $BASE/lipo/LOG

echo "training bace"
python nnclassify.py --train data/bace.csv --hidden $HIDDEN --save_dir $BASE/bace --seed $SEED --batch 10 --split scaffold | tee $BASE/bace/LOG

echo "training BBBP"
python nnclassify.py --train data/BBBP.csv --hidden $HIDDEN --save_dir $BASE/BBBP --seed $SEED --batch 10 --split scaffold | tee $BASE/BBBP/LOG

echo "training tox21"
python nnclassify.py --train data/tox21.csv --hidden $HIDDEN --save_dir $BASE/tox21 --seed $SEED | tee $BASE/tox21/LOG

echo "training toxcast"
python nnclassify.py --train data/toxcast.csv --hidden $HIDDEN --save_dir $BASE/toxcast --seed $SEED | tee $BASE/toxcast/LOG

echo "training sider"
python nnclassify.py --train data/sider.csv --hidden $HIDDEN --save_dir $BASE/sider --seed $SEED --batch 10 | tee $BASE/sider/LOG

echo "training clintox"
python nnclassify.py --train data/clintox.csv --hidden $HIDDEN --save_dir $BASE/clintox --seed $SEED --batch 10 | tee $BASE/clintox/LOG

#Large datasets
echo "training MUV"
python nnclassify.py --train data/muv.csv --hidden $HIDDEN --save_dir $BASE/muv --seed $SEED --metric prc | tee $BASE/muv/LOG

echo "training HIV"
python nnclassify.py --train data/HIV.csv --hidden $HIDDEN --save_dir $BASE/HIV --seed $SEED --split scaffold | tee $BASE/HIV/LOG

echo "training pcba"
python nnclassify.py --train data/pcba.csv --hidden $HIDDEN --save_dir $BASE/pcba --seed $SEED --metric prc --anneal 50000 | tee $BASE/pcba/LOG
