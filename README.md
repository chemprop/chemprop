# Property Prediction
This repository contains graph convolutional networks (or message passing network) for molecule property prediction. 
The input file has to be a CSV file with header row.

## Training (Classification task)
```
mkdir model
python train_classify.py --train $TRAIN --valid $VALID --test $TEST --save_dir model --epoch 30
```
This script will train the network 20 epochs, and save the best model in `model/model.best`

## Training (Regression task)
```
mkdir model
python train_regress.py --train $TRAIN --valid $VALID --test $TEST --save_dir model --epoch 30
```

## Testing (Classification)
```
python test_classify.py --test $TEST --model model/model.best
```
This will print out predicted property value for each input molecule.

## Testing (Regression)
```
python test_regress.py --test $TEST --model model/model.best
```
