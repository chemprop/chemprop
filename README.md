# Property Prediction
This repository contains graph convolutional networks (or message passing network) for molecule property prediction. 

## Installation
Requirements: Please use conda package manager for library installation.
 * Python 2.7: We currently does not support python 3.
 * RDKit: `conda install -c rdkit rdkit`
 * scikit-learn: `conda install scikit-learn`
 * pytorch (GPU): Please follow the installation guide on pytorch.org
 * numpy `conda install numpy`
 * cuda-8.0 + cuDNN

## Deepchem test
We provide scripts and README for running deepchem benchmark datasets in `deepchem-test/`

## Training (Classification task)
```
mkdir model
python train_classify.py --train $TRAIN_FILE --valid $VALID_FILE --test $TEST_FILE --save_dir model --epoch 30
```
This script will train the network 30 epochs, and save the best model in `model/model.best`.
The input file `TRAIN_FILE` has to be a CSV file with a header row.

The above code assumes the task is binary classification.

## Training (Regression task)
```
mkdir model
python train_regress.py --train $TRAIN_FILE --valid $VALID_FILE --test $TEST_FILE --save_dir model --epoch 30
```

## Testing (Classification)
```
python test_classify.py --test $TEST_FILE --model model/model.best --num_task $NTASK
```
where $NTASK$ is the number of tasks the model needs to predict (in the case of multi-task training).
This will print, in each line, the probability of an input molecule belonging to positive class for each task.

## Testing (Regression)
```
python test_regress.py --test $TEST_FILE --model model/model.best --num_task $NTASK
```
This will print, in each line, the predicted property value of each input molecule.
