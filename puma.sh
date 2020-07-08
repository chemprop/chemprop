python scripts/save_features.py  --data_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels.csv --features_generator rdkit_2d_normalized --save_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels.npz

notify python train.py  --data_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels.csv --dataset_type classification --num_folds 5 --batch_size 50 --split_type scaffold_balanced --save_smiles_splits

# 5-fold cross validation
# Seed 0 ==> test auc = 0.518638
# Seed 1 ==> test auc = 0.569686
# Seed 2 ==> test auc = 0.587024
# Seed 3 ==> test auc = 0.589690
# Seed 4 ==> test auc = 0.579378
# Overall test auc = 0.568883 +/- 0.026067


notify python train.py  --data_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels.csv --dataset_type classification --num_folds 5 --batch_size 50 --split_type scaffold_balanced --save_smiles_splits --features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels.npz

# 5-fold cross validation
# Seed 0 ==> test auc = 0.645477
# Seed 1 ==> test auc = 0.530097
# Seed 2 ==> test auc = 0.608559
# Seed 3 ==> test auc = 0.651375
# Seed 4 ==> test auc = 0.609788
# Overall test auc = 0.609059 +/- 0.043250

# Chemprop hyperopt with fingerprint and scaffold
python hyperparameter_optimization.py --data_path $DATA_PREFIX.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path $DATA_PREFIX.npz --split_type scaffold_balanced --no_features_scaling --config_save_path $DATA_PREFIX.fingerping.scaffold.json

# Run best hyperparam with fingerprint and scaffold
python train.py --data_path $DATA_PREFIX.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path $DATA_PREFIX.npz --split_type scaffold_balanced --no_features_scaling --config_path $DATA_PREFIX.fingerping.scaffold.json


# Using adam's split

python scripts/save_features.py  --data_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train.csv --features_generator rdkit_2d_normalized --save_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train.npz

python scripts/save_features.py  --data_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev.csv --features_generator rdkit_2d_normalized --save_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev.npz


# Chemprop only

python train.py  --data_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train.csv --separate_test_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev.csv --dataset_type classification --batch_size 50 --split_type scaffold_balanced

# Validation auc = 0.608760
# Test auc = 0.588071

# Chemprop + fingerprints

notify python train.py  \
  --data_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train.csv \
  --separate_test_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev.csv \
  --dataset_type classification --batch_size 50 --split_type scaffold_balanced \
  --features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train.npz \
  --separate_test_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev.npz

# Validation auc = 0.641247
# Test auc = 0.586269

# fingerprints only

notify python train.py  --data_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train.csv --separate_test_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev.csv --dataset_type classification --batch_size 50 --split_type scaffold_balanced --features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train.npz --separate_test_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev.npz --features_only

# Validation auc = 0.650945
# Test auc = 0.563533

## Using images as features

### Chemprop + Dummy 

notify python train.py  \
  --data_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train.csv \
  --separate_test_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev.csv \
  --dataset_type classification --batch_size 50 --split_type scaffold_balanced \
  --features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train_dummy_mean.npz \
  --separate_test_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev_dummy_mean.npz \
  --distill mse_distill

# Validation auc = 0.656967
# Test auc = 0.668180

### Dummy only

notify python train.py  \
  --data_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train.csv \
  --separate_test_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev.csv \
  --dataset_type classification --batch_size 50 --split_type scaffold_balanced \
  --features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train_dummy_mean.npz \
  --separate_test_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev_dummy_mean.npz \
  --features_only --no_features_scaling

# Validation auc = 0.620700
# Test auc = 0.633824

### Chemprop + Dummy + fingerprints

notify python train.py  \
  --data_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train.csv \
  --separate_test_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev.csv \
  --dataset_type classification --batch_size 50 --split_type scaffold_balanced \
  --features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train_dummy_mean.npz \
  --features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train.npz \
  --separate_test_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev_dummy_mean.npz \
  --separate_test_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev.npz

### Chmeprop + images
notify python train.py  \
  --data_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train.csv \
  --separate_test_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev.csv \
  --dataset_type classification --batch_size 50 --split_type scaffold_balanced \
  --features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train_train_on_activ_jun30_2020_mean.npz \
  --separate_test_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev_train_on_activ_jun30_2020_mean.npz \
  --no_features_scaling

# Validation auc = 0.684336
# Test auc = 0.677624

### Images only

notify python train.py  \
  --data_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train.csv \
  --separate_test_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev.csv \
  --dataset_type classification --batch_size 50 --split_type scaffold_balanced \
  --features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train_train_on_activ_jun30_2020_mean.npz \
  --separate_test_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev_train_on_activ_jun30_2020_mean.npz \
  --features_only --no_features_scaling

# Test auc = 0.690318

## Using images as aux targets

### Dummy as aux targets

notify python train.py  \
  --data_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train.csv \
  --separate_test_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev.csv \
  --dataset_type classification --batch_size 50 --split_type scaffold_balanced \
  --target_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train_dummy_mean.npz \
  --separate_test_target_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev_dummy_mean.npz \
  --distill pred_as_hidden_mse_distill \
  --distill_lambda 0.1 \
  --hidden_size 512


# Validation auc = 0.618860
# Test auc = 0.588618

### Images as aux targets

notify python train.py  \
  --data_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train.csv \
  --separate_test_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev.csv \
  --dataset_type classification --batch_size 50 --split_type scaffold_balanced \
  --target_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train_train_on_activ_jun30_2020_mean.npz \
  --separate_test_target_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev_train_on_activ_jun30_2020_mean.npz

# Validation auc = 0.619492
# test auc = 0.592979

### Images as aux targets

notify python train.py  \
  --data_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train.csv \
  --separate_test_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev.csv \
  --dataset_type classification --batch_size 50 --split_type scaffold_balanced \
  --target_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train_train_on_activ_jun30_2020_mean.npz \
  --separate_test_target_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev_train_on_activ_jun30_2020_mean.npz \
  --distill_lambda 1 \
  --distill mse_distill \
  --hidden_size  512



notify python train.py  \
  --data_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train.csv \
  --separate_test_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev.csv \
  --dataset_type classification --batch_size 50 --split_type scaffold_balanced \
  --target_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train_train_on_activ_jun30_2020_mean.npz \
  --separate_test_target_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev_train_on_activ_jun30_2020_mean.npz \
  --auxiliary_lambda 0.1


notify python train.py  \
  --data_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train.csv \
  --separate_test_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev.csv \
  --dataset_type classification --batch_size 50 --split_type scaffold_balanced \
  --target_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train_train_on_activ_jun30_2020_mean.npz \
  --separate_test_target_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev_train_on_activ_jun30_2020_mean.npz \
  --auxiliary_lambda 0.01

# Dummy as aux targets


notify python train.py  \
  --data_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train.csv \
  --separate_test_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev.csv \
  --dataset_type classification --batch_size 50 --split_type scaffold_balanced \
  --target_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train_dummy_mean.npz \
  --separate_test_target_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev_dummy_mean.npz \
  --auxiliary_lambda 1

notify python train.py  \
  --data_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train.csv \
  --separate_test_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev.csv \
  --dataset_type classification --batch_size 50 --split_type scaffold_balanced \
  --target_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train_dummy_mean.npz \
  --separate_test_target_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev_dummy_mean.npz \
  --auxiliary_lambda 0.1


notify python train.py  \
  --data_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train.csv \
  --separate_test_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev.csv \
  --dataset_type classification --batch_size 50 --split_type scaffold_balanced \
  --target_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_train_dummy_mean.npz \
  --separate_test_target_features_path /data/rsg/chemistry/quach/cellpainter/puma/smiles_with_labels_dev_dummy_mean.npz \
  --auxiliary_lambda 0.01

