python scripts/save_features.py  --data_path /data/rsg/chemistry/quach/covid/french_all/french_rdkit_smiles.csv --features_generator rdkit_2d_normalized --save_path /data/rsg/chemistry/quach/covid/french_all/french_rdkit_smiles.npz
python scripts/save_features.py  --data_path /data/rsg/chemistry/quach/covid/french_all/french_recursion_rdkit_smiles.csv --features_generator rdkit_2d_normalized --save_path /data/rsg/chemistry/quach/covid/french_all/french_recursion_rdkit_smiles.npz
python scripts/save_features.py  --data_path /data/rsg/chemistry/quach/covid/french_all/french_pubchem_smiles.csv --features_generator rdkit_2d_normalized --save_path /data/rsg/chemistry/quach/covid/french_all/french_pubchem_smiles.npz
python scripts/save_features.py  --data_path /data/rsg/chemistry/quach/covid/french_all/french_recursion_pubchem_smiles.csv --features_generator rdkit_2d_normalized --save_path /data/rsg/chemistry/quach/covid/french_all/french_recursion_pubchem_smiles.npz

python scripts/save_features.py  --data_path /data/rsg/chemistry/quach/covid/stanford/stanford_rdkit_any.csv --features_generator rdkit_2d_normalized --save_path /data/rsg/chemistry/quach/covid/stanford/stanford_rdkit_any.npz
python scripts/save_features.py  --data_path /data/rsg/chemistry/quach/covid/stanford/stanford_rdkit_majority.csv --features_generator rdkit_2d_normalized --save_path /data/rsg/chemistry/quach/covid/stanford/stanford_rdkit_majority.npz

python train.py  --data_path /data/rsg/chemistry/quach/covid/french_all/french_rdkit_smiles.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path /data/rsg/chemistry/quach/covid/french_all/french_rdkit_smiles.npz
CUDA_VISIBLE_DEVICES=2 python train.py  --data_path /data/rsg/chemistry/quach/covid/french_all/french_recursion_rdkit_smiles.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path /data/rsg/chemistry/quach/covid/french_all/french_recursion_rdkit_smiles.npz
python train.py  --data_path /data/rsg/chemistry/quach/covid/french_all/french_pubchem_smiles.csv --dataset_type classification --num_folds 5 --batch_size 50
CUDA_VISIBLE_DEVICES=3 python train.py  --data_path /data/rsg/chemistry/quach/covid/french_all/french_recursion_pubchem_smiles.csv --dataset_type classification --num_folds 5 --batch_size 50

python train.py  --data_path /data/rsg/chemistry/quach/covid/french_all/french_pubchem_smiles.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path /data/rsg/chemistry/quach/covid/french_all/french_pubchem_smiles.npz
CUDA_VISIBLE_DEVICES=2 python train.py  --data_path /data/rsg/chemistry/quach/covid/french_all/french_recursion_pubchem_smiles.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path /data/rsg/chemistry/quach/covid/french_all/french_recursion_pubchem_smiles.npz

export DATA_PREFIX=/data/rsg/chemistry/quach/covid/french_all/french_aicures_smiles

# Chemprop only -random split
python train.py  --data_path $DATA_PREFIX.csv --dataset_type classification --num_folds 5 --batch_size 50 
# Chemprop + fingerpint -random split
python train.py  --data_path $DATA_PREFIX.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path $DATA_PREFIX.npz

export DATA_PREFIX=/data/rsg/chemistry/quach/covid/recursion/french_recursion_name_rdkit_smiles
export DATA_PREFIX=/data/rsg/chemistry/quach/covid/recursion/stanford_recursion_name_rdikit_smiles_any
export DATA_PREFIX=/data/rsg/chemistry/quach/covid/recursion/stanford_recursion_name_rdikit_smiles_majority
export DATA_PREFIX=/data/rsg/chemistry/quach/covid/recursion/recursion_ellinger

export SAVE_PREFIX=/data/scratch/quach/serialize/covid_project/french_recursion
export SAVE_PREFIX=/data/scratch/quach/serialize/covid_project/stanford_recursion_any
export SAVE_PREFIX=/data/scratch/quach/serialize/covid_project/stanford_recursion_majority
export SAVE_PREFIX=/data/scratch/quach/serialize/covid_project/recursion_ellinger

export DATA_PREFIX=/data/rsg/chemistry/quach/covid/ellinger/ellinger

python train.py --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}.npz --split_type scaffold_balanced

python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_mean.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5

# Make features
notify python scripts/save_features.py  --data_path ${DATA_PREFIX}_activity.csv --features_generator rdkit_2d_normalized --save_path $DATA_PREFIX.npz 
python scripts/save_features.py  --data_path ${DATA_PREFIX}_activity.csv --features_generator rdkit_2d_normalized --save_path $DATA_PREFIX.npz
python scripts/save_features.py  --data_path ${DATA_PREFIX}_activity.csv --features_generator rdkit_2d_normalized --save_path $DATA_PREFIX.npz


# Chemprop + fingerpint --scaffold
notify python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path $DATA_PREFIX.npz --split_type scaffold_balanced --no_features_scaling

# Chemprop only --saffold
notify python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --split_type scaffold_balanced --save_smiles_splits --save_dir $SAVE_PREFIX/scaffold/chemprop
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --split_type scaffold_balanced --save_smiles_splits --save_dir $SAVE_PREFIX/scaffold/chemprop
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --split_type scaffold_balanced --save_smiles_splits --save_dir $SAVE_PREFIX/scaffold/chemprop

# Chemprop + fingerpint --scaffold
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path $DATA_PREFIX.npz --split_type scaffold_balanced --no_features_scaling --save_smiles_splits --save_dir $SAVE_PREFIX/scaffold/chemprop_fingerpints
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path $DATA_PREFIX.npz --split_type scaffold_balanced --no_features_scaling --save_smiles_splits --save_dir $SAVE_PREFIX/scaffold/chemprop_fingerpints
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path $DATA_PREFIX.npz --split_type scaffold_balanced --no_features_scaling --save_smiles_splits --save_dir $SAVE_PREFIX/scaffold/chemprop_fingerpints

# Chemprop + max recursion --scaffold
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_max.npz --split_type scaffold_balanced --no_features_scaling --save_dir $SAVE_PREFIX/scaffold/chemprop_max_recursion
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_max.npz --split_type scaffold_balanced --no_features_scaling --save_dir $SAVE_PREFIX/scaffold/chemprop_max_recursion
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_max.npz --split_type scaffold_balanced --no_features_scaling --save_dir $SAVE_PREFIX/scaffold/chemprop_max_recursion

# Chemprop + mean recursion --scaffold
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_mean.npz --split_type scaffold_balanced --no_features_scaling --save_dir $SAVE_PREFIX/scaffold/chemprop_mean_recursion
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_mean.npz --split_type scaffold_balanced --no_features_scaling --save_dir $SAVE_PREFIX/scaffold/chemprop_mean_recursion
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_mean.npz --split_type scaffold_balanced --no_features_scaling --save_dir $SAVE_PREFIX/scaffold/chemprop_mean_recursion

# Chemprop + images mean  --scaffold
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_images_mean.npz --split_type scaffold_balanced --no_features_scaling --save_dir $SAVE_PREFIX/scaffold/chemprop_images_mean_recursion
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_images_mean.npz --split_type scaffold_balanced --no_features_scaling --save_dir $SAVE_PREFIX/scaffold/chemprop_images_mean_recursion
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_images_mean.npz --split_type scaffold_balanced --no_features_scaling --save_dir $SAVE_PREFIX/scaffold/chemprop_images_mean_recursion

# Chemprop + images max  --scaffold
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_images_max.npz --split_type scaffold_balanced --no_features_scaling --save_dir $SAVE_PREFIX/scaffold/chemprop_images_max_recursion
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_images_max.npz --split_type scaffold_balanced --no_features_scaling --save_dir $SAVE_PREFIX/scaffold/chemprop_images_max_recursion
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_images_max.npz --split_type scaffold_balanced --no_features_scaling --save_dir $SAVE_PREFIX/scaffold/chemprop_images_max_recursion


# Chemprop + images random  --scaffold
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_images_random.npz --split_type scaffold_balanced --no_features_scaling --save_dir $SAVE_PREFIX/scaffold/chemprop_images_random_recursion
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_images_random.npz --split_type scaffold_balanced --no_features_scaling --save_dir $SAVE_PREFIX/scaffold/chemprop_images_random_recursion
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_images_random.npz --split_type scaffold_balanced --no_features_scaling --save_dir $SAVE_PREFIX/scaffold/chemprop_images_random_recursion



# Chemprop + random recursion --scaffold
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_random.npz --split_type scaffold_balanced --no_features_scaling --save_dir $SAVE_PREFIX/scaffold/chemprop_random_recursion
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_random.npz --split_type scaffold_balanced --no_features_scaling --save_dir $SAVE_PREFIX/scaffold/chemprop_random_recursion
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_random.npz --split_type scaffold_balanced --no_features_scaling --save_dir $SAVE_PREFIX/scaffold/chemprop_random_recursion

# Chemprop + fingerpints + random recursion --scaffold
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_random.npz --features_path ${DATA_PREFIX}.npz --split_type scaffold_balanced --no_features_scaling
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_random.npz --features_path ${DATA_PREFIX}.npz --split_type scaffold_balanced --no_features_scaling
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_random.npz --features_path ${DATA_PREFIX}.npz --split_type scaffold_balanced --no_features_scaling




# Chemprop hyperopt with fingerprint and scaffold
python hyperparameter_optimization.py --data_path $DATA_PREFIX.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path $DATA_PREFIX.npz --split_type scaffold_balanced --no_features_scaling --config_save_path $DATA_PREFIX.fingerping.scaffold.json

# Run best hyperparam with fingerprint and scaffold
python train.py --data_path $DATA_PREFIX.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path $DATA_PREFIX.npz --split_type scaffold_balanced --no_features_scaling --config_path $DATA_PREFIX.fingerping.scaffold.json


python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_images_max.npz --split_type scaffold_balanced --no_features_scaling --save_dir $SAVE_PREFIX/scaffold/chemprop_images_max_recursion
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_images_mean.npz --split_type scaffold_balanced --no_features_scaling --save_dir $SAVE_PREFIX/scaffold/chemprop_images_mean_recursion
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --features_path ${DATA_PREFIX}_images_random.npz --split_type scaffold_balanced --no_features_scaling --save_dir $SAVE_PREFIX/scaffold/chemprop_images_random_recursion
