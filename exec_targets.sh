export CUDA_VISIBLE_DEVICES=3
export CUDA_VISIBLE_DEVICES=3
export CUDA_VISIBLE_DEVICES=3
reconda
reconda
reconda
conda activate rxrx
conda activate rxrx
conda activate rxrx

export DATA_PREFIX=/data/rsg/chemistry/quach/covid/recursion/recursion_all_ellinger
export DATA_PREFIX=/data/rsg/chemistry/quach/covid/recursion/recursion_all_ellinger
export DATA_PREFIX=/data/rsg/chemistry/quach/covid/recursion/recursion_all_ellinger

export SAVE_PREFIX=/data/scratch/quach/serialize/covid_project/recursion_all_ellinger
export SAVE_PREFIX=/data/scratch/quach/serialize/covid_project/recursion_all_ellinger
export SAVE_PREFIX=/data/scratch/quach/serialize/covid_project/recursion_all_ellinger

# export LAMBDA=0.01
# export LAMBDA=0.01
# export LAMBDA=0.01

export LAMBDA=1
export LAMBDA=0.1
export LAMBDA=0.01

notify python scripts/save_features.py  --data_path ${DATA_PREFIX}_activity.csv --features_generator rdkit_2d_normalized --save_path $DATA_PREFIX.npz 

notify python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --num_folds 5 --batch_size 50 --split_type scaffold_balanced --save_smiles_splits --save_dir $SAVE_PREFIX/scaffold/chemprop

notify python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --split_type scaffold_balanced --no_features_scaling --num_folds 5 --features_path ${DATA_PREFIX}.npz

python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}.npz
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}.npz
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}.npz

# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_max.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}.npz
# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_mean.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}.npz
# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_random.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}.npz

# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_images_max.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}.npz
# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_images_mean.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}.npz
python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_images_random.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}.npz

# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA
# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA
# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA


# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_max.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA
# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_mean.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA
# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_random.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA

# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_images_max.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA
# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_images_mean.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA
# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_images_random.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA


# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_max.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}_mean.npz
# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_mean.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}_mean.npz
# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_random.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}_mean.npz

# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_images_max.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}_mean.npz
# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_images_mean.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}_mean.npz
# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_images_random.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}_mean.npz

# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_max.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}_random.npz
# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_mean.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}_random.npz
# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_random.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}_random.npz

# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_images_max.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}_random.npz
# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_images_mean.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}_random.npz
# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_images_random.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}_random.npz



# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_max.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}.npz
# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_mean.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}.npz
# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_random.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}.npz

# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_images_max.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}.npz
# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_images_mean.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}.npz
# python train.py  --data_path ${DATA_PREFIX}_activity.csv --dataset_type classification --target_features_path ${DATA_PREFIX}_images_random.npz --split_type scaffold_balanced --no_features_scaling --num_folds 5 --auxiliary_lambda $LAMBDA --features_path ${DATA_PREFIX}.npz

