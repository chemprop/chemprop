#!/bin/bash -l

CHEMPROP_ENV=$1
CHEMPROP_PATH=$2

if [ -z "${CHEMPROP_ENV}" ] || [ -z "${CHEMPROP_PATH}" ]; then
    echo "Usage: regenerate_models.sh <CHEMPROP_ENV> <CHEMPROP_PATH>"
    exit 1
fi

conda activate $CHEMPROP_ENV

data_dir="$CHEMPROP_PATH/tests/data/mol_atom_bond"
save_dir="save_dir"

rm -rf $save_dir
chemprop train -i $data_dir/regression.csv --mol-target-columns mol_y1 mol_y2 --atom-target-columns atom_y1 atom_y2 --bond-target-columns bond_y1 bond_y2 --accelerator cpu --epochs 3 --save-dir $save_dir --keep-h --reorder-atoms
cp -L $save_dir/model_0/best.pt $data_dir/example_models/regression.pt

rm -rf $save_dir
chemprop train -i $data_dir/regression.csv --mol-target-columns mol_y1 mol_y2 --atom-target-columns atom_y1 atom_y2 --bond-target-columns bond_y1 bond_y2 --accelerator cpu --epochs 3 --save-dir $save_dir --keep-h --reorder-atoms --descriptors-path $data_dir/descriptors.npz --atom-features-path $data_dir/atom_features_descriptors.npz --bond-features-path $data_dir/bond_features_descriptors.npz --atom-descriptors-path $data_dir/atom_features_descriptors.npz --bond-descriptors-path $data_dir/bond_features_descriptors.npz
cp -L $save_dir/model_0/best.pt $data_dir/example_models/regression_with_extras.pt

rm -rf $save_dir
chemprop train -i $data_dir/regression.csv --atom-target-columns atom_y1 atom_y2 --bond-target-columns bond_y1 bond_y2 --accelerator cpu --epochs 3 --save-dir $save_dir --keep-h --reorder-atoms
cp -L $save_dir/model_0/best.pt $data_dir/example_models/regression_no_mol.pt

rm -rf $save_dir
chemprop train -i $data_dir/regression.csv --mol-target-columns mol_y1 mol_y2 --bond-target-columns bond_y1 bond_y2 --accelerator cpu --epochs 3 --save-dir $save_dir --keep-h --reorder-atoms
cp -L $save_dir/model_0/best.pt $data_dir/example_models/regression_no_atom.pt

rm -rf $save_dir
chemprop train -i $data_dir/regression.csv --mol-target-columns mol_y1 mol_y2 --atom-target-columns atom_y1 atom_y2 --accelerator cpu --epochs 3 --save-dir $save_dir --keep-h --reorder-atoms
cp -L $save_dir/model_0/best.pt $data_dir/example_models/regression_no_bond.pt

rm -rf $save_dir
chemprop train -i $data_dir/regression.csv --mol-target-columns mol_y1 mol_y2 --accelerator cpu --epochs 3 --save-dir $save_dir --keep-h --reorder-atoms
cp -L $save_dir/model_0/best.pt $data_dir/example_models/regression_only_mol.pt

rm -rf $save_dir
chemprop train -i $data_dir/regression.csv --atom-target-columns atom_y1 atom_y2 --accelerator cpu --epochs 3 --save-dir $save_dir --keep-h --reorder-atoms
cp -L $save_dir/model_0/best.pt $data_dir/example_models/regression_only_atom.pt

rm -rf $save_dir
chemprop train -i $data_dir/regression.csv --bond-target-columns bond_y1 bond_y2 --accelerator cpu --epochs 3 --save-dir $save_dir --keep-h --reorder-atoms
cp -L $save_dir/model_0/best.pt $data_dir/example_models/regression_only_bond.pt

rm -rf $save_dir
chemprop train -i $data_dir/regression.csv --mol-target-columns mol_y1 mol_y2 --atom-target-columns atom_y1 atom_y2 --bond-target-columns bond_y1 bond_y2 --accelerator cpu --epochs 3 --save-dir $save_dir --keep-h --reorder-atoms -t regression-mve
cp -L $save_dir/model_0/best.pt $data_dir/example_models/regression_mve.pt

rm -rf $save_dir
chemprop train -i $data_dir/classification.csv --mol-target-columns mol_y1 mol_y2 --atom-target-columns atom_y1 atom_y2 --bond-target-columns bond_y1 bond_y2 --accelerator cpu --epochs 3 --save-dir $save_dir --keep-h --reorder-atoms -t classification
cp -L $save_dir/model_0/best.pt $data_dir/example_models/classification.pt

rm -rf $save_dir
chemprop train -i $data_dir/multiclass.csv --mol-target-columns mol_y1 mol_y2 --atom-target-columns atom_y1 atom_y2 --bond-target-columns bond_y1 bond_y2 --accelerator cpu --epochs 3 --save-dir $save_dir --keep-h --reorder-atoms -t multiclass
cp -L $save_dir/model_0/best.pt $data_dir/example_models/multiclass.pt

rm -rf $save_dir
chemprop train -i $data_dir/constrained_regression.csv --mol-target-columns mol_y --atom-target-columns atom_y1 atom_y2 --bond-target-columns bond_y1 bond_y2 --accelerator cpu --epochs 3 --save-dir $save_dir --keep-h --reorder-atoms --constraints-to-targets atom_target_0 atom_target_1 bond_target_1 --constraints-path $data_dir/constrained_regression_constraints.csv
cp -L $save_dir/model_0/best.pt $data_dir/example_models/regression_constrained.pt

rm -rf $save_dir
chemprop train -i $data_dir/atomic_bond_regression.csv --mol-target-columns homo lumo --atom-target-columns hirshfeld_charges hirshfeld_charges_plus1 hirshfeld_charges_minus1 hirshfeld_spin_density_plus1 hirshfeld_spin_density_minus1 hirshfeld_charges_fukui_neu hirshfeld_charges_fukui_elec NMR --bond-target-columns bond_length_matrix bond_index_matrix --constraints-to-targets atom_target_0 atom_target_1 atom_target_2 atom_target_3 atom_target_4 atom_target_5 atom_target_6 --constraints-path $data_dir/atomic_bond_constraints.csv --add-h --accelerator cpu --epochs 3 --save-dir $save_dir 
cp -L $save_dir/model_0/best.pt $data_dir/example_models/QM_descriptors.pt

chemprop predict -i $data_dir/atomic_bond_regression.csv --model-path $data_dir/example_models/QM_descriptors.pt --constraints-to-targets atom_target_0 atom_target_1 atom_target_2 atom_target_3 atom_target_4 atom_target_5 atom_target_6 --constraints-path $data_dir/atomic_bond_constraints.csv --add-h --accelerator cpu

rm -rf $save_dir
chemprop train -i $data_dir/atomic_regression_atom_mapped.csv --atom-target-columns charges --keep-h -reorder-atoms --accelerator cpu --epochs 3 --save-dir $save_dir 
cp -L $save_dir/model_0/best.pt $data_dir/example_models/atomic_regression_atom_mapped.pt

chemprop predict -i $data_dir/atomic_regression_atom_mapped.csv --model-path $data_dir/example_models/atomic_regression_atom_mapped.pt --keep-h --reorder-atoms --accelerator cpu

rm -rf $save_dir