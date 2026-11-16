import pytest
import pandas as pd

from chemprop.cli.utils.parsing import build_data_from_files


def test_invalid_smiles_breaks(data_dir):
    with pytest.raises(RuntimeError, match='SMILES not_a_smiles is invalid'):
        build_data_from_files(
            data_dir / 'invalid_smiles.csv',
            no_header_row=False,
            smiles_cols=['smiles'],
            rxn_cols=None,
            target_cols=None,
            ignore_cols=None,
            splits_col=None,
            weight_col=None,
            bounded=False,

            p_descriptors=None,
            p_atom_feats=None,
            p_bond_feats=None,
            p_atom_descs=None,
            
            descriptor_cols=None,
            n_workers=0,

            molecule_featurizers=None,
            keep_h=False,
            add_h=False,
            ignore_stereo=False,
            reorder_atoms=False,
            use_cuikmolmaker_featurization=False
        )


def test_skip_invalid_smiles(data_dir, caplog):
    data = build_data_from_files(
        data_dir / 'invalid_smiles.csv',
        no_header_row=False,
        smiles_cols=['smiles'],
        rxn_cols=None,
        target_cols=['target'],
        ignore_cols=None,
        splits_col=None,
        weight_col=None,
        bounded=False,

        p_descriptors=None,
        p_atom_feats=None,
        p_bond_feats=None,
        p_atom_descs=None,
        
        descriptor_cols=None,
        n_workers=0,

        skip_invalid_smiles=True,

        molecule_featurizers=None,
        keep_h=False,
        add_h=False,
        ignore_stereo=False,
        reorder_atoms=False,
        use_cuikmolmaker_featurization=False
        )
    assert len(data[0]) == 2
    assert all(dp.y is not None for dp in data[0])
    assert [dp.y[0] for dp in data[0]] == [1,2]
    assert "Skipped 1 of 3 rows" in caplog.text


def test_skip_invalid_smiles_parallel(data_dir, caplog):
    data = build_data_from_files(
        data_dir / 'invalid_smiles.csv',
        no_header_row=False,
        smiles_cols=['smiles'],
        rxn_cols=None,
        target_cols=['target'],
        ignore_cols=None,
        splits_col=None,
        weight_col=None,
        bounded=False,

        p_descriptors=None,
        p_atom_feats=None,
        p_bond_feats=None,
        p_atom_descs=None,
        
        descriptor_cols=None,
        n_workers=2,

        skip_invalid_smiles=True,

        molecule_featurizers=None,
        keep_h=False,
        add_h=False,
        ignore_stereo=False,
        reorder_atoms=False,
        use_cuikmolmaker_featurization=False
        )
    assert len(data[0]) == 2
    assert all(dp.y is not None for dp in data[0])
    assert [dp.y[0] for dp in data[0]] == [1,2]
    assert "Skipped 1 of 3 rows" in caplog.text


def test_skip_invalid_smiles_all_valid(data_dir, caplog):
    df = pd.read_csv(data_dir / 'smis.csv')
    data = build_data_from_files(
        data_dir / 'smis.csv',
        no_header_row=False,
        smiles_cols=['smiles'],
        rxn_cols=None,
        target_cols=None,
        ignore_cols=None,
        splits_col=None,
        weight_col=None,
        bounded=False,

        p_descriptors=None,
        p_atom_feats=None,
        p_bond_feats=None,
        p_atom_descs=None,
        
        descriptor_cols=None,
        n_workers=0,

        skip_invalid_smiles=True,

        molecule_featurizers=None,
        keep_h=False,
        add_h=False,
        ignore_stereo=False,
        reorder_atoms=False,
        use_cuikmolmaker_featurization=False
        )
    assert len(data[0]) == len(df)
    assert "rows containing invalid SMILES" not in caplog.text


def test_skip_invalid_smiles_multi_mol(data_dir, caplog):
    data = build_data_from_files(
        data_dir / 'invalid_smiles_multi_mol.csv',
        no_header_row=False,
        smiles_cols=['smiles_A', 'smiles_B'],
        rxn_cols=None,
        target_cols=['target'],
        ignore_cols=None,
        splits_col=None,
        weight_col=None,
        bounded=False,

        p_descriptors=None,
        p_atom_feats=None,
        p_bond_feats=None,
        p_atom_descs=None,
        
        descriptor_cols=None,
        n_workers=0,

        skip_invalid_smiles=True,

        molecule_featurizers=None,
        keep_h=False,
        add_h=False,
        ignore_stereo=False,
        reorder_atoms=False,
        use_cuikmolmaker_featurization=False
        )
    assert len(data) == 2
    assert len(data[0]) == 2
    assert len(data[1]) == 2
    assert all(dp.y is not None for dp in data[0])
    assert [dp.y[0] for dp in data[0]] == [1,2]

    # Make sure we get exactly one skip warning even for multi-molecule setup
    messages = [rec.getMessage() for rec in caplog.records]
    matches = [msg for msg in messages if "rows containing invalid SMILES" in msg]
    assert len(matches) == 1
    assert "Skipped 1 of 3 rows" in matches[0]
