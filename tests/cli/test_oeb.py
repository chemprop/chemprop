"""Tests for OpenEye OEB file format support.

These tests convert existing CSV test data to OEB format on-the-fly and verify
that OEB parsing produces identical results to CSV parsing.  Tests requiring
the OpenEye toolkit are skipped when no valid license is available.
"""

from pathlib import Path

import numpy as np
import pytest

from chemprop.cli.utils.parsing import (
    _is_oeb_file,
    build_data_from_files,
    get_column_names,
    get_oeb_column_names,
    parse_csv,
    parse_oeb,
)

pytestmark = pytest.mark.CLI


def _has_openeye_license():
    """Check if openeye-toolkits is installed AND has a valid license."""
    import os

    try:
        from openeye import oechem  # noqa: F401
    except ImportError:
        return False

    return bool(
        os.environ.get("OE_LICENSE") or os.environ.get("OE_DIR") or os.path.exists("oe_license.txt")
    )


# ---------------------------------------------------------------------------
# csv_to_oeb helper
# ---------------------------------------------------------------------------


def csv_to_oeb(csv_path: Path, oeb_path: Path, smiles_col: str | None = None):
    """Convert a CSV file to an OEB file.

    Parameters
    ----------
    csv_path : Path
        Path to source CSV file.
    oeb_path : Path
        Path to destination OEB file.
    smiles_col : str | None
        Name of the SMILES column.  If ``None``, uses the first column.
    """
    from openeye import oechem
    import pandas as pd

    df = pd.read_csv(csv_path, index_col=False)

    if smiles_col is None:
        smiles_col = df.columns[0]

    # All columns except the SMILES column become SD data tags
    data_cols = [c for c in df.columns if c != smiles_col]

    ofs = oechem.oemolostream()
    ofs.SetFormat(oechem.OEFormat_OEB)
    ofs.open(str(oeb_path))

    for _, row in df.iterrows():
        mol = oechem.OEGraphMol()
        if not oechem.OESmilesToMol(mol, str(row[smiles_col])):
            # Skip molecules that can't be parsed (shouldn't happen with test data)
            continue

        for col in data_cols:
            val = row[col]
            if pd.notna(val):
                oechem.OESetSDData(mol, col, str(val))

        oechem.OEWriteMolecule(ofs, mol)

    ofs.close()


# ---------------------------------------------------------------------------
# Unit tests for helper functions (no license required)
# ---------------------------------------------------------------------------


class TestIsOebFile:
    def test_csv_returns_false(self):
        assert not _is_oeb_file("data.csv")

    def test_oeb_returns_true(self):
        assert _is_oeb_file("data.oeb")

    def test_oez_returns_true(self):
        assert _is_oeb_file("data.oez")

    def test_uppercase_oeb(self):
        assert _is_oeb_file("data.OEB")

    def test_nested_path(self):
        assert _is_oeb_file(Path("/foo/bar/data.oeb"))


# ---------------------------------------------------------------------------
# parse_oeb vs parse_csv parity tests (requires license)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_openeye_license(), reason="openeye-toolkits license required")
class TestParseOebVsParseCsv:
    """Verify that parse_oeb returns the same data as parse_csv for the same
    underlying information."""

    @pytest.fixture
    def oeb_mol_regression(self, data_dir, tmp_path):
        csv_path = data_dir / "regression/mol/mol.csv"
        oeb_path = tmp_path / "mol_regression.oeb"
        csv_to_oeb(csv_path, oeb_path, smiles_col="smiles")
        return oeb_path

    @pytest.fixture
    def oeb_classification(self, data_dir, tmp_path):
        csv_path = data_dir / "classification/mol.csv"
        oeb_path = tmp_path / "classification.oeb"
        csv_to_oeb(csv_path, oeb_path, smiles_col="smiles")
        return oeb_path

    def test_simple_regression(self, data_dir, oeb_mol_regression):
        """Single SMILES column + single target."""
        csv_path = data_dir / "regression/mol/mol.csv"

        csv_result = parse_csv(
            csv_path,
            smiles_cols=["smiles"],
            rxn_cols=None,
            target_cols=["lipo"],
            ignore_cols=None,
            splits_col=None,
            weight_col=None,
            descriptor_cols=None,
            bounded=False,
            no_header_row=False,
        )

        oeb_result = parse_oeb(
            oeb_mol_regression,
            smiles_cols=None,
            rxn_cols=None,
            target_cols=["lipo"],
            ignore_cols=None,
            splits_col=None,
            weight_col=None,
            descriptor_cols=None,
            bounded=False,
            no_header_row=False,
        )

        # Both should have SMILES for every row
        assert csv_result[0] is not None
        assert oeb_result[0] is not None
        assert len(csv_result[0][0]) == len(oeb_result[0][0])
        assert all(smi for smi in csv_result[0][0] if smi)
        assert all(smi for smi in oeb_result[0][0] if smi)

        # Targets must match exactly
        np.testing.assert_array_almost_equal(csv_result[2], oeb_result[2], decimal=4)

        # No reaction, weight, or descriptors
        assert oeb_result[1] is None
        assert oeb_result[3] is None
        assert oeb_result[6] is None

    def test_auto_detect_targets(self, data_dir, oeb_mol_regression):
        """When target_cols is None, both parsers should auto-detect the same
        target columns."""
        csv_path = data_dir / "regression/mol/mol.csv"

        csv_result = parse_csv(
            csv_path,
            smiles_cols=["smiles"],
            rxn_cols=None,
            target_cols=None,
            ignore_cols=None,
            splits_col=None,
            weight_col=None,
            descriptor_cols=None,
            bounded=False,
            no_header_row=False,
        )

        oeb_result = parse_oeb(
            oeb_mol_regression,
            smiles_cols=None,
            rxn_cols=None,
            target_cols=None,
            ignore_cols=None,
            splits_col=None,
            weight_col=None,
            descriptor_cols=None,
            bounded=False,
            no_header_row=False,
        )

        # Both should detect 'lipo' as the target
        assert csv_result[2].shape == oeb_result[2].shape
        np.testing.assert_array_almost_equal(csv_result[2], oeb_result[2], decimal=4)

    def test_multi_target_classification(self, data_dir, oeb_classification):
        """Multi-target classification with missing values."""
        csv_path = data_dir / "classification/mol.csv"

        targets = ["NR-AhR", "NR-ER", "SR-ARE", "SR-MMP"]

        csv_result = parse_csv(
            csv_path,
            smiles_cols=["smiles"],
            rxn_cols=None,
            target_cols=targets,
            ignore_cols=None,
            splits_col=None,
            weight_col=None,
            descriptor_cols=None,
            bounded=False,
            no_header_row=False,
        )

        oeb_result = parse_oeb(
            oeb_classification,
            smiles_cols=None,
            rxn_cols=None,
            target_cols=targets,
            ignore_cols=None,
            splits_col=None,
            weight_col=None,
            descriptor_cols=None,
            bounded=False,
            no_header_row=False,
        )

        assert csv_result[2].shape == oeb_result[2].shape
        np.testing.assert_array_almost_equal(
            csv_result[2][~np.isnan(csv_result[2])],
            oeb_result[2][~np.isnan(oeb_result[2])],
            decimal=4,
        )

    def test_sample_count(self, data_dir, oeb_mol_regression):
        """Ensure OEB and CSV produce the same number of samples."""
        csv_path = data_dir / "regression/mol/mol.csv"

        csv_result = parse_csv(
            csv_path,
            smiles_cols=["smiles"],
            rxn_cols=None,
            target_cols=None,
            ignore_cols=None,
            splits_col=None,
            weight_col=None,
            descriptor_cols=None,
            bounded=False,
            no_header_row=False,
        )

        oeb_result = parse_oeb(
            oeb_mol_regression,
            smiles_cols=None,
            rxn_cols=None,
            target_cols=None,
            ignore_cols=None,
            splits_col=None,
            weight_col=None,
            descriptor_cols=None,
            bounded=False,
            no_header_row=False,
        )

        assert len(csv_result[0][0]) == len(oeb_result[0][0])
        assert csv_result[2].shape[0] == oeb_result[2].shape[0]


# ---------------------------------------------------------------------------
# get_oeb_column_names tests (requires license)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_openeye_license(), reason="openeye-toolkits license required")
class TestGetOebColumnNames:
    def test_simple_regression(self, data_dir, tmp_path):
        csv_path = data_dir / "regression/mol/mol.csv"
        oeb_path = tmp_path / "test.oeb"
        csv_to_oeb(csv_path, oeb_path, smiles_col="smiles")

        input_cols, target_cols = get_oeb_column_names(
            oeb_path,
            smiles_cols=None,
            rxn_cols=None,
            target_cols=None,
            ignore_cols=None,
            splits_col=None,
            weight_col=None,
            no_header_row=False,
        )

        assert input_cols == ["<generated>"]
        assert "lipo" in target_cols

    def test_with_smiles_cols(self, data_dir, tmp_path):
        """When smiles_cols are specified (as SD tags), they should appear as
        input columns."""
        csv_path = data_dir / "regression/mol/mol.csv"
        oeb_path = tmp_path / "test.oeb"
        csv_to_oeb(csv_path, oeb_path, smiles_col="smiles")

        input_cols, target_cols = get_oeb_column_names(
            oeb_path,
            smiles_cols=["smiles"],
            rxn_cols=None,
            target_cols=None,
            ignore_cols=None,
            splits_col=None,
            weight_col=None,
            no_header_row=False,
        )

        assert "smiles" in input_cols
        assert "smiles" not in target_cols

    def test_excludes_splits_col(self, data_dir, tmp_path):
        csv_path = data_dir / "regression/mol/mol_with_splits.csv"
        oeb_path = tmp_path / "test.oeb"
        csv_to_oeb(csv_path, oeb_path, smiles_col="smiles")

        input_cols, target_cols = get_oeb_column_names(
            oeb_path,
            smiles_cols=None,
            rxn_cols=None,
            target_cols=None,
            ignore_cols=None,
            splits_col="split",
            weight_col=None,
            no_header_row=False,
        )

        assert "split" not in target_cols
        assert "lipo" in target_cols


# ---------------------------------------------------------------------------
# build_data_from_files integration tests (requires license)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_openeye_license(), reason="openeye-toolkits license required")
class TestBuildDataFromOeb:
    """End-to-end: build datapoints from OEB files."""

    def test_mol_regression(self, data_dir, tmp_path):
        csv_path = data_dir / "regression/mol/mol.csv"
        oeb_path = tmp_path / "mol_regression.oeb"
        csv_to_oeb(csv_path, oeb_path, smiles_col="smiles")

        csv_data = build_data_from_files(
            csv_path,
            no_header_row=False,
            smiles_cols=["smiles"],
            rxn_cols=None,
            target_cols=["lipo"],
            ignore_cols=None,
            splits_col=None,
            weight_col=None,
            bounded=False,
            p_descriptors=None,
            p_atom_feats={},
            p_bond_feats={},
            p_atom_descs={},
            descriptor_cols=None,
            n_workers=0,
            molecule_featurizers=None,
            keep_h=False,
            add_h=False,
            ignore_stereo=False,
            reorder_atoms=False,
            use_cuikmolmaker_featurization=False,
        )

        oeb_data = build_data_from_files(
            oeb_path,
            no_header_row=False,
            smiles_cols=None,
            rxn_cols=None,
            target_cols=["lipo"],
            ignore_cols=None,
            splits_col=None,
            weight_col=None,
            bounded=False,
            p_descriptors=None,
            p_atom_feats={},
            p_bond_feats={},
            p_atom_descs={},
            descriptor_cols=None,
            n_workers=0,
            molecule_featurizers=None,
            keep_h=False,
            add_h=False,
            ignore_stereo=False,
            reorder_atoms=False,
            use_cuikmolmaker_featurization=False,
        )

        assert len(csv_data[0]) == len(oeb_data[0])

        csv_targets = np.array([dp.y for dp in csv_data[0]])
        oeb_targets = np.array([dp.y for dp in oeb_data[0]])
        np.testing.assert_array_almost_equal(csv_targets, oeb_targets, decimal=4)

    def test_mol_regression_auto_targets(self, data_dir, tmp_path):
        """Build data with auto-detected target columns."""
        csv_path = data_dir / "regression/mol/mol.csv"
        oeb_path = tmp_path / "mol_regression.oeb"
        csv_to_oeb(csv_path, oeb_path, smiles_col="smiles")

        oeb_data = build_data_from_files(
            oeb_path,
            no_header_row=False,
            smiles_cols=None,
            rxn_cols=None,
            target_cols=None,
            ignore_cols=None,
            splits_col=None,
            weight_col=None,
            bounded=False,
            p_descriptors=None,
            p_atom_feats={},
            p_bond_feats={},
            p_atom_descs={},
            descriptor_cols=None,
            n_workers=0,
            molecule_featurizers=None,
            keep_h=False,
            add_h=False,
            ignore_stereo=False,
            reorder_atoms=False,
            use_cuikmolmaker_featurization=False,
        )

        assert len(oeb_data[0]) == 100

        for dp in oeb_data[0]:
            assert dp.mol is not None
            assert dp.y is not None

    def test_classification_multi_target(self, data_dir, tmp_path):
        csv_path = data_dir / "classification/mol.csv"
        oeb_path = tmp_path / "classification.oeb"
        csv_to_oeb(csv_path, oeb_path, smiles_col="smiles")

        targets = ["NR-AhR", "NR-ER", "SR-ARE", "SR-MMP"]

        oeb_data = build_data_from_files(
            oeb_path,
            no_header_row=False,
            smiles_cols=None,
            rxn_cols=None,
            target_cols=targets,
            ignore_cols=None,
            splits_col=None,
            weight_col=None,
            bounded=False,
            p_descriptors=None,
            p_atom_feats={},
            p_bond_feats={},
            p_atom_descs={},
            descriptor_cols=None,
            n_workers=0,
            molecule_featurizers=None,
            keep_h=False,
            add_h=False,
            ignore_stereo=False,
            reorder_atoms=False,
            use_cuikmolmaker_featurization=False,
        )

        assert len(oeb_data[0]) == 500

        for dp in oeb_data[0]:
            assert dp.y.shape[0] == 4

    def test_with_splits(self, data_dir, tmp_path):
        """Build data from OEB that has a split column stored as SD data."""
        csv_path = data_dir / "regression/mol/mol_with_splits.csv"
        oeb_path = tmp_path / "mol_with_splits.oeb"
        csv_to_oeb(csv_path, oeb_path, smiles_col="smiles")

        oeb_data = build_data_from_files(
            oeb_path,
            no_header_row=False,
            smiles_cols=None,
            rxn_cols=None,
            target_cols=["lipo"],
            ignore_cols=None,
            splits_col="split",
            weight_col=None,
            bounded=False,
            p_descriptors=None,
            p_atom_feats={},
            p_bond_feats={},
            p_atom_descs={},
            descriptor_cols=None,
            n_workers=0,
            molecule_featurizers=None,
            keep_h=False,
            add_h=False,
            ignore_stereo=False,
            reorder_atoms=False,
            use_cuikmolmaker_featurization=False,
        )

        assert len(oeb_data[0]) == 100

    def test_multimol_classification(self, data_dir, tmp_path):
        """Dual molecule classification (mol+mol)."""
        csv_path = data_dir / "classification/mol+mol.csv"
        oeb_path = tmp_path / "mol_plus_mol.oeb"

        from openeye import oechem
        import pandas as pd

        df = pd.read_csv(csv_path, index_col=False)

        ofs = oechem.oemolostream()
        ofs.SetFormat(oechem.OEFormat_OEB)
        ofs.open(str(oeb_path))

        for _, row in df.iterrows():
            mol = oechem.OEGraphMol()
            oechem.OESmilesToMol(mol, str(row["mol a smiles"]))
            oechem.OESetSDData(mol, "mol b Smiles", str(row["mol b Smiles"]))
            oechem.OESetSDData(mol, "synergy", str(row["synergy"]))
            oechem.OEWriteMolecule(ofs, mol)

        ofs.close()

        oeb_data = build_data_from_files(
            oeb_path,
            no_header_row=False,
            smiles_cols=None,
            rxn_cols=None,
            target_cols=["synergy"],
            ignore_cols=["mol b Smiles"],
            splits_col=None,
            weight_col=None,
            bounded=False,
            p_descriptors=None,
            p_atom_feats={},
            p_bond_feats={},
            p_atom_descs={},
            descriptor_cols=None,
            n_workers=0,
            molecule_featurizers=None,
            keep_h=False,
            add_h=False,
            ignore_stereo=False,
            reorder_atoms=False,
            use_cuikmolmaker_featurization=False,
        )

        # Number of datapoints should match the number of molecules written to OEB
        # (which may be less than CSV rows if OpenEye can't parse some SMILES)
        csv_rows = len(pd.read_csv(csv_path))
        assert len(oeb_data[0]) <= csv_rows
        for dp in oeb_data[0]:
            assert dp.y.shape[0] == 1


# ---------------------------------------------------------------------------
# Reaction SMILES tests (requires license)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_openeye_license(), reason="openeye-toolkits license required")
class TestReactionOeb:
    """Test OEB parsing with reaction SMILES stored as SD data."""

    def test_rxn_smiles_as_sd_tag(self, data_dir, tmp_path):
        csv_path = data_dir / "regression/rxn/rxn.csv"
        oeb_path = tmp_path / "rxn.oeb"

        from openeye import oechem
        import pandas as pd

        df = pd.read_csv(csv_path, index_col=False)

        ofs = oechem.oemolostream()
        ofs.SetFormat(oechem.OEFormat_OEB)
        ofs.open(str(oeb_path))

        for _, row in df.iterrows():
            rxn_smiles = str(row["smiles"])
            reactant_smi = rxn_smiles.split(">")[0]
            mol = oechem.OEGraphMol()
            oechem.OESmilesToMol(mol, reactant_smi)
            oechem.OESetSDData(mol, "smiles", rxn_smiles)
            oechem.OESetSDData(mol, "ea", str(row["ea"]))
            oechem.OEWriteMolecule(ofs, mol)

        ofs.close()

        oeb_data = build_data_from_files(
            oeb_path,
            no_header_row=False,
            smiles_cols=None,
            rxn_cols=["smiles"],
            target_cols=["ea"],
            ignore_cols=None,
            splits_col=None,
            weight_col=None,
            bounded=False,
            p_descriptors=None,
            p_atom_feats={},
            p_bond_feats={},
            p_atom_descs={},
            descriptor_cols=None,
            n_workers=0,
            molecule_featurizers=None,
            keep_h=False,
            add_h=False,
            ignore_stereo=False,
            reorder_atoms=False,
            use_cuikmolmaker_featurization=False,
        )

        # Should produce reaction datapoints
        assert len(oeb_data[1]) == 100
