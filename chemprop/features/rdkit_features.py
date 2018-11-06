import numpy as np

from rdkit import Chem
# from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem.GraphDescriptors import BalabanJ, BertzCT, Ipc, Chi0, Chi0n, Chi0v, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, \
    Chi3n, Chi3v, Chi4n, Chi4v, HallKierAlpha, Kappa1, Kappa2, Kappa3
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit.Chem.Descriptors import MolWt, ExactMolWt, HeavyAtomMolWt, NumValenceElectrons
from rdkit.Chem.Lipinski import HeavyAtomCount, NHOHCount, NOCount, NumHAcceptors, NumHDonors, NumHeteroatoms, \
    NumRotatableBonds, NumAromaticRings, NumSaturatedRings, NumAliphaticRings, NumAromaticHeterocycles, \
    NumAromaticCarbocycles, NumSaturatedHeterocycles, NumSaturatedCarbocycles, NumAliphaticHeterocycles, \
    NumAliphaticCarbocycles, RingCount, FractionCSP3
from rdkit.Chem.rdMolDescriptors import CalcNumSpiroAtoms, CalcNumBridgeheadAtoms, CalcTPSA, CalcLabuteASA, PEOE_VSA_, \
    SMR_VSA_, SlogP_VSA_, MQNs_, CalcAUTOCORR2D, CalcNumAmideBonds
from rdkit.Chem.Fragments import fr_ketone_Topliss
from rdkit.Chem.EState.EState_VSA import EState_VSA1, EState_VSA2, EState_VSA3, EState_VSA4, EState_VSA5, EState_VSA6, \
    EState_VSA7, EState_VSA8, EState_VSA9, EState_VSA10, EState_VSA11, VSA_EState1, VSA_EState2, VSA_EState3, \
    VSA_EState4, VSA_EState5, VSA_EState6, VSA_EState7, VSA_EState8, VSA_EState9, VSA_EState10

FEATURE_FUNCTIONS = [
    # ComputeGasteigerCharges,
    BalabanJ, BertzCT, Ipc, Chi0, Chi0n, Chi0v, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3n, Chi3v, Chi4n, Chi4v,
    HallKierAlpha, Kappa1, Kappa2, Kappa3,
    MolLogP, MolMR,
    MolWt, ExactMolWt, HeavyAtomMolWt, NumValenceElectrons,
    HeavyAtomCount, NHOHCount, NOCount, NumHAcceptors, NumHDonors, NumHeteroatoms, NumRotatableBonds, NumAromaticRings,
    NumSaturatedRings, NumAliphaticRings, NumAromaticHeterocycles, NumAromaticCarbocycles, NumSaturatedHeterocycles,
    NumSaturatedCarbocycles, NumAliphaticHeterocycles, NumAliphaticCarbocycles, RingCount, FractionCSP3,
    CalcNumSpiroAtoms, CalcNumBridgeheadAtoms, CalcTPSA, CalcLabuteASA, PEOE_VSA_, SMR_VSA_, SlogP_VSA_, MQNs_,
    CalcAUTOCORR2D, CalcNumAmideBonds,
    fr_ketone_Topliss,
    EState_VSA1, EState_VSA2, EState_VSA3, EState_VSA4, EState_VSA5, EState_VSA6, EState_VSA7, EState_VSA8, EState_VSA9,
    EState_VSA10, EState_VSA11, VSA_EState1, VSA_EState2, VSA_EState3, VSA_EState4, VSA_EState5, VSA_EState6,
    VSA_EState7, VSA_EState8, VSA_EState9, VSA_EState10
]


def rdkit_2d_features(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    features = []
    for f in FEATURE_FUNCTIONS:
        try:
            feature = f(mol)
        except:  # very very rarely, something like BalabanJ crashes
            dummy_mol = Chem.MolFromSmiles('c1ccc2cc(CC3=NCCN3)ccc2c1')
            feature = f(dummy_mol)
        if type(feature) == list:
            features.extend(feature)
        else:
            features.append(feature)
    return np.clip(np.nan_to_num(np.array(features)), -1e2, 1e2)
