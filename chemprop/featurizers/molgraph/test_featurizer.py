from molecule import PretrainMoleculeMolGraphFeaturizer
from rdkit import Chem
from rdkit.Chem import Mol
from chemprop.data.collate import PreBatchMolGraph
import copy
SMI = "Cn1nc(CC(=O)Nc2ccc3oc4ccccc4c3c2)c2ccccc2c1=O"
mol = Chem.MolFromSmiles(SMI)
Featurizer =PretrainMoleculeMolGraphFeaturizer()
premolgraph = Featurizer(mol)
premolgraph1 = copy.deepcopy(premolgraph)
premolgraph2 = copy.deepcopy(premolgraph)
premolgraph3 = copy.deepcopy(premolgraph)
premolgraph1.masked_atom_pretraining(0.3)
premolgraph2.subgraph_deletion(3,0.3)
premolgraph3.bond_deletion_complete(0.3)
SMI_list = ["Cn1nc(CC(=O)Nc2ccc3oc4ccccc4c3c2)c2ccccc2c1=O","C", "CC", "CCC", "C1CC1", "C1CCC1"]
molgraph_list = []
for sm in SMI_list:
    mm = Chem.MolFromSmiles(SMI)
    pre_mol = Featurizer(mm)
    molgraph_list.append(pre_mol)
batch = PreBatchMolGraph(molgraph_list)
original_batch = batch.prepare_batch()
mask_batch = copy.deepcopy(batch).apply_mask(0.3)
bond_batch = copy.deepcopy(batch).apply_bond_deletion(0.3)
subgraph_batch = copy.deepcopy(batch).apply_subgraph_deletion(0,0.3)

print('cool')