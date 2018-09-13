from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
import sys

#header line
sys.stdin.readline()

for line in sys.stdin:
    smiles, label = line.split(',')
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    data = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, data)
    print smiles, ''.join([str(int(x)) for x in data])


