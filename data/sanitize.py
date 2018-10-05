import rdkit.Chem as Chem
import sys

for i,line in enumerate(sys.stdin):
    line = line.strip("\r\n ")
    if i == 0: print line

    items = line.split(',')
    mol = Chem.MolFromSmiles(items[0])
    if mol is None: continue
    print line
