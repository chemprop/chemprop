

from chemprop.features import MolGraph
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.atom_messages=False
    m = MolGraph('C/C=C/C(=O)N1CCC(CC1)CC(=O)N2CC[C@@H](C2)Nc3ccc4cc(ccc4n3)OC', args)
    print(m.f_atoms)
    print(m.f_bonds)
