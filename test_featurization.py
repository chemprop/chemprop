

from chemprop.features import MolGraph
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.atom_messages=False
    m = MolGraph('CC',args)
    print(m.f_atoms)
    print(m.f_bonds)
