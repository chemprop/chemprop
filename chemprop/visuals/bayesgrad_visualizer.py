from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


class BayesEnsembleVisualizer(object):

    @staticmethod
    def color(x):
        """Given a gradient value between -1 and 1, either return
        blue or red color, respectively."""
        if x > 0:
            # x=0 -> 1, 1, 1 (white)
            # x=1 -> 1, 0.5, 0.5 (red)
            return 1., 1. - x/2, 1. - x/2
        else:
            # x=0 -> 1, 1, 1 (white)
            # x=1 -> 0.5, 0.5, 1 (blue)
            return 1. + x/2, 1. + x/2, 1.

    def visualize(self, smiles, avg_grads):
        """Visualize average (across ensemble) sum of gradients on atoms/bonds for a given SMILES string

        :param smiles: Smiles string in question
        :type smiles: string

        :param avg_grads: Sum of gradients on each bond & atom in the molecule, as indexed in `chemprop.features.featurization.MolGraph` class.
        :type avg_grads: dict

        :return: string representing SVG object.
        """

        # If we do not have any gradient date (in the case of ions), then return None
        if not avg_grads['atoms'] or not avg_grads['bonds']:
            return None

        # Convert the SMILES string to a molecule
        mol = Chem.MolFromSmiles(smiles)

        # Compute 2D coordinates for this molecule
        rdDepictor.Compute2DCoords(mol)

        # Get the number of atoms in the molecule
        n_atoms = mol.GetNumAtoms()

        # Get bond features. Same loop order as in `chemprop.features.featurization.MolGraph` class.
        # Doing this loop we get the bond indexes in the molecular structure (needed for visualization)
        # and at the same time we can sum the gradients we have for the directed bonds (one for each direction)
        bond_counter = 0
        bond_grads = {}
        for a1 in range(n_atoms):
            for a2 in range(a1 + 1, n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)
                if bond is None:
                    continue
                # Get the index of bond in the molecular graph
                bond_idx = bond.GetIdx()
                # Sum up the gradients for both ways
                bond_grad = avg_grads['bonds'][bond_counter] + avg_grads['bonds'][bond_counter+1]
                # Store this mapping of bond index to summed gradient
                bond_grads[bond_idx] = bond_grad
                # Increase the counter for going through the features
                bond_counter += 2

        # For clarity, set the summed atom gradients to its own variable
        # This is mapping atom IDs to summed gradient on that atom
        atom_grads = avg_grads['atoms']

        # Scale average gradient sums such that the maximum absolute value is 1
        scale = max(
            np.concatenate([
                np.abs(list(atom_grads.values())),
                np.abs(list(bond_grads.values()))
            ])
        )
        atom_grads = {k: v / scale for k, v in atom_grads.items()}
        bond_grads = {k: v / scale for k, v in bond_grads.items()}

        # Get object for drawing molecule
        drawer = rdMolDraw2D.MolDraw2DSVG(500, 375)

        # Set some drawing options so as to not color atom labels and bonds
        opts = drawer.drawOptions()

        # Use black white
        opts.useBWAtomPalette()

        # Draw the molecule
        drawer.DrawMolecule(
            mol,
            highlightAtoms=atom_grads.keys(),
            highlightAtomColors={i: self.color(atom_grads[i]) for i in atom_grads.keys()},
            highlightBonds=bond_grads.keys(),
            highlightBondColors={i: self.color(bond_grads[i]) for i in bond_grads.keys()},
            legend=smiles
        )
        drawer.FinishDrawing()

        # Get the SVG file
        svg = drawer.GetDrawingText()
        return svg
