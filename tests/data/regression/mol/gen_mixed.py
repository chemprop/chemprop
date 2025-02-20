import csv
import json
import random

from rdkit import Chem


def main(args):
    # Load the CSV
    input_filename = "mol.csv"
    output_filename = "mixed.csv"

    # Read the CSV into a list of dictionaries
    with open(input_filename, newline="") as infile:
        reader = csv.DictReader(infile)
        rows = [row for row in reader]

    # Process each row
    for row in rows:
        # Get the SMILES string
        smiles = row["smiles"]

        # Create a molecule object using RDKit
        molecule = Chem.MolFromSmiles(smiles)

        if molecule:
            # Generate random molecule value
            row["molecule"] = random.random()

            # Generate random atom values and convert to JSON string
            atom_values = [random.random() for _ in range(molecule.GetNumAtoms())]
            row["atom"] = json.dumps(atom_values)  # Convert list to JSON string

            # Generate random bond values and convert to JSON string
            bond_values = [random.random() for _ in range(molecule.GetNumBonds())]
            row["bond"] = json.dumps(bond_values)  # Convert list to JSON string

    # Write only the required columns to the new CSV file
    with open(output_filename, mode="w", newline="") as outfile:
        fieldnames = ["smiles", "molecule", "atom", "bond"]  # Only write these columns
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in rows:
            # Only keep the relevant columns ('smiles', 'molecule', 'atom', 'bond')
            filtered_row = {key: row[key] for key in fieldnames}
            writer.writerow(filtered_row)

    print(f"CSV file '{output_filename}' has been updated successfully!")


# Run the main function
if __name__ == "__main__":
    main(None)
