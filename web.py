from flask import Flask, render_template, request
import os
from typing import List

import torch

from chemprop.data import MoleculeDataset, MoleculeDatapoint
from chemprop.utils import load_args, load_checkpoint, load_scalers

app = Flask(__name__)


def make_predictions(checkpoint_path: str, smiles: List[str]) -> List[List[float]]:
    """Makes predictions for a SMILES string."""
    # Load scalers and training args
    scaler, features_scaler = load_scalers(checkpoint_path)
    train_args = load_args(checkpoint_path)

    # Conver smiles to data
    data = MoleculeDataset([MoleculeDatapoint([smile]) for smile in smiles])

    # Normalize features
    if train_args.features_scaling:
        data.normalize_features(features_scaler)

    # Load model
    model = load_checkpoint(checkpoint_path)
    model.eval()

    # Make predictions
    with torch.no_grad():
        preds = model(data.smiles(), data.features())
        preds = preds.data.cpu().numpy()
        if scaler is not None:
            preds = scaler.inverse_transform(preds)
            preds = preds.tolist()

    return preds


def get_task_names(checkpoint_path: str) -> List[str]:
    args = load_args(checkpoint_path)

    return args.task_names


@app.route("/")
def home():
    return render_template('home.html')


@app.route('/', methods=['POST'])
def predict():
    checkpoint, smiles = request.form['checkpoint'], request.form['smiles']
    smiles = smiles.split()

    if not os.path.exists(checkpoint):
        return render_template('home.html', error='Error: checkpoint not found.')

    task_names = get_task_names(checkpoint)
    preds = make_predictions(checkpoint, smiles)

    return render_template('home.html',
                           smiles=smiles,
                           num_smiles=len(smiles),
                           task_names=task_names,
                           num_tasks=len(task_names),
                           preds=preds)


if __name__ == "__main__":
    app.run(debug=True)
