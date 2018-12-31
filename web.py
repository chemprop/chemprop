from argparse import ArgumentParser
import os
from tempfile import TemporaryDirectory
from typing import List

from flask import Flask, render_template, request, send_from_directory
import torch
from werkzeug.utils import secure_filename

from chemprop.data import MoleculeDataset, MoleculeDatapoint
from chemprop.parsing import add_train_args, modify_train_args
from chemprop.train.run_training import run_training
from chemprop.utils import load_args, load_checkpoint, load_scalers


UPLOAD_FOLDER = TemporaryDirectory()
DOWNLOAD_FOLDER = TemporaryDirectory()
PREDICTIONS_FILENAME = 'predictions.csv'
MODEL_FILENAME = 'model.pt'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER.name
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER.name
app.config['PREDICTIONS_FILENAME'] = PREDICTIONS_FILENAME
app.config['MODEL_FOLDER'] = os.path.join(app.config['DOWNLOAD_FOLDER'], 'model_0')
app.config['MODEL_FILENAME'] = MODEL_FILENAME


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


def save_predictions(save_path: str,
                     task_names: List[str],
                     smiles: List[str],
                     preds: List[List[float]]):
    with open(save_path, 'w') as f:
        f.write('smiles,' + ','.join(task_names) + '\n')
        for smile, pred in zip(smiles, preds):
            f.write(smile + ',' + ','.join(str(p) for p in pred) + '\n')


def get_task_names(checkpoint_path: str) -> List[str]:
    args = load_args(checkpoint_path)

    return args.task_names


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'GET':
        return render_template('train.html')

    # Get dataset type and epochs
    dataset_type, epochs = request.form['datasetType'], int(request.form['epochs'])

    # Upload data file
    data = request.files['data']
    data_name = secure_filename(data.filename)
    data_path = os.path.join(app.config['UPLOAD_FOLDER'], data_name)
    data.save(data_path)

    # Create and modify args
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()

    args.data_path = data_path
    args.dataset_type = dataset_type
    args.save_dir = app.config['DOWNLOAD_FOLDER']
    args.epochs = epochs

    modify_train_args(args)

    # Run training
    run_training(args)

    return render_template('train.html', trained=True)


@app.route('/download_model')
def download_model():
    return send_from_directory(app.config['MODEL_FOLDER'], app.config['MODEL_FILENAME'], as_attachment=True)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    # Get smiles
    smiles = request.form['smiles']
    smiles = smiles.split()

    # Upload model checkpoint
    checkpoint = request.files['checkpoint']
    checkpoint_name = secure_filename(checkpoint.filename)
    checkpoint_path = os.path.join(app.config['UPLOAD_FOLDER'], checkpoint_name)
    checkpoint.save(checkpoint_path)

    # Run prediction
    task_names = get_task_names(checkpoint_path)
    preds = make_predictions(checkpoint_path, smiles)

    # Save preds
    preds_path = os.path.join(app.config['DOWNLOAD_FOLDER'], app.config['PREDICTIONS_FILENAME'])
    save_predictions(preds_path, task_names, smiles, preds)

    return render_template('predict.html',
                           predicted=True,
                           smiles=smiles,
                           num_smiles=len(smiles),
                           task_names=task_names,
                           num_tasks=len(task_names),
                           preds=preds)


@app.route('/download_predictions')
def download_predictions():
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], app.config['PREDICTIONS_FILENAME'], as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
