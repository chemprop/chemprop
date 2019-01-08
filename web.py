from argparse import ArgumentParser
import os
from tempfile import TemporaryDirectory
from typing import List

from flask import Flask, redirect, render_template, request, send_from_directory, url_for
import torch
from werkzeug.utils import secure_filename

from chemprop.data import MoleculeDataset, MoleculeDatapoint
from chemprop.parsing import add_train_args, modify_train_args
from chemprop.train.run_training import run_training
from chemprop.utils import load_args, load_checkpoint, load_scalers


app = Flask(__name__)
app.config['DATA_FOLDER'] = 'web_data'
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)
app.config['CHECKPOINT_FOLDER'] = 'web_checkpoints'
os.makedirs(app.config['CHECKPOINT_FOLDER'], exist_ok=True)
app.config['PREDICTIONS_FOLDER'] = 'web_predictions'
os.makedirs(app.config['PREDICTIONS_FOLDER'], exist_ok=True)
app.config['PREDICTIONS_FILENAME'] = 'predictions.csv'


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


def get_datasets() -> List[str]:
    return os.listdir(app.config['DATA_FOLDER'])


def get_checkpoints() -> List[str]:
    return os.listdir(app.config['CHECKPOINT_FOLDER'])


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'GET':
        return render_template('train.html', datasets=get_datasets())

    # Get arguments
    data_name, dataset_type, epochs, checkpoint_name = \
        request.form['dataName'], request.form['datasetType'], int(request.form['epochs']), request.form['checkpointName']

    if not checkpoint_name.endswith('.pt'):
        checkpoint_name += '.pt'

    # Create and modify args
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()

    args.data_path = os.path.join(app.config['DATA_FOLDER'], data_name)
    args.dataset_type = dataset_type
    args.epochs = epochs

    with TemporaryDirectory() as temp_dir:
        args.save_dir = temp_dir
        modify_train_args(args)

        # Run training
        run_training(args)

        # Move checkpoint
        os.rename(os.path.join(args.save_dir, 'model_0', 'model.pt'),
                  os.path.join(app.config['CHECKPOINT_FOLDER'], checkpoint_name))

    return render_template('train.html', datasets=get_datasets(), trained=True)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html', checkpoints=get_checkpoints())

    # Get smiles and checkpoint path
    smiles, checkpoint_name = request.form['smiles'], request.form['checkpointName']
    smiles = smiles.split()
    checkpoint_path = os.path.join(app.config['CHECKPOINT_FOLDER'], checkpoint_name)

    # Run prediction
    task_names = get_task_names(checkpoint_path)
    preds = make_predictions(checkpoint_path, smiles)

    # Save preds
    preds_path = os.path.join(app.config['PREDICTIONS_FOLDER'], app.config['PREDICTIONS_FILENAME'])
    save_predictions(preds_path, task_names, smiles, preds)

    return render_template('predict.html',
                           checkpoints=get_checkpoints(),
                           predicted=True,
                           smiles=smiles,
                           num_smiles=len(smiles),
                           task_names=task_names,
                           num_tasks=len(task_names),
                           preds=preds)


@app.route('/download_predictions')
def download_predictions():
    return send_from_directory(app.config['PREDICTIONS_FOLDER'], app.config['PREDICTIONS_FILENAME'], as_attachment=True)


@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'GET':
        return render_template('data.html', datasets=get_datasets())

    # Upload data file
    data = request.files['data']
    data_name = secure_filename(data.filename)
    data_path = os.path.join(app.config['DATA_FOLDER'], data_name)
    data.save(data_path)

    return render_template('data.html', datasets=get_datasets())


@app.route('/data/download/<string:dataset>')
def download_data(dataset: str):
    return send_from_directory(app.config['DATA_FOLDER'], dataset, as_attachment=True)


@app.route('/data/delete/<string:dataset>')
def delete_data(dataset: str):
    os.remove(os.path.join(app.config['DATA_FOLDER'], dataset))
    return redirect(url_for('data'))


@app.route('/checkpoints', methods=['GET', 'POST'])
def checkpoints():
    if request.method == 'GET':
        return render_template('checkpoints.html', checkpoints=get_checkpoints())

    # Upload checkpoint file
    checkpoint = request.files['data']
    checkpoint_name = secure_filename(checkpoint.filename)
    checkpoint_path = os.path.join(app.config['CHECKPOINT_FOLDER'], checkpoint_name)
    checkpoint.save(checkpoint_path)

    return render_template('checkpoints.html', checkpoints=get_checkpoints())


@app.route('/checkpoints/download/<string:checkpoint>')
def download_checkpoint(checkpoint: str):
    return send_from_directory(app.config['CHECKPOINT_FOLDER'], checkpoint, as_attachment=True)


@app.route('/checkpoints/delete/<string:checkpoint>')
def delete_checkpoint(checkpoint: str):
    os.remove(os.path.join(app.config['CHECKPOINT_FOLDER'], checkpoint))
    return redirect(url_for('checkpoints'))


if __name__ == "__main__":
    app.run(debug=True)
