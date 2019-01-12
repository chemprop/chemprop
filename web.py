from argparse import ArgumentParser, Namespace
import os
import shutil
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import List, Tuple
import time
import multiprocessing as mp
import logging

from flask import Flask, json, jsonify, redirect, render_template, request, send_from_directory, url_for
import numpy as np
from rdkit import Chem
import torch
from werkzeug.utils import secure_filename

from chemprop.data.utils import get_data, validate_data
from chemprop.parsing import add_predict_args, add_train_args, modify_train_args
from chemprop.train.make_predictions import make_predictions
from chemprop.train.run_training import run_training
from chemprop.utils import load_task_names, set_logger

TEMP_FOLDER = TemporaryDirectory()

app = Flask(__name__)
app.config['DATA_FOLDER'] = 'web_data'
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)
app.config['CHECKPOINT_FOLDER'] = 'web_checkpoints'
os.makedirs(app.config['CHECKPOINT_FOLDER'], exist_ok=True)
app.config['TEMP_FOLDER'] = TEMP_FOLDER.name
app.config['SMILES_FILENAME'] = 'smiles.csv'
app.config['PREDICTIONS_FILENAME'] = 'predictions.csv'
app.config['CUDA'] = torch.cuda.is_available()
app.config['GPUS'] = list(range(torch.cuda.device_count()))

running = 0
progress = mp.Value('d', 0.0)


def progress_bar(args: Namespace, progress: mp.Value):
    # no code to handle crashes in model training yet, though
    current_epoch = -1
    while current_epoch < args.epochs - 1:
        if os.path.exists(os.path.join(args.save_dir, 'verbose.log')):
            with open(os.path.join(args.save_dir, 'verbose.log'), 'r') as f:
                content = f.read()
                if 'Epoch ' + str(current_epoch + 1) in content:
                    current_epoch += 1
                    progress.value = (current_epoch + 1) * 100 / args.epochs
        else:
            pass
        time.sleep(0)


def get_datasets() -> List[str]:
    return sorted(os.listdir(app.config['DATA_FOLDER']))


def get_checkpoints() -> List[str]:
    return sorted(os.listdir(app.config['CHECKPOINT_FOLDER']))


def find_unique_path(name: str) -> str:
    if not os.path.exists(name):
        return name

    base_name, ext = os.path.splitext(name)

    i = 2
    while os.path.exists(name):
        name = base_name + str(i) + ext
        i += 1

    return name


def name_already_exists_message(thing_being_named: str, original_path: str, new_path: str) -> str:
    return '{} "{}" already exists. Saving to "{}"'.format(
        thing_being_named, os.path.basename(original_path), os.path.basename(new_path))


def get_data_upload_warnings_errors() -> Tuple[List[str], List[str]]:
    warnings_raw = request.args.get('data_upload_warnings')
    errors_raw = request.args.get('data_upload_errors')
    warnings = json.loads(warnings_raw) if warnings_raw is not None else None
    errors = json.loads(errors_raw) if errors_raw is not None else None

    return warnings, errors


@app.route('/receiver', methods=['POST'])
def receiver():
    return jsonify(progress=progress.value, running=running)


@app.route('/')
def home():
    return render_template('home.html')


def render_train(**kwargs):
    data_upload_warnings, data_upload_errors = get_data_upload_warnings_errors()

    return render_template('train.html',
                           datasets=get_datasets(),
                           cuda=app.config['CUDA'],
                           gpus=app.config['GPUS'],
                           data_upload_warnings=data_upload_warnings,
                           data_upload_errors=data_upload_errors,
                           **kwargs)


@app.route('/train', methods=['GET', 'POST'])
def train():
    warnings, errors = [], []

    if request.method == 'GET':
        return render_train()

    # Get arguments
    data_name, epochs, checkpoint_name = \
        request.form['dataName'], int(request.form['epochs']), request.form['checkpointName']
    gpu = request.form.get('gpu')
    data_path = os.path.join(app.config['DATA_FOLDER'], data_name)
    dataset_type = request.form.get('datasetType', 'regression')

    if not checkpoint_name.endswith('.pt'):
        checkpoint_name += '.pt'

    # Create and modify args
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()

    args.data_path = data_path
    args.dataset_type = dataset_type
    args.epochs = epochs

    # Check if regression/classification selection matches data
    data = get_data(data_path)
    targets = data.targets()
    unique_targets = set(np.unique(targets))

    if dataset_type == 'classification' and len(unique_targets - {0, 1}) > 0:
        errors.append('Selected classification dataset but not all labels are 0 or 1. Select regression instead.')

        return render_train(warnings=warnings, errors=errors)

    if dataset_type == 'regression' and unique_targets <= {0, 1}:
        errors.append('Selected regression dataset but all labels are 0 or 1. Select classification instead.')

        return render_train(warnings=warnings, errors=errors)

    if gpu is not None:
        if gpu == 'None':
            args.no_cuda = True
        else:
            args.gpu = int(gpu)

    with TemporaryDirectory() as temp_dir:
        args.save_dir = temp_dir
        modify_train_args(args)

        logger = logging.getLogger('train')
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        set_logger(logger, args.save_dir, args.quiet)

        global progress, running
        process = mp.Process(target=progress_bar, args=(args, progress))
        process.start()
        running = 1

        # Run training
        run_training(args, logger)
        process.join()

        # Reset globals
        running = 0
        progress = mp.Value('d', 0.0)

        # Check if name overlap
        original_save_path = os.path.join(app.config['CHECKPOINT_FOLDER'], checkpoint_name)
        save_path = find_unique_path(original_save_path)
        if save_path != original_save_path:
            warnings.append(name_already_exists_message('Checkpoint', original_save_path, save_path))

        # Move checkpoint
        shutil.move(os.path.join(args.save_dir, 'model_0', 'model.pt'), save_path)

    return render_train(trained=True, warnings=warnings, errors=errors)


def render_predict(**kwargs):
    return render_template('predict.html',
                           checkpoints=get_checkpoints(),
                           cuda=app.config['CUDA'],
                           gpus=app.config['GPUS'],
                           **kwargs)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_predict()

    # Get arguments
    checkpoint_name = request.form['checkpointName']

    if 'data' in request.files:
        # Upload data file with SMILES
        show_file_upload = True
        data = request.files['data']
        data_name = secure_filename(data.filename)
        data_path = os.path.join(app.config['TEMP_FOLDER'], data_name)
        data.save(data_path)

        smiles = []
        with open(data_path, 'r') as f:
            header = f.readline()
            try:  # if there's no header, add the smiles in the first line
                possible_smiles = header.strip().split(',')[0]
                mol = Chem.MolFromSmiles(possible_smiles)
                smiles.append(possible_smiles)
            except:
                pass
            for line in f:
                smiles.append(line.strip().split(',')[0])
    else:
        show_file_upload = False
        smiles = request.form['smiles']
        smiles = smiles.split()

    checkpoint_path = os.path.join(app.config['CHECKPOINT_FOLDER'], checkpoint_name)
    task_names = load_task_names(checkpoint_path)
    gpu = request.form.get('gpu')

    # Create and modify args
    parser = ArgumentParser()
    add_predict_args(parser)
    args = parser.parse_args()

    preds_path = os.path.join(app.config['TEMP_FOLDER'], app.config['PREDICTIONS_FILENAME'])
    args.preds_path = preds_path
    args.checkpoint_paths = [checkpoint_path]
    if gpu is not None:
        if gpu == 'None':
            args.no_cuda = True
        else:
            args.gpu = int(gpu)

    invalid_smiles_warning = "Invalid SMILES String"
    if len(smiles) > 0:
        # Run prediction
        preds = make_predictions(args, smiles=smiles, invalid_smiles_warning=invalid_smiles_warning)
    else:
        preds = []

    return render_predict(predicted=True,
                          smiles=smiles,
                          num_smiles=min(10, len(smiles)),
                          show_more=max(0, len(smiles)-10),
                          task_names=task_names,
                          num_tasks=len(task_names),
                          preds=preds,
                          show_file_upload=show_file_upload,
                          warnings=["List contains invalid SMILES strings"] if invalid_smiles_warning in preds else None,
                          errors=["No SMILES strings given"] if len(preds) == 0 else None)


@app.route('/download_predictions')
def download_predictions():
    return send_from_directory(app.config['TEMP_FOLDER'], app.config['PREDICTIONS_FILENAME'], as_attachment=True)


@app.route('/data')
def data():
    data_upload_warnings, data_upload_errors = get_data_upload_warnings_errors()

    return render_template('data.html',
                           datasets=get_datasets(),
                           data_upload_warnings=data_upload_warnings,
                           data_upload_errors=data_upload_errors)


@app.route('/data/upload/<string:return_page>', methods=['POST'])
def upload_data(return_page: str):
    warnings, errors = [], []

    data = request.files['data']

    with NamedTemporaryFile() as temp_file:
        data.save(temp_file.name)
        data_errors = validate_data(temp_file.name)

        if len(data_errors) > 0:
            errors.extend(data_errors)
        else:
            data_name = secure_filename(data.filename)
            original_data_path = os.path.join(app.config['DATA_FOLDER'], data_name)

            data_path = find_unique_path(original_data_path)
            if data_path != original_data_path:
                warnings.append(name_already_exists_message('Data', original_data_path, data_path))

            shutil.copy(temp_file.name, data_path)

    warnings, errors = json.dumps(warnings), json.dumps(errors)

    return redirect(url_for(return_page, data_upload_warnings=warnings, data_upload_errors=errors))


@app.route('/data/download/<string:dataset>')
def download_data(dataset: str):
    return send_from_directory(app.config['DATA_FOLDER'], dataset, as_attachment=True)


@app.route('/data/delete/<string:dataset>')
def delete_data(dataset: str):
    os.remove(os.path.join(app.config['DATA_FOLDER'], dataset))
    return redirect(url_for('data'))


@app.route('/checkpoints')
def checkpoints():
    return render_template('checkpoints.html', checkpoints=get_checkpoints())


@app.route('/checkpoints/upload/<string:return_page>', methods=['POST'])
def upload_checkpoint(return_page: str):
    checkpoint = request.files['checkpoint']
    checkpoint_name = secure_filename(checkpoint.filename)
    checkpoint_path = os.path.join(app.config['CHECKPOINT_FOLDER'], checkpoint_name)
    checkpoint.save(checkpoint_path)

    return redirect(url_for(return_page))


@app.route('/checkpoints/download/<string:checkpoint>')
def download_checkpoint(checkpoint: str):
    return send_from_directory(app.config['CHECKPOINT_FOLDER'], checkpoint, as_attachment=True)


@app.route('/checkpoints/delete/<string:checkpoint>')
def delete_checkpoint(checkpoint: str):
    os.remove(os.path.join(app.config['CHECKPOINT_FOLDER'], checkpoint))
    return redirect(url_for('checkpoints'))


if __name__ == "__main__":
    app.run(debug=True)
