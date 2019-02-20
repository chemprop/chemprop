"""Runs the web interface version of chemprop, allowing for training and predicting in a web browser."""

from argparse import ArgumentParser, Namespace
import os
import sys
import shutil
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import List, Tuple
import time
import multiprocessing as mp
import sqlite3
import click

from flask import Flask, json, jsonify, redirect, render_template, request, send_from_directory, url_for
from flask.cli import with_appcontext
import numpy as np
from rdkit import Chem
import torch
from werkzeug.utils import secure_filename

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from web import db
from chemprop.data.utils import get_data, get_header, get_smiles, validate_data
from chemprop.parsing import add_predict_args, add_train_args, modify_predict_args, modify_train_args
from chemprop.train.make_predictions import make_predictions
from chemprop.train.run_training import run_training
from chemprop.utils import create_logger, load_task_names

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

TRAINING = 0
PROGRESS = mp.Value('d', 0.0)


def progress_bar(args: Namespace, progress: mp.Value):
    """
    Updates a progress bar displayed during training.

    :param args: Arguments.
    :param progress: The current progress.
    """
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
    """
    Gets a list of the available datasets.

    :return: A list of dataset names.
    """
    return sorted(os.listdir(app.config['DATA_FOLDER']))


def get_checkpoints() -> List[str]:
    """
    Gets a list of the available checkpoints.

    :return: A list of checkpoint names.
    """
    return sorted(os.listdir(app.config['CHECKPOINT_FOLDER']))


def find_unused_path(path: str) -> str:
    """
    Given an initial path, finds an unused path by appending different numbers to the filename.

    :param path: An initial path.
    :return: An unused path.
    """
    if not os.path.exists(path):
        return path

    base_name, ext = os.path.splitext(path)

    i = 2
    while os.path.exists(path):
        path = base_name + str(i) + ext
        i += 1

    return path


def name_already_exists_message(thing_being_named: str, original_path: str, new_path: str) -> str:
    """
    Creates a message about a path already existing and therefore being renamed.

    :param thing_being_named: The thing being renamed (ex. Data, Checkpoint).
    :param original_path: The original path where the object was supposed to be saved.
    :param new_path: The new path where the object will actually be saved.
    :return: A string with a message about the changed path.
    """
    return f'{thing_being_named} "{os.path.basename(original_path)} already exists. ' \
           f'Saving to "{os.path.basename(new_path)}".'


def get_upload_warnings_errors(upload_item: str) -> Tuple[List[str], List[str]]:
    """
    Gets any upload warnings passed along in the request.

    :param upload_item: The thing being uploaded (ex. Data, Checkpoint).
    :return: A tuple with a list of warning messages and a list of error messages.
    """
    warnings_raw = request.args.get(f'{upload_item}_upload_warnings')
    errors_raw = request.args.get(f'{upload_item}_upload_errors')
    warnings = json.loads(warnings_raw) if warnings_raw is not None else None
    errors = json.loads(errors_raw) if errors_raw is not None else None

    return warnings, errors


def format_float(value: float, precision: int = 4) -> str:
    """
    Formats a float value to a specific precision.

    :param value: The float value to format.
    :param precision: The number of decimal places to use.
    :return: A string containing the formatted float.
    """
    return f'{value:.{precision}f}'


def format_float_list(array: List[float], precision: int = 4) -> List[str]:
    """
    Formats a list of float values to a specific precision.

    :param array: A list of float values to format.
    :param precision: The number of decimal places to use.
    :return: A list of strings containing the formatted floats.
    """
    return [format_float(f, precision) for f in array]


@app.route('/receiver', methods=['POST'])
def receiver():
    """Receiver monitoring the progress of training."""
    return jsonify(progress=PROGRESS.value, training=TRAINING)


@app.route('/')
def home():
    """Renders the home page."""
    return render_template('home.html', users=db.get_all_users())

@app.route('/change_user', methods=['GET', 'POST'])
def change_user():
    """    
    If a POST request is made, creates a new user.
    Renders the change_user page.
    """
    if request.method == 'GET':   
        return render_template('change_user.html', users=db.get_all_users())

    new_name = request.form['newUserName']

    if new_name != None:
        db.insert_user(new_name)

    return redirect(url_for('change_user'))

def render_train(**kwargs):
    """Renders the train page with specified kwargs."""
    data_upload_warnings, data_upload_errors = get_upload_warnings_errors('data')

    return render_template('train.html',
                           datasets=get_datasets(),
                           cuda=app.config['CUDA'],
                           gpus=app.config['GPUS'],
                           data_upload_warnings=data_upload_warnings,
                           data_upload_errors=data_upload_errors,
                           users=db.get_all_users(),
                           **kwargs)

@app.route('/train', methods=['GET', 'POST'])
def train():
    """Renders the train page and performs training if request method is POST."""
    global PROGRESS, TRAINING

    warnings, errors = [], []

    if request.method == 'GET':
        return render_train()

    # Get arguments
    data_name, epochs, checkpoint_name = \
        request.form['dataName'], int(request.form['epochs']), request.form['checkpointName']
    gpu = request.form.get('gpu')
    data_path = os.path.join(app.config['DATA_FOLDER'], data_name)
    dataset_type = request.form.get('datasetType', 'regression')

    # Create and modify args
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args([])

    args.data_path = data_path
    args.dataset_type = dataset_type
    args.epochs = epochs

    # Check if regression/classification selection matches data
    data = get_data(path=data_path)
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

    current_user = request.cookies.get('currentUser')

    if not current_user:
        # Use DEFAULT as current user if the client's cookie is not set.
        current_user = 1
    
    
    ckpt_id, ckpt_name = db.insert_ckpt(checkpoint_name, current_user, args.dataset_type, args.epochs)

    with TemporaryDirectory() as temp_dir:
        args.save_dir = temp_dir
        modify_train_args(args)

        process = mp.Process(target=progress_bar, args=(args, PROGRESS))
        process.start()
        TRAINING = 1

        # Run training
        logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
        task_scores = run_training(args, logger)
        process.join()

        # Reset globals
        TRAINING = 0
        PROGRESS = mp.Value('d', 0.0)

        # Check if name overlap
        original_save_path = os.path.join(app.config['CHECKPOINT_FOLDER'], str(ckpt_id) + ".pt")
        save_path = find_unused_path(original_save_path)
        if save_path != original_save_path:
            warnings.append(name_already_exists_message('Checkpoint', original_save_path, save_path))

        # Move checkpoint
        shutil.move(os.path.join(args.save_dir, 'model_0', 'model.pt'), save_path)

    return render_train(trained=True,
                        metric=args.metric,
                        num_tasks=len(args.task_names),
                        task_names=args.task_names,
                        task_scores=format_float_list(task_scores),
                        mean_score=format_float(np.mean(task_scores)),
                        warnings=warnings,
                        errors=errors)


def render_predict(**kwargs):
    """Renders the predict page with specified kwargs"""
    checkpoint_upload_warnings, checkpoint_upload_errors = get_upload_warnings_errors('checkpoint')

    return render_template('predict.html',
                           checkpoints=db.get_ckpts(request.cookies.get('currentUser')),
                           cuda=app.config['CUDA'],
                           gpus=app.config['GPUS'],
                           checkpoint_upload_warnings=checkpoint_upload_warnings,
                           checkpoint_upload_errors=checkpoint_upload_errors,
                           users=db.get_all_users(),
                           **kwargs)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Renders the predict page and makes predictions if the method is POST."""
    if request.method == 'GET':
        return render_predict()

    # Get arguments
    checkpoint_name = request.form['checkpointName']

    if 'data' in request.files:
        # Upload data file with SMILES
        data = request.files['data']
        data_name = secure_filename(data.filename)
        data_path = os.path.join(app.config['TEMP_FOLDER'], data_name)
        data.save(data_path)

        # Check if header is smiles
        possible_smiles = get_header(data_path)[0]
        smiles = [possible_smiles] if Chem.MolFromSmiles(possible_smiles) is not None else []

        # Get remaining smiles
        smiles.extend(get_smiles(data_path))
    elif request.form['textSmiles'] != '':
        smiles = request.form['textSmiles'].split()
    else:
        smiles = [request.form['drawSmiles']]

    checkpoint_path = os.path.join(app.config['CHECKPOINT_FOLDER'], checkpoint_name)
    task_names = load_task_names(checkpoint_path)
    num_tasks = len(task_names)
    gpu = request.form.get('gpu')

    # Create and modify args
    parser = ArgumentParser()
    add_predict_args(parser)
    args = parser.parse_args([])

    preds_path = os.path.join(app.config['TEMP_FOLDER'], app.config['PREDICTIONS_FILENAME'])
    args.test_path = 'None'  # TODO: Remove this hack to avoid assert crashing in modify_predict_args
    args.preds_path = preds_path
    args.checkpoint_path = checkpoint_path
    if gpu is not None:
        if gpu == 'None':
            args.no_cuda = True
        else:
            args.gpu = int(gpu)

    modify_predict_args(args)

    # Run predictions
    preds = make_predictions(args, smiles=smiles)

    if all(p is None for p in preds):
        return render_predict(errors=['All SMILES are invalid'])

    # Replace invalid smiles with message
    invalid_smiles_warning = "Invalid SMILES String"
    preds = [pred if pred is not None else [invalid_smiles_warning] * num_tasks for pred in preds]

    return render_predict(predicted=True,
                          smiles=smiles,
                          num_smiles=min(10, len(smiles)),
                          show_more=max(0, len(smiles)-10),
                          task_names=task_names,
                          num_tasks=len(task_names),
                          preds=preds,
                          warnings=["List contains invalid SMILES strings"] if None in preds else None,
                          errors=["No SMILES strings given"] if len(preds) == 0 else None)


@app.route('/download_predictions')
def download_predictions():
    """Downloads predictions as a .csv file."""
    return send_from_directory(app.config['TEMP_FOLDER'], app.config['PREDICTIONS_FILENAME'], as_attachment=True)


@app.route('/data')
def data():
    """Renders the data page."""
    data_upload_warnings, data_upload_errors = get_upload_warnings_errors('data')

    return render_template('data.html',
                           datasets=get_datasets(),
                           data_upload_warnings=data_upload_warnings,
                           data_upload_errors=data_upload_errors,
                           users=db.get_all_users())


@app.route('/data/upload/<string:return_page>', methods=['POST'])
def upload_data(return_page: str):
    """
    Uploads a data .csv file.

    :param return_page: The name of the page to render to after uploading the dataset.
    """
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

            data_path = find_unused_path(original_data_path)
            if data_path != original_data_path:
                warnings.append(name_already_exists_message('Data', original_data_path, data_path))

            shutil.copy(temp_file.name, data_path)

    warnings, errors = json.dumps(warnings), json.dumps(errors)

    return redirect(url_for(return_page, data_upload_warnings=warnings, data_upload_errors=errors))


@app.route('/data/download/<string:dataset>')
def download_data(dataset: str):
    """
    Downloads a dataset as a .csv file.

    :param dataset: The name of the dataset to download.
    """
    return send_from_directory(app.config['DATA_FOLDER'], dataset, as_attachment=True)


@app.route('/data/delete/<string:dataset>')
def delete_data(dataset: str):
    """
    Deletes a dataset.

    :param dataset: The name of the dataset to delete.
    """
    os.remove(os.path.join(app.config['DATA_FOLDER'], dataset))
    return redirect(url_for('data'))


@app.route('/checkpoints')
def checkpoints():
    """Renders the checkpoints page."""
    checkpoint_upload_warnings, checkpoint_upload_errors = get_upload_warnings_errors('checkpoint')

    return render_template('checkpoints.html',
                           checkpoints=get_checkpoints(),
                           checkpoint_upload_warnings=checkpoint_upload_warnings,
                           checkpoint_upload_errors=checkpoint_upload_errors,
                           users=db.get_all_users())


@app.route('/checkpoints/upload/<string:return_page>', methods=['POST'])
def upload_checkpoint(return_page: str):
    """
    Uploads a checkpoint .pt file.

    :param return_page: The name of the page to render after uploading the checkpoint file.
    """
    warnings, errors = [], []

    checkpoint = request.files['checkpoint']
    checkpoint_name = secure_filename(checkpoint.filename)
    original_checkpoint_path = os.path.join(app.config['CHECKPOINT_FOLDER'], checkpoint_name)

    checkpoint_path = find_unused_path(original_checkpoint_path)
    if checkpoint_path != original_checkpoint_path:
        warnings.append(name_already_exists_message('Checkpoint', original_checkpoint_path, checkpoint_path))

    checkpoint.save(checkpoint_path)

    warnings, errors = json.dumps(warnings), json.dumps(errors)

    return redirect(url_for(return_page, checkpoint_upload_warnings=warnings, checkpoint_upload_errors=errors))


@app.route('/checkpoints/download/<string:checkpoint>')
def download_checkpoint(checkpoint: str):
    """
    Downloads a checkpoint .pt file.

    :param checkpoint: The name of the checkpoint file to download.
    """
    return send_from_directory(app.config['CHECKPOINT_FOLDER'], checkpoint, as_attachment=True)


@app.route('/checkpoints/delete/<string:checkpoint>')
def delete_checkpoint(checkpoint: str):
    """
    Deletes a checkpoint file.

    :param checkpoint: The name of the checkpoint to delete.
    """
    os.remove(os.path.join(app.config['CHECKPOINT_FOLDER'], checkpoint))
    return redirect(url_for('checkpoints'))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host IP address')
    parser.add_argument('--port', type=int, default=5000, help='Port')
    parser.add_argument('--debug', action='store_true', default=False, help='Whether to run in debug mode')
    parser.add_argument('--initdb', action='store_true', default=False, help='Initialize Database')
    args = parser.parse_args()

    db.init_app(app)

    if args.initdb or not os.path.isfile('chemprop.sqlite3'):
        with app.app_context():
            db.init_db()
            print("-- INITIALIZED DATABASE --")

    app.run(host=args.host, port=args.port, debug=args.debug)