from argparse import ArgumentParser, Namespace
import os
import shutil
from tempfile import TemporaryDirectory
from typing import List
import time
import multiprocessing as mp
import logging

from flask import Flask, redirect, render_template, request, send_from_directory, url_for, jsonify
import torch
from werkzeug.utils import secure_filename

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

started = 0
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
                    progress.value = (current_epoch+1)*100/args.epochs # TODO communicate with other process
        else:
            pass
        time.sleep(0)

def get_datasets() -> List[str]:
    return os.listdir(app.config['DATA_FOLDER'])


def get_checkpoints() -> List[str]:
    return os.listdir(app.config['CHECKPOINT_FOLDER'])

@app.route('/receiver', methods= ['POST'])
def receiver():
    return jsonify(progress=progress.value, started=started)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'GET':
        return render_template('train.html', 
                               datasets=get_datasets(), 
                               started=False,
                               cuda=app.config['CUDA'],
                               gpus=app.config['GPUS'])

    # Get arguments
    data_name, epochs, checkpoint_name = \
        request.form['dataName'], int(request.form['epochs']), request.form['checkpointName']
    gpu = request.form.get('gpu', None)
    try:
        dataset_type = request.form['datasetType']
    except:
        dataset_type = 'regression' # default

    if not checkpoint_name.endswith('.pt'):
        checkpoint_name += '.pt'

    # Create and modify args
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()

    args.data_path = os.path.join(app.config['DATA_FOLDER'], data_name)
    args.dataset_type = dataset_type
    args.epochs = epochs
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

        global progress
        process = mp.Process(target=progress_bar, args=(args, progress))
        process.start()
        global started
        started = 1
        # Run training
        run_training(args, logger)
        process.join()

        # reset globals
        started = 0
        progress = mp.Value('d', 0.0)

        # Move checkpoint
        shutil.move(os.path.join(args.save_dir, 'model_0', 'model.pt'),
                    os.path.join(app.config['CHECKPOINT_FOLDER'], checkpoint_name))

    return render_template('train.html',
                           datasets=get_datasets(),
                           cuda=app.config['CUDA'],
                           gpus=app.config['GPUS'],
                           trained=True)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html',
                               checkpoints=get_checkpoints(),
                               cuda=app.config['CUDA'],
                               gpus=app.config['GPUS'])

    # Get arguments
    checkpoint_name = request.form['checkpointName']

    try:
        smiles = request.form('smiles')
        smiles = smiles.split()
    except:
        # Upload data file
        data = request.files['data']
        data_name = secure_filename(data.filename)
        data_path = os.path.join(app.config['TEMP_FOLDER'], data_name)
        data.save(data_path)
        smiles = []
        with open(data_path, 'r') as f:
            header = f.readline()
            if 'smiles' not in header:  # in which case there's no header, and first line is actually just the first smiles
                smiles.append(header.strip())
            for line in f:
                smiles.append(line.strip())

    checkpoint_path = os.path.join(app.config['CHECKPOINT_FOLDER'], checkpoint_name)
    task_names = load_task_names(checkpoint_path)
    gpu = request.form.get('gpu', None)

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

    # Run prediction
    preds = make_predictions(args, smiles=smiles)

    return render_template('predict.html',
                           checkpoints=get_checkpoints(),
                           cuda=app.config['CUDA'],
                           gpus=app.config['GPUS'],
                           predicted=True,
                           smiles=smiles,
                           num_smiles=min(10, len(smiles)),
                           show_more=max(0, len(smiles)-10),
                           task_names=task_names,
                           num_tasks=len(task_names),
                           preds=preds)


@app.route('/download_predictions')
def download_predictions():
    return send_from_directory(app.config['TEMP_FOLDER'], app.config['PREDICTIONS_FILENAME'], as_attachment=True)


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
