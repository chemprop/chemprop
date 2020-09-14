"""
Sets the config parameters for the flask app object.
These are accessible in a dictionary, with each line defining a key.
"""

import os
from tempfile import TemporaryDirectory

import torch

_TEMP_FOLDER_OBJECT = TemporaryDirectory()

DEFAULT_USER_ID = 1
if os.access(os.path.dirname(os.path.realpath(__file__)), os.W_OK):
    ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))
elif os.access(os.getcwd(), os.W_OK):
    ROOT_FOLDER = os.path.join(os.getcwd(), "chemprop_web_app")
else :
    raise ValueError("Failed to find a writable ROOT_FOLDER for web app data and checkpoints.")
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'app/web_data')
CHECKPOINT_FOLDER = os.path.join(ROOT_FOLDER, 'app/web_checkpoints')
TEMP_FOLDER = os.path.join(ROOT_FOLDER, _TEMP_FOLDER_OBJECT.name)
SMILES_FILENAME = 'smiles.csv'
PREDICTIONS_FILENAME = 'predictions.csv'
DB_FILENAME = 'chemprop.sqlite3'
CUDA = torch.cuda.is_available()
GPUS = list(range(torch.cuda.device_count()))
