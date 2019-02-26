"""
Sets the config parameters for the flask app object.
These are accessible in a dictionary, with each line defining a key.
"""

import os
import torch
from tempfile import TemporaryDirectory

ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'app/web_data')
CHECKPOINT_FOLDER = os.path.join(ROOT_FOLDER, 'app/web_checkpoints')
TEMP_FOLDER = os.path.join(ROOT_FOLDER, TemporaryDirectory().name)
SMILES_FILENAME = 'smiles.csv'
PREDICTIONS_FILENAME = 'predictions.csv'
CUDA = torch.cuda.is_available()
GPUS = list(range(torch.cuda.device_count()))