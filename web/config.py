import os
import torch
from tempfile import TemporaryDirectory

ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = 'app/web_data'
CHECKPOINT_FOLDER = 'app/web_checkpoints'
TEMP_FOLDER = TemporaryDirectory().name
SMILES_FILENAME = 'smiles.csv'
PREDICTIONS_FILENAME = 'predictions.csv'
CUDA = torch.cuda.is_available()
GPUS = list(range(torch.cuda.device_count()))