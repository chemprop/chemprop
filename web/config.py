import torch
from tempfile import TemporaryDirectory

DATA_FOLDER = 'app/web_data'
CHECKPOINT_FOLDER = 'app/web_checkpoints'
TEMP_FOLDER = TemporaryDirectory().name
SMILES_FILENAME = 'smiles.csv'
PREDICTIONS_FILENAME = 'predictions.csv'
CUDA = torch.cuda.is_available()
GPUS = list(range(torch.cuda.device_count()))