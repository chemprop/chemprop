from datetime import datetime
import logging
import os
from pathlib import Path

LOG_DIR = Path(os.getenv("CHEMPROP_LOG_DIR", "chemprop_logs"))
LOG_LEVELS = {0: logging.INFO, 1: logging.DEBUG, -1: logging.WARNING, -2: logging.ERROR}
NOW = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
CHEMPROP_TRAIN_DIR = Path(os.getenv("CHEMPROP_TRAIN_DIR", "chemprop_training"))
