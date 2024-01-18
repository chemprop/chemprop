from datetime import datetime
import os
from pathlib import Path

CKPT_DIR = Path(os.getenv("CHEMPROP_CKPT_DIR", "model_checkpoints"))
LOG_DIR = Path(os.getenv("CHEMPROP_LOG_DIR", "logs/chemprop"))
NOW = datetime.now().isoformat("T", "seconds")
NOW = datetime.datetime.strptime(NOW, "%Y-%m-%dT%H-%M-%S")
