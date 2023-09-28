from datetime import datetime
import os
from pathlib import Path

CHK_DIR = Path(os.getenv("CHEMPROP_CHK_DIR", "model_checkpoints"))
LOG_DIR = Path(os.getenv("CHEMPROP_LOG_DIR", "logs/chemprop"))
NOW = datetime.now().isoformat("T", "seconds")
