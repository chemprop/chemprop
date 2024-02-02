from datetime import datetime
import logging
import os
from pathlib import Path

LOG_DIR = Path(os.getenv("CHEMPROP_LOG_DIR", "chemprop_logs"))
LOG_LEVELS = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
NOW = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
