from datetime import datetime
import os
from pathlib import Path

LOG_DIR = Path(os.getenv("CHEMPROP_LOG_DIR", "chemprop_logs"))
NOW = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
