from datetime import datetime
import os
from pathlib import Path

LOG_DIR = Path(os.getenv("CHEMPROP_LOG_DIR", "logs/chemprop"))
NOW = datetime.now().isoformat("T", "seconds")
