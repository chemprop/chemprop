"""Runs the web interface version of chemprop, allowing for training and predicting in a web browser."""
import os
from flask import Flask

app = Flask(__name__)
app.config.from_object('chemprop.web.config')

os.makedirs(app.config['CHECKPOINT_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)

from chemprop.web.app import views
