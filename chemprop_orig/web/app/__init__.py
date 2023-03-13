"""Runs the web interface version of chemprop, allowing for training and predicting in a web browser."""
import os

from flask import Flask

from chemprop.web.utils import set_root_folder


app = Flask(__name__)
app.config.from_object('chemprop.web.config')
set_root_folder(
    app=app,
    root_folder=os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    create_folders=False
)

from chemprop.web.app import views
