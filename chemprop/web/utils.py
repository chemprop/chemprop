"""Contains utility functions for the Flask web app."""

import os
import shutil

from flask import Flask


def set_root_folder(app: Flask, root_folder: str = None, create_folders: bool = True) -> None:
    """
    Sets the root folder for the config along with subfolders like the data and checkpoint folders.

    :param app: Flask app.
    :param root_folder: Path to the root folder. If None, the current root folders is unchanged.
    :param create_folders: Whether to create the root folder and subfolders.
    """
    # Set root folder and subfolders
    if root_folder is not None:
        app.config['ROOT_FOLDER'] = root_folder
        app.config['DATA_FOLDER'] = os.path.join(app.config['ROOT_FOLDER'], 'app/web_data')
        app.config['CHECKPOINT_FOLDER'] = os.path.join(app.config['ROOT_FOLDER'], 'app/web_checkpoints')
        app.config['TEMP_FOLDER'] = os.path.join(app.config['ROOT_FOLDER'], 'app/temp')
        app.config['DB_PATH'] = os.path.join(app.config['ROOT_FOLDER'], app.config['DB_FILENAME'])

    # Create folders
    if create_folders:
        if not os.access(os.path.dirname(app.config['ROOT_FOLDER']), os.W_OK):
            raise ValueError(f'You do not have write permissions on the root_folder: {app.config["ROOT_FOLDER"]}\n'
                             f'Please specify a different root_folder while starting the web app.')

        for folder_name in ['ROOT_FOLDER', 'DATA_FOLDER', 'CHECKPOINT_FOLDER', 'TEMP_FOLDER']:
            os.makedirs(app.config[folder_name], exist_ok=True)


def clear_temp_folder(app: Flask) -> None:
    """Clears the temporary folder."""
    shutil.rmtree(app.config['TEMP_FOLDER'])
    os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)
