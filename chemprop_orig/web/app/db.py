"""Defines a number of database helper functions."""

import os
import shutil
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

from flask import current_app, Flask, g

from chemprop.web.app import app


DB_PATH = 'chemprop.sqlite3'


def init_app(app: Flask):
    global DB_PATH

    app.teardown_appcontext(close_db)
    DB_PATH = app.config['DB_PATH']


def init_db():
    """
    Initializes the database by running schema.sql.
    This will wipe existing tables and the corresponding files.
    """
    shutil.rmtree(app.config['DATA_FOLDER'])
    os.makedirs(app.config['DATA_FOLDER'])

    shutil.rmtree(app.config['CHECKPOINT_FOLDER'])
    os.makedirs(app.config['CHECKPOINT_FOLDER'])

    db = get_db()

    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))


def get_db():
    """
    Connects to the database.
    Returns a database object that can be queried.
    """
    if 'db' not in g:
        g.db = sqlite3.connect(
            DB_PATH,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row

    return g.db


def query_db(query: str, args: Tuple[Any] = (), one: bool = False) -> Union[Optional[sqlite3.Row], List[sqlite3.Row]]:
    """
    Helper function to allow for easy queries.

    :param query: The query to be executed.
    :param args: The arguments to be passed into the query.
    :param one: Whether the query should return all results or just the first.
    :return The results of the query.
    """
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv


def close_db(e: Optional[Any] = None):
    """
    Closes the connection to the database. Called after every request.

    :param e: Error object from last call.
    """
    db = g.pop('db', None)

    if db is not None:
        db.close()


# Table Specific Functions
def get_all_users() -> Dict[int, Dict[str, Any]]:
    """
    Returns all users.

    :return A dictionary of users with their ids as keys.
    """
    rows = query_db("SELECT * FROM user")

    return {row['id']: {"username": row['username'], "preferences": row['preferences']} for row in rows} if rows else {}


def insert_user(username: str) -> Tuple[int, str]:
    """
    Inserts a new user. If the desired username is already taken,
    appends integers incrementally until an open name is found.

    :param username: The desired username for the new user.
    :return A tuple containing the id and name of the new user.
    """
    db = get_db()

    new_user_id = None
    count = 0
    while new_user_id is None:
        temp_name = username

        if count != 0:
            temp_name += str(count)
        try:
            cur = db.execute('INSERT INTO user (username) VALUES (?)', [temp_name])
            new_user_id = cur.lastrowid
        except sqlite3.IntegrityError:
            count += 1

    db.commit()
    cur.close()

    return new_user_id, temp_name


def get_ckpts(user_id: int) -> List[sqlite3.Row]:
    """
    Returns the checkpoints associated with the given user.
    If no user_id is provided, return the checkpoints associated
    with the default user.

    :param user_id: The id of the user whose checkpoints are returned.
    :return A list of checkpoints.
    """
    if not user_id:
        user_id = app.config['DEFAULT_USER_ID']

    return query_db(f'SELECT * FROM ckpt WHERE associated_user = {user_id}')


def insert_ckpt(ckpt_name: str,
                associated_user: str,
                model_class: str,
                num_epochs: int,
                ensemble_size: int,
                training_size: int) -> Tuple[int, str]:
    """
    Inserts a new checkpoint. If the desired name is already taken,
    appends integers incrementally until an open name is found.

    :param ckpt_name: The desired name for the new checkpoint.
    :param associated_user: The user that should be associated with the new checkpoint.
    :param model_class: The class of the new checkpoint.
    :param num_epochs: The number of epochs the new checkpoint will run for.
    :param ensemble_size: The number of models included in the ensemble.
    :param training_size: The number of molecules used for training.
    :return A tuple containing the id and name of the new checkpoint.
    """
    db = get_db()

    new_ckpt_id = None
    count = 0
    while new_ckpt_id is None:
        temp_name = ckpt_name

        if count != 0:
            temp_name += str(count)
        try:
            cur = db.execute('INSERT INTO ckpt '
                             '(ckpt_name, associated_user, class, epochs, ensemble_size, training_size) '
                             'VALUES (?, ?, ?, ?, ?, ?)',
                             [temp_name, associated_user, model_class, num_epochs, ensemble_size, training_size])
            new_ckpt_id = cur.lastrowid
        except sqlite3.IntegrityError:
            count += 1
            continue

    db.commit()
    cur.close()

    return new_ckpt_id, temp_name


def delete_ckpt(ckpt_id: int):
    """
    Removes the checkpoint with the specified id from the database,
    associated model columns, and the corresponding files.

    :param ckpt_id: The id of the checkpoint to be deleted.
    """
    rows = query_db(f'SELECT * FROM model WHERE associated_ckpt = {ckpt_id}')

    for row in rows:
        os.remove(os.path.join(app.config['CHECKPOINT_FOLDER'], f'{row["id"]}.pt'))

    db = get_db()
    cur = db.cursor()
    db.execute(f'DELETE FROM ckpt WHERE id = {ckpt_id}')
    db.execute(f'DELETE FROM model WHERE associated_ckpt = {ckpt_id}')
    db.commit()
    cur.close()


def get_models(ckpt_id: int) -> List[sqlite3.Row]:
    """
    Returns the models associated with the given ckpt.

    :param ckpt_id: The id of the ckpt whose component models are returned.
    :return A list of models.
    """
    return query_db(f'SELECT * FROM model WHERE associated_ckpt = {ckpt_id}')


def insert_model(ckpt_id: int) -> str:
    """
    Inserts a new model.

    :param ckpt_id: The id of the checkpoint this model should be associated with.
    :return: The id of the new model.
    """
    db = get_db()
    cur = db.execute('INSERT INTO model (associated_ckpt) VALUES (?)', [ckpt_id])
    new_model_id = cur.lastrowid
    db.commit()
    cur.close()

    return new_model_id


def get_datasets(user_id: int) -> List[sqlite3.Row]:
    """
    Returns the datasets associated with the given user.
    If no user_id is provided, return the datasets associated
    with the default user.

    :param user_id: The id of the user whose datasets are returned.
    :return A list of datasets.
    """
    if not user_id:
        user_id = app.config['DEFAULT_USER_ID']

    return query_db(f'SELECT * FROM dataset WHERE associated_user = {user_id}')


def insert_dataset(dataset_name: str,
                   associated_user: str,
                   dataset_class: str) -> Tuple[int, str]:
    """
    Inserts a new dataset. If the desired name is already taken,
    appends integers incrementally until an open name is found.

    :param dataset_name: The desired name for the new dataset.
    :param associated_user: The user to be associated with the new dataset.
    :param dataset_class: The class of the new dataset.
    :return A tuple containing the id and name of the new dataset.
    """
    db = get_db()

    new_dataset_id = None
    count = 0
    while new_dataset_id is None:
        temp_name = dataset_name

        if count != 0:
            temp_name += str(count)
        try:
            cur = db.execute('INSERT INTO dataset (dataset_name, associated_user, class) VALUES (?, ?, ?)',
                             [temp_name, associated_user, dataset_class])
            new_dataset_id = cur.lastrowid
        except sqlite3.IntegrityError:
            count += 1
            continue

    db.commit()
    cur.close()

    return new_dataset_id, temp_name


def delete_dataset(dataset_id: int):
    """
    Removes the dataset with the specified id from the database,
    and deletes the corresponding file.

    :param dataset_id: The id of the dataset to be deleted.
    """
    db = get_db()
    cur = db.execute(f'DELETE FROM dataset WHERE id = {dataset_id}')
    db.commit()
    cur.close()
