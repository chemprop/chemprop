"""Defines a number of database helper functions."""

import os
import shutil
import sqlite3

from flask import current_app, g
from typing import Tuple
from app import app 

def init_app(app: Flask):
    app.teardown_appcontext(close_db)

# Database setup.
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

# Database access.
def get_db():
    """
    Connects to the database.
    Returns a database object that can be queried.
    """
    if 'db' not in g:
        g.db = sqlite3.connect(
            'chemprop.sqlite3',
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row

    return g.db

def query_db(query: str, args = (), one: bool = False):
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


def close_db(e = None):
    """
    Closes the connection to the database. Called after every request.
    """
    db = g.pop('db', None)

    if db is not None:
        db.close()

# Table Specific Functions
def get_all_users():
    """
    Returns all users.

    :return A dictionary of users with their ids as keys.
    """
    rows = query_db("SELECT * FROM user")

    if rows:
        return {row[0]: {"username": row[1], "preferences": row[2]} for row in rows}
    else:
        return {}

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
    while new_user_id == None:
        temp_name = username
        
        if count != 0:
            temp_name += str(count)
        try:
            cur = db.execute('INSERT INTO user (username) VALUES (?)', [temp_name])
            new_user_id = cur.lastrowid
        except sqlite3.IntegrityError:
            count += 1
            continue
    
    db.commit()
    cur.close()

    return new_user_id, temp_name

def get_ckpts(user_id: int):
    """
    Returns the checkpoints associated with the given user.
    If no user_id is provided, return the checkpoints associated
    with the default user.

    :param user_id: The id of the user whose checkpoints are returned.
    :return A list of checkpoints.
    """
    if not user_id:
        user_id = 0

    return query_db(f'SELECT * FROM ckpt WHERE associated_user = {user_id}')

def insert_ckpt(ckpt_name: str, 
                associated_user: str, 
                model_class: str, 
                num_epochs: int) -> Tuple[int, str]:
    """
    Inserts a new checkpoint. If the desired name is already taken,  
    appends integers incrementally until an open name is found.   

    :param ckpt_name: The desired name for the new checkpoint.
    :param associated_user: The user that should be associated with the new checkpoint.
    :param model_class: The class of the new checkpoint.
    :param num_epochs: The number of epochs the new checkpoint will run for.
    :return A tuple containing the id and name of the new checkpoint.   
    """
    db = get_db()

    new_ckpt_id = None
    count = 0
    while new_ckpt_id == None:
        temp_name = ckpt_name

        if count != 0:
            temp_name += str(count)
        try:
            cur = db.execute('INSERT INTO ckpt (ckpt_name, associated_user, class, epochs) VALUES (?, ?, ?, ?)', 
                                [temp_name, associated_user, model_class, num_epochs])
            new_ckpt_id = cur.lastrowid
        except sqlite3.IntegrityError as e:
            count += 1
            continue
    
    db.commit()
    cur.close()

    return new_ckpt_id, temp_name

def delete_ckpt(ckpt_id: int):
    """
    Removes the checkpoint with the specified id from the database,
    and deletes the corresponding file.

    :param ckpt_id: The id of the checkpoint to be deleted.
    """
    db = get_db()
    cur = db.execute(f'DELETE FROM ckpt WHERE id = {ckpt_id}')
    db.commit()
    cur.close()

def get_datasets(user_id: int):
    """
    Returns the datasets associated with the given user.
    If no user_id is provided, return the datasets associated
    with the default user.

    :param user_id: The id of the user whose datasets are returned.
    :return A list of datasets.
    """
    if not user_id:
        user_id = DEFAULT_USER_ID

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
    while new_dataset_id == None:
        temp_name = dataset_name

        if count != 0:
            temp_name += str(count)
        try:
            cur = db.execute('INSERT INTO dataset (dataset_name, associated_user, class) VALUES (?, ?, ?)', 
                                [temp_name, associated_user, dataset_class])
            new_dataset_id = cur.lastrowid
        except sqlite3.IntegrityError as e:
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
