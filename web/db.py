import os
import shutil
import sqlite3
from flask import current_app, g
from flask.cli import with_appcontext

def init_app(app):
    app.teardown_appcontext(close_db)


# Database setup.
def init_db():
    """
    Executes schema.sql to initialize the database.
    This will wipe existing tables.
    """
    shutil.rmtree('web_checkpoints')
    shutil.rmtree('web_data')
    os.makedirs('web_checkpoints')
    os.makedirs('web_data')
    db = get_db()

    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))

# Database access.
def get_db():
    """
    Returns a representation of the database.
    Forms a connection with the database if one is not found.
    """
    if 'db' not in g:
        g.db = sqlite3.connect(
            'chemprop.sqlite3',
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row

    return g.db

def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv


def close_db(e=None):
    """
    Closes the connection to the database. Called after every request.
    """
    db = g.pop('db', None)

    if db is not None:
        db.close()

# Table Specific Functions
def get_all_users():
    rows = query_db("SELECT * FROM user")

    if rows:
        return {row[0]: {"username": row[1], "preferences": row[2]} for row in rows}
    else:
        return {}

def insert_user(username):
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

def get_ckpts(user_id):
    if not user_id:
        user_id = 0

    return query_db('SELECT * FROM ckpt WHERE associated_user = ' + str(user_id))

def insert_ckpt(ckpt_name, associated_user, model_class, num_epochs):
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

def delete_ckpt(ckpt_id):
    db = get_db()
    cur = db.execute('DELETE FROM ckpt WHERE id =' + str(ckpt_id))
    db.commit()
    cur.close()

def get_datasets(user_id):
    if not user_id:
        user_id = 0

    return query_db('SELECT * FROM dataset WHERE associated_user = ' + str(user_id))

def insert_dataset(dataset_name, associated_user, dataset_class):
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

def delete_dataset(dataset_id):
    db = get_db()
    cur = db.execute('DELETE FROM dataset WHERE id =' + str(dataset_id))
    db.commit()
    cur.close()