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