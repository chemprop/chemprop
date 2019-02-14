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