"""
Runs the web interface version of Chemprop.
Designed to be used for production only, along with Gunicorn.
"""
from web.app import app, db
from web.utils import clear_temp_folder, set_root_folder


def build_app(*args, **kwargs):
    # Set up root folder and subfolders
    set_root_folder(
        app=app,
        root_folder=kwargs.get('root_folder', None),
        create_folders=True
    )
    clear_temp_folder(app=app)

    db.init_app(app)
    if 'init_db' in kwargs:
        with app.app_context():
            db.init_db()
            print("-- INITIALIZED DATABASE --")

    app.config['DEMO'] = kwargs.get('demo', False)

    return app
