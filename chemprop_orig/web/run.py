"""
Runs the web interface version of Chemprop.
This allows for training and predicting in a web browser.
"""

import os

from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)

from chemprop.web.app import app, db
from chemprop.web.utils import clear_temp_folder, set_root_folder


class WebArgs(Tap):
    host: str = '127.0.0.1'  # Host IP address
    port: int = 5000  # Port
    debug: bool = False  # Whether to run in debug mode
    demo: bool = False  # Display only demo features
    initdb: bool = False  # Initialize Database
    root_folder: str = None  # Root folder where web data and checkpoints will be saved (defaults to chemprop/web/app)


def run_web(args: WebArgs) -> None:
    app.config['DEMO'] = args.demo

    # Set up root folder and subfolders
    set_root_folder(
        app=app,
        root_folder=args.root_folder,
        create_folders=True
    )
    clear_temp_folder(app=app)

    db.init_app(app)

    if args.initdb or not os.path.isfile(app.config['DB_PATH']):
        with app.app_context():
            db.init_db()
            print("-- INITIALIZED DATABASE --")

    app.run(host=args.host, port=args.port, debug=args.debug)


def chemprop_web() -> None:
    """Runs the Chemprop website locally.

    This is the entry point for the command line command :code:`chemprop_web`.
    """
    run_web(args=WebArgs().parse_args())
