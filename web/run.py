"""
Runs the web interface version of Chemprop.
This allows for training and predicting in a web browser.
"""

from argparse import ArgumentParser
import os

from app import app, db

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host IP address')
    parser.add_argument('--port', type=int, default=5000, help='Port')
    parser.add_argument('--debug', action='store_true', default=False, help='Whether to run in debug mode')
    parser.add_argument('--demo', action='store_true', default=False, help='Display only demo features')
    parser.add_argument('--initdb', action='store_true', default=False, help='Initialize Database')
    args = parser.parse_args()

    app.config['DEMO'] = args.demo

    db.init_app(app)

    if args.initdb or not os.path.isfile(app.config['DB_FILENAME']):
        with app.app_context():
            db.init_db()
            print("-- INITIALIZED DATABASE --")

    app.run(host=args.host, port=args.port, debug=args.debug)
