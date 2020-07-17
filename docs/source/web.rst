Web Interface
=============


For those less familiar with the command line, we also have a web interface which allows for basic training and predicting. An example of the website (in demo mode with training disabled) is available here: `<chemprop.csail.mit.edu>`_.

You can start the web interface on your local machine in two ways:

1. Run :code:`chemprop_web` (or optionally :code:`python web.py` if installed from source) and then navigate to `localhost:5000 <http://localhost:5000>`_ in a web browser. This will start the site in development mode.
2. (Only if install from source) Navigate to :code:`chemprop/web` and run :code:`gunicorn --bind {host}:{port} 'wsgi:build_app()'`. This will start the site in production mode.

   * To run this server in the background, add the :code:`--daemon` flag.
   * Arguments including :code:`init_db` and :code:`demo` can be passed with this pattern: :code:`'wsgi:build_app(init_db=True, demo=True)'`
   * Gunicorn documentation can be found [here](http://docs.gunicorn.org/en/stable/index.html).

.. image:: _static/images/web_train.png
   :alt: Training with our web interface

.. image:: _static/images/web_predict.png
   :alt: Predicting with our web interface
