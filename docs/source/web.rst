.. _web:

Web Interface
=============

Overview
--------

For those less familiar with the command line, Chemprop also includes a web interface which allows for basic training and predicting. An example of the website (in demo mode with training disabled) is available here: `<chemprop.csail.mit.edu>`_.

.. image:: _static/images/web_train.png
   :alt: Training with our web interface

.. image:: _static/images/web_predict.png
   :alt: Predicting with our web interface

You can start the web interface on your local machine in two ways. Flask is used for development mode while gunicorn is used for production mode.

Flask
-----

Run :code:`chemprop_web` (or optionally :code:`python web.py` if installed from source) and then navigate to `localhost:5000 <http://localhost:5000>`_ in a web browser.

Gunicorn
--------

Gunicorn is only available for a UNIX environment, meaning it will not work on Windows. It is not installed by default with the rest of Chemprop, so first run:

.. code-block::

   pip install gunicorn

Next, navigate to :code:`chemprop/web` and run :code:`gunicorn --bind {host}:{port} 'wsgi:build_app()'`. This will start the site in production mode.

   * To run this server in the background, add the :code:`--daemon` flag.
   * Arguments including :code:`init_db` and :code:`demo` can be passed with this pattern: :code:`'wsgi:build_app(init_db=True, demo=True)'`
   * Gunicorn documentation can be found [here](http://docs.gunicorn.org/en/stable/index.html).
