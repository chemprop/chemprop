import os
from setuptools import find_packages, setup

# Load version number
__version__ = None

src_dir = os.path.abspath(os.path.dirname(__file__))
version_file = os.path.join(src_dir, 'chemprop', '_version.py')

with open(version_file, encoding='utf-8') as fd:
    exec(fd.read())

# Load README
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Load requirements
with open('requirements.txt', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() != '']


setup(
    name='chemprop',
    version=__version__,
    author='Kyle Swanson, Kevin Yang, Wengong Jin, Lior Hirschfeld',
    author_email='chemprop@mit.edu',
    description='Molecular Property Prediction with Message Passing Neural Networks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/chemprop/chemprop',
    download_url=f'https://github.com/chemprop/chemprop/v_{__version__}.tar.gz',
    project_urls={
        # 'Documentation': 'https://chemprop.readthedocs.io/en/latest/',
        'Source': 'https://github.com/chemprop/chemprop',
        'Demo': 'http://chemprop.csail.mit.edu/'
    },
    license='MIT',
    packages=find_packages(),
    package_data={'chemprop': ['py.typed']},
    entry_points={
        'console_scripts': [
            'chemprop_train=chemprop.train.cross_validate:chemprop_train',
            'chemprop_predict=chemprop.train.make_predictions:chemprop_predict',
            'chemprop_hyperopt=chemprop.hyperparameter_optimization:chemprop_hyperopt',
            'chemprop_interpret=chemprop.interpret:chemprop_interpret',
            'chemprop_web=chemprop.web.run:chemprop_web',
            'sklearn_train=chemprop.sklearn_train:sklearn_train',
            'sklearn_predict=chemprop.sklearn_predict:sklearn_predict',
        ]
    },
    install_requires=requirements,
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    keywords=[
        'chemistry',
        'machine learning',
        'property prediction',
        'message passing neural network',
        'graph neural network'
    ]
)
