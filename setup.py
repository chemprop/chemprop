from setuptools import find_packages, setup

__version__ = "1.5.2"

# Load README
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='chemprop',
    author='Kyle Swanson, Kevin Yang, Wengong Jin, Lior Hirschfeld, Allison Tam',
    author_email='chemprop@mit.edu',
    description='Molecular Property Prediction with Message Passing Neural Networks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/chemprop/chemprop',
    download_url=f'https://github.com/chemprop/chemprop/v_{__version__}.tar.gz',
    project_urls={
        'Documentation': 'https://chemprop.readthedocs.io/en/latest/',
        'Source': 'https://github.com/chemprop/chemprop',
        'PyPi': 'https://pypi.org/project/chemprop/',
        'Demo': 'http://chemprop.csail.mit.edu/',
    },
    license='MIT',
    packages=find_packages(),
    package_data={'chemprop': ['py.typed']},
    entry_points={
        'console_scripts': [
            'chemprop_train=chemprop.train:chemprop_train',
            'chemprop_predict=chemprop.train:chemprop_predict',
            'chemprop_fingerprint=chemprop.train:chemprop_fingerprint',
            'chemprop_hyperopt=chemprop.hyperparameter_optimization:chemprop_hyperopt',
            'chemprop_interpret=chemprop.interpret:chemprop_interpret',
            'chemprop_web=chemprop.web.run:chemprop_web',
            'sklearn_train=chemprop.sklearn_train:sklearn_train',
            'sklearn_predict=chemprop.sklearn_predict:sklearn_predict',
        ]
    },
    install_requires=[
        'flask>=1.1.2',
        'hyperopt>=0.2.3',
        'matplotlib>=3.1.3',
        'numpy>=1.18.1',
        'pandas>=1.0.3',
        'pandas-flavor>=0.2.0',
        'scikit-learn>=0.22.2.post1',
        'scipy>=1.5.2',
        'sphinx>=3.1.2',
        'tensorboardX>=2.0',
        'torch>=1.5.2',
        'tqdm>=4.45.0',
        'typed-argument-parser>=1.6.1'
    ],
    extras_require={
        'test': [
            'pytest>=6.2.2',
            'parameterized>=0.8.1'
        ]
    },
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
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
