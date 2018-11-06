from setuptools import setup

setup(
    name='chemprop',
    packages=['chemprop'],
    version='1.0',
    description='Molecular Property Prediction with Message Passing Neural Networks',
    author='Wengong Jin, Kyle Swanson, Kevin Yang',
    author_email='wengong@csail.mit.edu, swansonk@mit.edu, yangk@mit.edu',
    url='https://github.com/wengong-jin/chemprop',
    license='MIT',
    install_requires=['hpbandster', 'matplotlib', 'numpy', 'scikit-learn', 'scipy', 'tensorboardX', 'tqdm'],
    keywords=['chemistry', 'machine learning', 'property prediction', 'message passing neural network'],
)
