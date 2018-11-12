from setuptools import find_packages, setup

setup(
    name='chemprop',
    version='1.0',
    author='Wengong Jin, Kyle Swanson, Kevin Yang',
    author_email='wengong@csail.mit.edu, swansonk@mit.edu, yangk@mit.edu',
    description='Molecular Property Prediction with Message Passing Neural Networks',
    url='https://github.com/wengong-jin/chemprop',
    license='MIT',
    packages=find_packages(),
    install_requires=['hpbandster', 'matplotlib', 'numpy', 'scikit-learn', 'scipy', 'tensorboardX', 'tqdm'],
    keywords=['chemistry', 'machine learning', 'property prediction', 'message passing neural network'],
)
