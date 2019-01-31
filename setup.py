from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if not line.startswith('git+http')]

setup(
    name='chemprop',
    version='0.0.1',
    author='Kyle Swanson, Kevin Yang, Wengong Jin',
    author_email='swansonk@mit.edu, yangk@mit.edu, wengong@csail.mit.edu',
    description='Molecular Property Prediction with Message Passing Neural Networks',
    url='https://github.com/swansonk14/chemprop',
    license='MIT',
    packages=find_packages(),
    install_requires=requirements,
    keywords=['chemistry', 'machine learning', 'property prediction', 'message passing neural network']
)
