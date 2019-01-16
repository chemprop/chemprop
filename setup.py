from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if not line.startswith('git+http')]

setup(
    name='chemprop',
    version='1.0',
    author='Wengong Jin, Kyle Swanson, Kevin Yang',
    author_email='wengong@csail.mit.edu, swansonk@mit.edu, yangk@mit.edu',
    description='Molecular Property Prediction with Message Passing Neural Networks',
    url='https://github.com/wengong-jin/chemprop',
    license='MIT',
    packages=find_packages(),
    install_requires=requirements,
    keywords=['chemistry', 'machine learning', 'property prediction', 'message passing neural network'],
)
