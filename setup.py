from setuptools import find_packages, setup

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='chemprop',
    version='0.0.2',
    author='Kyle Swanson, Kevin Yang, Wengong Jin',
    author_email='swansonk.14@gmail.com, yangk@mit.edu, wengong@csail.mit.edu',
    description='Molecular Property Prediction with Message Passing Neural Networks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/chemprop/chemprop',
    license='MIT',
    packages=find_packages(),
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
