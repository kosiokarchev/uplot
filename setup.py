from setuptools import setup, find_packages

setup(
    name='uplot',
    description='Utilities for plotting.',
    version='0.1',
    packages=['uplot'],
    install_requires=[
        'frozendict', 'more_itertools',
        'numpy', 'scipy', 'matplotlib',
    ],
)
