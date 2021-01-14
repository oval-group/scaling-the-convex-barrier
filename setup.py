import sys

from os import path
from setuptools import setup, find_packages

if sys.version_info < (3,6):
    sys.exit("Sorry, only Python >= 3.6 is supported")
here = path.abspath(path.dirname(__file__))

setup(
    name='scaling-the-convex-barrier',
    version='0.0.1',
    description='Scaling the convex barrier for piecewise-linear neural network verification',
    author='OVAL group, University of Oxford',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(),
    install_requires=['sh', 'numpy', 'torch', 'scipy', 'pandas'],
    extras_require={
        'tests': ['mypy', 'flake8'],
        'dev': ['ipython', 'ipdb']
    },
)
