#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="models",
    version=0.1,
    description="Various circular regression modules",
    author="Sergey Prokudin",
    author_email="sergey.prokudin@gmail.com",
    packages=["models"],
)


setup(
    name="utils",
    version=0.1,
    description="Misc utils (angle conversion, datasets preprocessing, etc.)",
    author="Sergey Prokudin",
    author_email="sergey.prokudin@gmail.com",
    packages=["utils"],
)

setup(
  name = 'von_mises_fisher',
  packages = find_packages(),
  version = '0.1',
  description = 'Model your data with the n-dimensional Von Mises-Fisher distribution',
  author = 'Louis Cialdella',
  author_email = 'louiscialdella@gmail.com',
  classifiers = ['Programming Language :: Python :: 3 :: Only'],
  url = 'https://github.com/lmc2179/von_mises_fisher',
  keywords = ['unit sphere', 'probability', 'distribution', 'von mises', 'fisher']
)