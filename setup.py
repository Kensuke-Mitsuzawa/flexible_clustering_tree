#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import unittest
import codecs

name = 'flexible_clustering_tree'
version = '0.16'
description='easy interface for ensemble clustering'
author = 'Kensuke Mitsuzawa'
author_email = 'kensuke.mit@gmail.com'
url = 'https://github.com/Kensuke-Mitsuzawa/flexible_clustering_tree'
license_name = 'MIT'

install_requires = [
    'sqlitedict',
    'scipy',
    'numpy',
    'scikit-learn',
    'hdbscan',
    'pyexcelerate',
    'jinja2',
    'pypandoc'
]

dependency_links = []


def my_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
    long_description = codecs.open('README.md', 'r', 'utf-8').read()


classifiers = [
    "Development Status :: 5 - Production/Stable",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python",
    "Natural Language :: Japanese",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3.5"
]

setup(
    name=name,
    version=version,
    description=description,
    author=author,
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email=author_email,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=url,
    packages=find_packages(),
    test_suite='setup.my_test_suite',
    include_package_data=True,
    package_data={
        'flexible_clustering_tree': [
            'resources/tree_template.html',
        ]
    }
)
