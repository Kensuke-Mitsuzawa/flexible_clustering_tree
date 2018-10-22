#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import unittest

name = 'flexible_clustering_tree'
version = '0.1'
description=''
author = 'Kensuke Mitsuzawa'
author_email = 'kensuke.mit@gmail.com'
url = ''
license_name = 'MIT'

install_requires = [
    'sqlitedict',
    'scipy',
    'numpy',
    'scikit-learn',
    'hdbscan',
    'pyexcelerate',
    'jinja2'
]

dependency_links = []


def my_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite


setup(
    name=name,
    version=version,
    description=description,
    author=author,
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email=author_email,
    url=url,
    license=license_name,
    packages=find_packages(),
    test_suite='setup.my_test_suite',
    include_package_data=True,
    package_data={
        'flexible_clustering_tree': [
            'resources/tree_template.html',
        ]
    }
)
