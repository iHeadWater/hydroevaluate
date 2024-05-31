#!/usr/bin/env python
"""
Author: Wenyu Ouyang
Date: 2024-05-31 14:21:54
LastEditTime: 2024-05-31 14:24:52
LastEditors: Wenyu Ouyang
Description: the setup file for hydroevaluate
FilePath: \hydroevaluate\setup.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import io
from os import path as op
from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

here = op.abspath(op.dirname(__file__))

# get the dependencies and installs
with io.open(op.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")

install_requires = [x.strip() for x in all_reqs if "git+" not in x]
dependency_links = [x.strip().replace("git+", "") for x in all_reqs if "git+" not in x]

requirements = [ ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Wenyu Ouyang",
    author_email='wenyuouyang@outlook.com',
    python_requires='>=3.10',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    description="A real evaluation tool for hydrological forecast",
    install_requires=install_requires,
    dependency_links=dependency_links,
    license="BSD license",
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='hydroevaluate',
    name='hydroevaluate',
    packages=find_packages(include=['hydroevaluate', 'hydroevaluate.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/OuyangWenyu/hydroevaluate',
    version='0.0.1',
    zip_safe=False,
)
