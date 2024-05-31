#!/usr/bin/env python
"""
Author: Wenyu Ouyang
Date: 2024-05-31 14:21:54
LastEditTime: 2024-05-31 15:44:52
LastEditors: Wenyu Ouyang
Description: Tests for `hydroevaluate` package.
FilePath: \hydroevaluate\tests\test_hydroevaluate.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import pytest


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
