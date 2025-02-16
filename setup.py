from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

from setuptools import setup, find_packages

install_requires = [
    "tqdm>=4.48.2",
]

setup_requires = []

extras_require = {"hyperopt": ["hyperopt==0.2.5"]}

classifiers = ["License :: OSI Approved :: MIT License"]

long_description = (
    "RecBole is developed based on Python and PyTorch for reproducing and developing "
    "recommendation algorithms in a unified, comprehensive, and efficient framework for "
    "research purposes. The library provides a flexible and extensible platform for benchmarking "
    "and enhancing recommendation models.\n\n"

    "In the first version, our library includes 53 recommendation algorithms, covering four "
    "major categories:\n"
    "- General Recommendation\n"
    "- Sequential Recommendation\n"
    "- Context-aware Recommendation\n"
    "- Knowledge-based Recommendation\n\n"

    "In this extension 2025.17.2, several new features and improvements have been implemented to enhance the "
    "capabilities of the library:\n"
    "- Leave-k-out split: A more flexible data splitting strategy for training and evaluation.\n"
    "- Implementation of AsymKNN and ALS algorithms for collaborative filtering-based recommendations.\n"
    "- User-based evaluation for specific topological metrics, enabling a deeper analysis of graph-based "
    "recommendation models.\n\n"

    "These additions improve the evaluation process and expand the range of algorithms supported, "
    "allowing for more comprehensive benchmarking and experimentation.\n\n"

    "For more information, visit the RecBole homepage: https://recbole.io"
)

# Readthedocs requires Sphinx extensions to be specified as part of
# install_requires in order to build properly.
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if on_rtd:
    install_requires.extend(setup_requires)

setup(
    name="recbole",
    version="2025.17.2",  # please remember to edit recbole/__init__.py in response, once updating the version
    description="A unified, comprehensive and efficient recommendation library used for my thesis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mkhe93/RecBole/tree/mkhe/thesis",
    author="Markus Hoefling",
    author_email="markus.hoefling01@gmail.com",
    packages=[package for package in find_packages() if package.startswith("recbole")],
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    zip_safe=False,
    classifiers=classifiers,
)
