#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""AUTHOR: KENNETH CHEN.

Service package.

"""

import os
import re

from setuptools import find_packages, setup


version_file_contents = open(
    os.path.join(os.getcwd(), "classificationconfidenceintervals/version.py")
).read()
version = re.compile(r"\d+\.\d+\.\d+").findall(version_file_contents)[0]
setup(
    name="classification-confidence-intervals",
    version=version,
    description="A package to calculate confidence intervals for classification "
    "positive rate, precision, NPV, and recall using a labeled sample of the population "
    "via exact & approximate Frequentist & Bayesian setups.",
    url="https://github.com/kennethschen/Classification-Confidence-Intervals",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Kenneth S. Chen",
    author_email="kennychen12@gmail.com",
    package_dir={"": "classificationconfidenceintervals"},
    packages=find_packages(where="classificationconfidenceintervals"),
    include_package_data=True,
    install_requires=["matplotlib>=3.2.2", "numpy>=1.19.0", "scipy>=1.5.0", "seaborn>=0.10.1"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
