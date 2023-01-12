#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("CHANGELOG.rst") as history_file:
    history = history_file.read()

requirements = []

test_requirements = [
    "pytest>=3",
]

setup(
    author="Olivier Binette",
    author_email="olivier.binette@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    description=" An End-to-End Evaluation Framework for Entity Resolution Systems.",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="er_evaluation",
    name="er_evaluation",
    packages=find_packages(include=["er_evaluation", "er_evaluation.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/OlivierBinette/er_evaluation",
    version="1.2.0",
    zip_safe=False,
    install_requires=[
            "pandas",
            "numpy",
            "scipy",
            "plotly",
            "igraph",
            "scikit-learn",
        ],
)
