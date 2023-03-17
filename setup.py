#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("CHANGELOG.rst") as history_file:
    history = history_file.read()

requirements = (
    [
        "pandas",
        "numpy",
        "scipy",
        "plotly",
        "igraph",
        "scikit-learn",
        "pyarrow",
    ],
)

test_requirements = [
    "pytest>=3",
    "testbook",
    "jupyter",
    "pyhamcrest",
]

setup(
    author="Olivier Binette",
    author_email="olivier.binette@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    description="An End-to-End Evaluation Framework for Entity Resolution Systems.",
    install_requires=requirements,
    license="GNU Affero General Public License v3",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="er_evaluation",
    name="ER-Evaluation",
    packages=find_packages(include=["er_evaluation", "er_evaluation.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/OlivierBinette/er_evaluation",
    version="1.2.0",
    zip_safe=False,
)
