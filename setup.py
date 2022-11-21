#!/usr/bin/env python3
from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="er_evaluation",
        version="1.0.0",
        author="Olivier Binette",
        author_email="olivier.binette@gmail.com",
        description="An end-to-end evaluation framework for entity resolution systems.",
        license_files=("LICENSE.txt",),
        url="https://github.com/OlivierBinette/er-evaluation",
        include_package_data=True,
        packages=["er_evaluation"],
        install_requires=[
            "pandas",
            "numpy",
            "scipy",
            "plotly",
            "igraph",
        ],
    )
