#!/usr/bin/env python3
from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="er_evaluation",
        version="0.0.1",
        author="Olivier Binette",
        author_email="olivier.binette@gmail.com",
        description="",
        url="https://github.com/OlivierBinette/er-evaluation",
        include_package_data=True,
        packages=["er_evaluation"],
        install_requires=[
            "pandas",
            "numpy",
            "scipy",
        ],
    )
