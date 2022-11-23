#!/usr/bin/env python3
from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

if __name__ == "__main__":
    setup(
        name="er_evaluation",
        version="1.0.0.post0",
        author="Olivier Binette",
        author_email="olivier.binette@gmail.com",
        description="An end-to-end evaluation framework for entity resolution systems.",
        long_description=long_description,
        long_description_content_type='text/markdown',
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
