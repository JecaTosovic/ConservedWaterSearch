"""
ConservedWaterSearch
Module for identification of conserved water molecules from MD trajectories.
"""
import sys

from setuptools import find_packages, setup

short_description = __doc__.split("\n")

needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []

try:
    with open("README.rst") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:])


setup(
    name="ConservedWaterSearch",
    version="0.1.2",
    author="Domagoj Fijan, Jelena Tosovic, Marko Jukic, Urban Bren",
    author_email="jecat_90@live.com",
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/x-rst",
    license="BSD-3-Clause",
    keywords=("simulation analysis molecular dynamics biosimulation conserved water "),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    url="https://github.com/JecaTosovic/ConservedWaterSearch",
    download_url="https://pypi.org/project/ConservedWaterSearch/",
    project_urls={
        "Homepage": "https://github.com/JecaTosovic/ConservedWaterSearch",
        "Documentation": "https://ConservedWaterSearch.readthedocs.io/",
        "Source Code": "https://github.com/JecaTosovic/ConservedWaterSearch",
        "Issue Tracker": "https://github.com/JecaTosovic/ConservedWaterSearch/issues",
    },
    python_requires=">=3.8",
    packages=find_packages(),
    include_package_data=True,
    setup_requires=[] + pytest_runner,
    # extras_require={
    #    "debug": ["matplotlib>=3.4"],
    #    "nglview": ["nglview>3.0.0"],
    #    "hdbscan": ["hdbscan>=0.8.27",]
    # },
    # },
    install_requires=[
        "hdbscan>=0.8.27",
        "numpy>=1.21",
        "scikit-learn>=1.0",
        "matplotlib>=3.4",
        "nglview>3.0.0",
    ],
    platforms=["Linux", "Mac OS-X", "Unix", "Windows"],
    zip_safe=False,
)
