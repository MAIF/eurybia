#!/usr/bin/env python

"""The setup script."""
import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with open("README.md", encoding="utf8") as readme_file:
    long_description = readme_file.read()

# Load the package's __version__.py module as a dictionary.
version_d: dict = {}
with open(os.path.join(here, "eurybia", "__version__.py")) as f:
    exec(f.read(), version_d)

requirements = [
    "catboost>=0.22",
    "datapane==0.14.0",
    "ipywidgets>=7.4.2",
    "jinja2>=2.11.0,<3.1.0",
    "shapash>=2.0.0",
    "seaborn<=0.11.1",
    "scipy>=1.4.0",
    "jupyter",
]


setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest",
]

setup(
    name="eurybia",  # Replace with your own username
    version=version_d["__version__"],
    python_requires=">3.6, < 3.10",
    url="https://github.com/MAIF/eurybia",
    author="Nicolas Roux, Johann Martin, Thomas Bouché",
    author_email="thomas.bouche@maif.fr",
    description="Eurybia monitor model drift over time and securize model deployment with data validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    license="Apache Software License 2.0",
    keywords="eurybia",
    package_dir={
        "eurybia": "eurybia",
        "eurybia.data": "eurybia/data",
        "eurybia.core": "eurybia/core",
        "eurybia.report": "eurybia/report",
        "eurybia.style": "eurybia/style",
        "eurybia.utils": "eurybia/utils",
    },
    packages=["eurybia", "eurybia.data", "eurybia.core", "eurybia.report", "eurybia.style", "eurybia.utils"],
    data_files=[
        ("data", ["eurybia/data/house_prices_dataset.csv"]),
        ("data", ["eurybia/data/house_prices_labels.json"]),
        ("data", ["eurybia/data/titanicdata.csv"]),
        ("data", ["eurybia/data/project_info_car_accident.yml"]),
        ("data", ["eurybia/data/project_info_house_price.yml"]),
        ("data", ["eurybia/data/project_info_titanic.yml"]),
        ("data", ["eurybia/data/titanic_altered.csv"]),
        ("data", ["eurybia/data/titanic_original.csv"]),
        ("data", ["eurybia/data/US_Accidents_extract.csv"]),
        ("style", ["eurybia/style/colors.json"]),
    ],
    include_package_data=True,
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    zip_safe=False,
)
