"""
@author: bugra

Setup configuration for EuBI-Bridge.

JDK binaries are downloaded at build time (not install time) by the custom
PEP 517 build backend (_build_backend.py) and bundled in the wheel.
"""

import os
import tomllib
from setuptools import setup, find_packages


def readme():
    """Read the README file."""
    for filename in ['README.md', 'README.rst', 'README.txt']:
        if os.path.exists(filename):
            with open(filename, encoding='utf-8') as f:
                return f.read()
    return ""


with open(os.path.join(os.path.dirname(__file__), "pyproject.toml"), "rb") as _f:
    _version = tomllib.load(_f)["project"]["version"]


setup(
    name='eubi_bridge',
    version=_version,
    author='Bugra Özdemir',
    author_email='bugraa.ozdemir@gmail.com',
    description='A package for converting datasets to OME-Zarr format.',
    long_description=readme(),
    long_description_content_type="text/markdown",
    url='https://github.com/Euro-BioImaging/EuBI-Bridge',
    license='MIT',
    packages=find_packages(exclude=['tests', 'test_data', 'docs', '_archive', 'new_tests', 'eubi_bridge.bioformats', 'eubi_bridge.bioformats.*']),
    include_package_data=True,
    package_data={
        "eubi_bridge": [
            "bioformats/**",
        ],
    },
    python_requires='>=3.11,<3.13',
    entry_points={
        'console_scripts': [
            "eubi = eubi_bridge.cli:main",
            "eubi-gui = eubi_bridge.app:main",
            "eubi-gui-react = eubi_bridge.gui_react:main"
        ]
    },
)
