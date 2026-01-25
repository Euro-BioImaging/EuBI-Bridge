"""
@author: bugra

Setup configuration for EuBI-Bridge.

JDK binaries are downloaded at build time (not install time) by the custom
PEP 517 build backend (_build_backend.py) and bundled in the wheel.
"""

import os
from setuptools import setup, find_packages


def readme():
    """Read the README file."""
    for filename in ['README.md', 'README.rst', 'README.txt']:
        if os.path.exists(filename):
            with open(filename, encoding='utf-8') as f:
                return f.read()
    return ""


setup(
    name='eubi_bridge',
    version='0.1.0c1',
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
            "eubi-gui = eubi_bridge.app:main"
        ]
    },
)
