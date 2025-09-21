# -*- coding: utf-8 -*-
"""
@author: bugra
"""

import setuptools
import os

def get_requirements():
    """Get requirements from requirements.txt or return default requirements."""
    requirements = [
        "aicspylibczi>=0.0.0",
        "asciitree>=0.3.3",
        "bfio>=0.0.0",
        "bioformats_jar>=0.0.0",
        "bioio-base>=0.0.0",
        "bioio-bioformats==1.1.0",
        "bioio-czi==2.1.0",
        "bioio-imageio==1.1.0",
        "bioio-lif==1.1.0",
        "bioio-nd2==1.1.0",
        "bioio-ome-tiff-fork-by-bugra==0.0.1b2",
        "bioio-tifffile-fork-by-bugra>=0.0.1b2",
        "cmake==4.0.2",
        "dask>=2024.12.1",
        "dask-jobqueue>=0.0.0",
        "distributed>=2024.12.1",
        "elementpath==5.0.1",
        "fasteners==0.19",
        "fire>=0.0.0",
        "imageio==2.27.0",
        "imageio-ffmpeg==0.6.0",
        "natsort>=0.0.0",
        "nd2>=0.0.0",
        "numpy>=0.0.0",
        # "openjdk==8.*",
        "pydantic>=2.11.7",
        "pylibczirw>=0.0.0",
        "readlif==0.6.5",
        "s3fs>=0.0.0",
        "scipy>=1.8",
        "tensorstore>=0.0.0",
        "tifffile>=2025.5.21",
        "validators==0.35.0",
        "xarray>=0.0.0",
        "xmlschema>=0.0.0",
        "xmltodict==0.14.2",
        "zarr>=3.0",
        "zstandard>=0.0.0",
        #
        "aiofiles>=24.1.0",
        "blosc2>=3.7.1",
        "fastapi>=0.116.1",
        "lz4>=4.4.4",
        "numpy>=2.3.2",
        "psutil>=7.0.0",
        "rich>=14.1.0",
        "uvicorn>=0.35.0",
        "websockets>=15.0.1",
    ]

    # Optionally still try to read from requirements.txt if it exists
    # if os.path.exists('../requirements.txt'):
    #     with open('../requirements.txt', encoding='utf-8') as f:
    #         requirements = [
    #             line.strip() for line in f
    #             if line.strip() and not line.startswith('#')
    #         ]
    return requirements

def readme():
    """Read the README file."""
    for filename in ['README.md', 'README.rst', 'README.txt']:
        if os.path.exists(filename):
            with open(filename, encoding='utf-8') as f:
                return f.read()
    return ""

setuptools.setup(
    name='eubi_bridge',
    version='0.0.8c1',
    author='Bugra Özdemir',
    author_email='bugraa.ozdemir@gmail.com',
    description='A package for converting datasets to OME-Zarr format.',
    long_description=readme(),
    long_description_content_type="text/markdown",
    url='https://github.com/Euro-BioImaging/EuBI-Bridge',
    license='MIT',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=get_requirements(),
    python_requires='>=3.11,<3.13',
    extras_require={
        # Include CUDA variants if needed
        'cuda11': ['cupy-cuda11x'],
        'cuda12': ['cupy-cuda12x'],
    },
    entry_points={
        'console_scripts': [
            "eubi = eubi_bridge.cli:main"
        ]
    },
)
