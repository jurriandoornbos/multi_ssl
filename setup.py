# Copyright 2025 Jurrian Doornbos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from setuptools import setup, find_packages

setup(
    name="multissl",
    version="0.2.0",
    author="Jurrian Doornbos",
    author_email="jurrian.dooornbos@wur.nl",
    description="A multispectral adaptation for lightly-ssl",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jurriandoornbos/multi_ssl",  # Change this to your repository link
    packages=find_packages(),  # Automatically finds all packages in 'multissl/'
    install_requires=[
        "torch",
        "torchvision",
        "pytorch-lightning",
        "lightly",
        "numpy",
        "opencv-python-headless",
        "tifffile",
        "wandb",
        "rioxarray",
        "scikit-learn",
        "pycocotools",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)