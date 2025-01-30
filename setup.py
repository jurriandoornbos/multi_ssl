from setuptools import setup, find_packages

setup(
    name="Multispectral SSL",
    version="0.1.0",
    author="Jurrian Doornbos",
    author_email="jurrian.dooornbos@wur.nl",
    description="A multispectral adapataion for lightly-ssl",
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
        "wandb"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)