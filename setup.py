from setuptools import setup, find_packages

setup(
    name="connect4",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "gymnasium",
        "torch",  # PyTorch for neural networks
    ],
)