"""
Setup script for TransMLA Core module
"""

from setuptools import setup, find_packages

setup(
    name="transmla-core",
    version="1.0.0",
    description="Minimal TransMLA implementation for Multi-head Latent Attention",
    author="FedCore Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.52.0",
        "datasets",
        "numpy",
        "tqdm"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
