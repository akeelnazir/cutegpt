#!/usr/bin/env python3
"""
CuteLLM - A custom small LLM for educational purposes
"""

from setuptools import setup, find_packages

setup(
    name="cutellm",
    version="0.1.0",
    description="A custom small LLM for educational purposes",
    author="CuteLLM Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.26.0",
        "tokenizers>=0.15.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
