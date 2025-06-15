#!/usr/bin/env python3
"""
CuteLLM - A custom small LLM for educational purposes
This is the main entry point for running the model training
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the training module
from cutellm.training_base import run_training

if __name__ == "__main__":
    print("CuteLLM - A simple educational LLM")
    run_training()
