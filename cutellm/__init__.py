# CuteLLM - A custom small LLM for educational purposes
# This is a simple educational implementation to understand the basics of language models

# Import model components
from .model_base import CuteLLM

# Import training functionality
from .training_base import run_training

# Import inference functionality
from .inference_base import generate, sample_generate
