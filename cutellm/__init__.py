# CuteLLM - A custom small LLM for educational purposes
# This is a simple educational implementation to understand the basics of language models

# Configure logging from environment
import logging
from .config import setup_logging

# Set up logging based on environment configuration
log_level = setup_logging()

# Import model components
from .model_base import CuteLLM

# Import training functionality
from .training_base import run_training

# Import inference functionality
from .inference_base import generate, sample_generate

# Import configuration utilities
from .config import get_model_config, get_training_config

# Log package initialization
logger = logging.getLogger(__name__)
logger.info("CuteLLM package initialized")

