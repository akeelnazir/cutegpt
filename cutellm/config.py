"""
Configuration utilities for CuteGPT.
Handles environment variables and logging configuration.
"""
import os
import logging
from pathlib import Path

def _read_env_file(env_path):
    """
    Simple function to read key-value pairs from a .env file
    and set them as environment variables if not already set.
    """
    if not env_path.exists():
        return False
    
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Split on first equals sign
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Only set if not already in environment
                    if key and key not in os.environ:
                        os.environ[key] = value
        return True
    except Exception as e:
        print(f"Error reading .env file: {e}")
        return False

def setup_logging():
    """
    Set up logging based on environment configuration.
    Reads LOG_LEVEL from .env file or environment variables.
    """
    # Try to load .env file if it exists
    env_path = Path(os.path.dirname(os.path.dirname(__file__))) / '.env'
    if env_path.exists():
        _read_env_file(env_path)
        print(f"Loaded environment configuration from {env_path}")
    
    # Get log level from environment, default to INFO
    log_level_name = os.environ.get('LOG_LEVEL', 'INFO').upper()
    
    # Map string log level to logging constants
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    # Get the numeric log level, default to INFO if not found
    log_level = log_level_map.get(log_level_name, logging.INFO)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logging.info(f"Logging configured with level: {log_level_name}")
    
    return log_level

def get_model_config():
    """
    Get model configuration from environment variables.
    Returns a dictionary with model parameters.
    """
    # Try to load .env file if it exists and not already loaded
    env_path = Path(os.path.dirname(os.path.dirname(__file__))) / '.env'
    if env_path.exists():
        _read_env_file(env_path)
    
    # Get model configuration from environment with defaults
    config = {
        "vocab_size": int(os.environ.get('MODEL_VOCAB_SIZE', 1000)),
        "d_model": int(os.environ.get('MODEL_D_MODEL', 128)),
        "n_heads": int(os.environ.get('MODEL_N_HEADS', 4)),
        "n_layers": int(os.environ.get('MODEL_N_LAYERS', 2)),
    }
    
    logging.info(f"Model configuration loaded: {config}")
    return config

def get_training_config():
    """
    Get training configuration from environment variables.
    Returns a dictionary with training parameters.
    """
    # Try to load .env file if it exists and not already loaded
    env_path = Path(os.path.dirname(os.path.dirname(__file__))) / '.env'
    if env_path.exists():
        _read_env_file(env_path)
    
    # Get training configuration from environment with defaults
    config = {
        "batch_size": int(os.environ.get('TRAINING_BATCH_SIZE', 8)),
        "seq_length": int(os.environ.get('TRAINING_SEQ_LENGTH', 20)),
        "epochs": int(os.environ.get('TRAINING_EPOCHS', 10)),
    }
    
    logging.info(f"Training configuration loaded: {config}")
    return config
