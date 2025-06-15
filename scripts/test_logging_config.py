"""
Test script to demonstrate environment-based logging configuration.
"""
import os
import sys
import logging

# Add the parent directory to sys.path to import cutellm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set log level in environment for testing
os.environ['LOG_LEVEL'] = 'WARNING'  # Try changing this to INFO, DEBUG, etc.

# Import our config module
from cutellm.config import setup_logging

def main():
    # Set up logging based on environment
    log_level = setup_logging()
    
    # Create a logger for this module
    logger = logging.getLogger(__name__)
    
    # Log messages at different levels
    logger.debug("This is a DEBUG message - shows detailed information")
    logger.info("This is an INFO message - shows confirmation that things are working")
    logger.warning("This is a WARNING message - shows something unexpected happened")
    logger.error("This is an ERROR message - shows a more serious problem")
    logger.critical("This is a CRITICAL message - shows a serious error")
    
    # Show the current log level
    print(f"\nCurrent log level: {logging.getLevelName(log_level)}")
    print("You should see all messages at or above this level.")
    print("To change the level, modify the LOG_LEVEL environment variable or .env file.")

if __name__ == "__main__":
    main()
