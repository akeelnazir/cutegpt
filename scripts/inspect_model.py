#!/usr/bin/env python3
"""
CuteGPT - Model Inspection Script
This script demonstrates how to inspect a saved PyTorch model
"""

import torch
import numpy as np
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cutellm import CuteLLM

def main():
    print("CuteGPT - Model Inspection Demo")
    
    # Define the same configuration used during training
    config = {
        "vocab_size": 1000,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
    }
    
    # Load the saved model state dictionary
    print("\nLoading model state dictionary...")
    model_path = "models/tiny_llm.pth"
    state_dict = torch.load(model_path)
    
    # Print the keys (layer names) in the state dictionary
    print("\nModel contains the following layers:")
    for key in state_dict.keys():
        # Get the shape of each parameter tensor
        shape = state_dict[key].shape
        # Calculate the number of parameters in this layer
        num_params = np.prod(shape)
        print(f"  {key}: shape={shape}, parameters={num_params}")
    
    # Calculate the total number of parameters
    total_params = sum(np.prod(v.shape) for v in state_dict.values())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Initialize a new model
    print("\nInitializing a new model...")
    model = CuteLLM(config)
    
    # Print a sample of parameters before loading
    print("\nSample of embedding weights before loading (random initialization):")
    print(model.embed.weight[0, :10])  # First 10 values of first embedding vector
    
    # Load the state dictionary into the model
    print("\nLoading saved weights into model...")
    model.load_state_dict(state_dict)
    
    # Print the same sample after loading
    print("\nSame sample after loading trained weights:")
    print(model.embed.weight[0, :10])  # Should be different now
    
    print("\nThis is how you use a saved model:")
    print("1. Define the model architecture with the same configuration")
    print("2. Initialize the model: model = CuteLLM(config)")
    print("3. Load the saved weights: model.load_state_dict(torch.load('tiny_llm.pth'))")
    print("4. Set to evaluation mode for inference: model.eval()")
    print("5. Use the model for predictions!")

if __name__ == "__main__":
    main()
