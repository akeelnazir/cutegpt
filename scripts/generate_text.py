#!/usr/bin/env python3
"""
CuteGPT - Text Generation Example
This script demonstrates how to use the trained model for text generation
"""

import torch
import numpy as np
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cutellm import CuteLLM, generate, sample_generate
from cutellm.tokenizer import SimpleWordTokenizer

# Using the centralized SimpleWordTokenizer from cutellm.tokenizer

def main():
    print("CuteGPT - Text Generation Demo")
    
    # Define the same configuration used during training
    config = {
        "vocab_size": 1000,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
    }
    
    # Initialize model
    model = CuteLLM(config)
    
    # Try to load saved model weights
    model_path = "models/tiny_llm.pth"
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"✓ Loaded model weights from {model_path}")
    except FileNotFoundError:
        print(f"✗ No saved model found at {model_path}, using untrained model")
    
    # Create tokenizer
    tokenizer = SimpleWordTokenizer(vocab_size=config["vocab_size"])
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate text
    prompt = "Hello world"
    print(f"\nPrompt: {prompt}")
    
    print("\nGenerating with greedy sampling:")
    output = generate(model, tokenizer, prompt, max_len=10)
    print(f"Generated: {output}")
    
    print("\nGenerating with temperature sampling (more diverse):")
    output = sample_generate(model, tokenizer, prompt, max_len=10, temperature=0.8)
    print(f"Generated: {output}")
    
    print("\nNote: This is using our SimpleWordTokenizer for text generation.")
    print("The quality of generated text depends on how well the model was trained.")
    print("Since this is an educational model with limited training, outputs may be simple.")

if __name__ == "__main__":
    main()
