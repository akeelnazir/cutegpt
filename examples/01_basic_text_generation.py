"""
Basic Text Generation Example with CuteLLM

This example demonstrates how to load a trained CuteLLM model and generate text
from a given prompt. It shows the simplest way to use the model for inference.
"""

import torch
import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cutellm.model_base import CuteLLM
from cutellm.tokenizer import SimpleWordTokenizer
from cutellm.inference_base import generate, sample_generate

def main():
    # Define the model configuration (must match the trained model)
    config = {
        "vocab_size": 1000,  # Size of vocabulary
        "d_model": 128,      # Embedding dimension
        "n_heads": 4,        # Number of attention heads
        "n_layers": 2,       # Number of transformer layers
    }
    
    # Initialize the model and tokenizer
    model = CuteLLM(config)
    tokenizer = SimpleWordTokenizer(vocab_size=config["vocab_size"])
    
    # Load the trained model weights
    model_path = "models/cute_llm.pth"
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"✓ Loaded model weights from {model_path}")
    except FileNotFoundError:
        print(f"✗ Model weights not found at {model_path}")
        print("  You need to train the model first using cutellm/training_base.py")
        return
    
    # Set the model to evaluation mode
    model.eval()
    
    # Define a prompt for text generation
    prompt = "the cat sat on"
    
    print("\n" + "="*50)
    print("BASIC TEXT GENERATION EXAMPLE")
    print("="*50)
    
    print(f"\nPrompt: '{prompt}'")
    
    # Generate text using greedy decoding (always picking the most likely next token)
    print("\n1. Greedy generation (deterministic):")
    generated_text = generate(model, tokenizer, prompt, max_len=10)
    print(f"   Result: '{generated_text}'")
    
    # Generate text using sampling (more diverse but less predictable)
    print("\n2. Sampling with temperature=1.0 (more diverse):")
    for i in range(3):
        generated_text = sample_generate(model, tokenizer, prompt, max_len=10, temperature=1.0)
        print(f"   Sample {i+1}: '{generated_text}'")
    
    # Generate text with lower temperature (more focused)
    print("\n3. Sampling with temperature=0.5 (more focused):")
    for i in range(3):
        generated_text = sample_generate(model, tokenizer, prompt, max_len=10, temperature=0.5)
        print(f"   Sample {i+1}: '{generated_text}'")
    
    print("\nExplanation:")
    print("- Greedy generation always selects the most probable next token")
    print("- Sampling introduces randomness for more diverse outputs")
    print("- Lower temperature makes sampling more conservative (closer to greedy)")
    print("- Higher temperature makes sampling more random and diverse")

if __name__ == "__main__":
    main()
