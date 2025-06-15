#!/usr/bin/env python3
"""
CuteGPT - Enhanced Phrase Completion Test
This script tests the model's ability to generate coherent short phrases
by completing simple prompts like "The cat sat on the ___"
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

def complete_phrase(model, tokenizer, prompt, num_candidates=5, temperature=0.8):
    """
    Complete a phrase by predicting the masked word
    
    Args:
        model: The trained CuteLLM model
        tokenizer: A tokenizer object with encode and decode methods
        prompt: The text prompt containing "___" to indicate completion point
        num_candidates: Number of different completions to generate
        temperature: Controls randomness (higher = more diverse)
        
    Returns:
        List of completed phrases
    """
    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    
    # Find the position of the mask token
    mask_position = tokenizer.get_mask_position(input_ids)
    
    if mask_position == -1:
        return ["Error: No mask token found in prompt"]
    
    # Get model predictions for the masked position
    with torch.no_grad():
        logits = model(torch.tensor([input_ids]))
        
    # Get probabilities for the masked position
    mask_logits = logits[0, mask_position, :]
    
    # Apply temperature for sampling
    mask_logits = mask_logits / temperature
    probs = torch.softmax(mask_logits, dim=-1)
    
    # Sample top candidates
    completions = []
    for _ in range(num_candidates):
        # Sample from the distribution
        next_id = torch.multinomial(probs, num_samples=1).item()
        
        # Create a copy of input_ids and replace mask with predicted token
        completed_ids = input_ids.copy()
        completed_ids[mask_position] = next_id
        
        # Decode the completed phrase
        completed_text = tokenizer.decode(completed_ids)
        completions.append(completed_text)
    
    return completions

def main():
    print("CuteGPT - Enhanced Phrase Completion Test")
    
    # Define the same configuration used during training
    config = {
        "vocab_size": 1000,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
    }
    
    # Initialize model
    model = CuteLLM(config)
    
    # Try to load saved model weights if they exist
    model_path = "models/cute_llm.pth"
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model weights from {model_path}")
    except FileNotFoundError:
        print(f"No saved model found at {model_path}, using untrained model")
    
    # Create tokenizer
    tokenizer = SimpleWordTokenizer(vocab_size=config["vocab_size"])
    
    # Set model to evaluation mode
    model.eval()
    
    # Test with simple prompts
    test_prompts = [
        "The cat sat on the ___",
        "The dog is in the ___",
        "I like to eat ___",
        "She went to the ___",
        "They were playing with a ___"
    ]
    
    print("\nTesting phrase completion with the model:")
    for prompt in test_prompts:
        print("\n" + "-"*50)
        print(f"Prompt: {prompt}")
        
        # Get completions
        completions = complete_phrase(model, tokenizer, prompt)
        
        print("Possible completions:")
        for i, completion in enumerate(completions):
            print(f"  {i+1}. {completion}")
    
    print("\n" + "="*50)
    print("Note: This is using a simple word tokenizer for demonstration purposes.")
    print("The quality of completions depends on how well the model was trained.")
    print("Since this is an educational model with limited training, completions may not be coherent.")

if __name__ == "__main__":
    main()
