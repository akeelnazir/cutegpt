#!/usr/bin/env python3
"""
CuteGPT - Comprehensive Phrase Completion Evaluation
This script evaluates the model's ability to generate coherent short phrases
by completing simple prompts like "The cat sat on the ___"
"""

import torch
import numpy as np
import argparse
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cutellm import CuteLLM
from cutellm.tokenizer import SimpleWordTokenizer

# Using the centralized SimpleWordTokenizer from cutellm.tokenizer

def evaluate_phrase_completion(model, tokenizer, prompt, top_k=5, temperature_values=[0.5, 0.7, 1.0, 1.2]):
    """
    Evaluate phrase completion with different temperature settings
    
    Args:
        model: The trained CuteGPT model
        tokenizer: A tokenizer object with encode and decode methods
        prompt: The text prompt containing "___" to indicate completion point
        top_k: Number of top predictions to show
        temperature_values: List of temperature values to test
        
    Returns:
        Dictionary with evaluation results
    """
    results = {}
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    
    # Find the position of the mask token
    mask_position = tokenizer.get_mask_position(input_ids)
    
    if mask_position == -1:
        return {"error": "No mask token found in prompt"}
    
    # Get model predictions for the masked position
    with torch.no_grad():
        logits = model(torch.tensor([input_ids]))
    
    # Get raw logits for the masked position
    mask_logits = logits[0, mask_position, :]
    
    # Get top-k predictions without temperature
    top_values, top_indices = torch.topk(mask_logits, k=top_k)
    
    # Convert to probabilities
    top_probs = torch.softmax(top_values, dim=-1)
    
    # Store top predictions
    top_predictions = []
    for i in range(top_k):
        token_id = top_indices[i].item()
        token = tokenizer.get_word_from_id(token_id)
        probability = top_probs[i].item()
        top_predictions.append({
            "token": token,
            "probability": probability,
            "token_id": token_id
        })
    
    results["top_predictions"] = top_predictions
    
    # Test with different temperature values
    temperature_results = {}
    for temp in temperature_values:
        # Apply temperature
        temp_logits = mask_logits / temp
        temp_probs = torch.softmax(temp_logits, dim=-1)
        
        # Sample 5 completions with this temperature
        completions = []
        for _ in range(5):
            # Sample from the distribution
            next_id = torch.multinomial(temp_probs, num_samples=1).item()
            token = tokenizer.get_word_from_id(next_id)
            
            # Create a copy of input_ids and replace mask with predicted token
            completed_ids = input_ids.copy()
            completed_ids[mask_position] = next_id
            
            # Decode the completed phrase
            completed_text = tokenizer.decode(completed_ids)
            completions.append({
                "text": completed_text,
                "token": token,
                "token_id": next_id
            })
        
        temperature_results[temp] = completions
    
    results["temperature_results"] = temperature_results
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate phrase completion with CuteGPT")
    parser.add_argument("--prompt", type=str, default="The cat sat on the ___",
                        help="Prompt to complete (use ___ for the masked word)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top predictions to show")
    parser.add_argument("--temperatures", type=str, default="0.5,0.7,1.0,1.2",
                        help="Comma-separated list of temperature values to test")
    
    args = parser.parse_args()
    
    print("CuteGPT - Phrase Completion Evaluation")
    
    # Parse temperature values
    temperature_values = [float(t) for t in args.temperatures.split(",")]
    
    # Define the model configuration
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
        print(f"Loaded model weights from {model_path}")
    except FileNotFoundError:
        print(f"No saved model found at {model_path}, using untrained model")
    
    # Create tokenizer
    tokenizer = SimpleWordTokenizer(vocab_size=config["vocab_size"])
    print(f"Tokenizer vocabulary size: {tokenizer.get_vocabulary_size()}")
    
    # Set model to evaluation mode
    model.eval()
    
    # If no prompt is provided, use a set of test prompts
    if args.prompt == "The cat sat on the ___":
        test_prompts = [
            "The cat sat on the ___",
            "The dog is in the ___",
            "I like to eat ___",
            "She went to the ___",
            "They were playing with a ___"
        ]
    else:
        test_prompts = [args.prompt]
    
    # Evaluate each prompt
    for prompt in test_prompts:
        print("\n" + "="*60)
        print(f"Evaluating prompt: \"{prompt}\"")
        print("="*60)
        
        results = evaluate_phrase_completion(
            model, tokenizer, prompt, 
            top_k=args.top_k, 
            temperature_values=temperature_values
        )
        
        if "error" in results:
            print(f"Error: {results['error']}")
            continue
        
        # Print top predictions
        print("\nTop predictions (without temperature):")
        for i, pred in enumerate(results["top_predictions"]):
            print(f"  {i+1}. \"{pred['token']}\" (probability: {pred['probability']:.4f})")
        
        # Print results with different temperatures
        print("\nCompletions with different temperature settings:")
        for temp, completions in results["temperature_results"].items():
            print(f"\nTemperature = {temp}:")
            for i, comp in enumerate(completions):
                print(f"  {i+1}. \"{comp['text']}\" (filled with: \"{comp['token']}\")")
    
    print("\n" + "="*60)
    print("Note: This is an educational model with limited training.")
    print("The quality of completions depends on how well the model was trained.")

if __name__ == "__main__":
    main()
