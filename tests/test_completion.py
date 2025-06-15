#!/usr/bin/env python3
"""
CuteGPT - Simple Phrase Completion Test
This script provides a user-friendly interface to test the model's ability 
to complete simple prompts like "The cat sat on the ___"
"""

import torch
import argparse
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cutellm import CuteLLM
from cutellm.tokenizer import SimpleWordTokenizer

# SimpleWordTokenizer is now imported from cutellm.tokenizer

def complete_phrase(model, tokenizer, prompt, num_completions=5, temperature=0.8):
    """
    Complete a phrase by predicting the masked word
    
    Args:
        model: The trained CuteLLM model
        tokenizer: A tokenizer object with encode and decode methods
        prompt: The text prompt containing "___" to indicate completion point
        num_completions: Number of different completions to generate
        temperature: Controls randomness (higher = more diverse)
        
    Returns:
        Tuple of (completions, top_words)
    """
    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    
    # Find the position of the mask token
    mask_position = tokenizer.get_mask_position(input_ids)
    
    # Handle case where no mask token is found
    if mask_position == -1:
        error_msg = "Error: No mask token found in prompt. Use ___ to indicate where to complete."
        # Return a tuple with error message and empty top words list
        return [(error_msg, "<error>")], ["<error>"]
    
    # Get model predictions for the masked position
    with torch.no_grad():
        logits = model(torch.tensor([input_ids]))
        
    # Get probabilities for the masked position
    mask_logits = logits[0, mask_position, :]
    
    # Apply temperature for sampling
    mask_logits = mask_logits / temperature
    probs = torch.softmax(mask_logits, dim=-1)
    
    # Get top-5 predictions
    top_values, top_indices = torch.topk(probs, k=5)
    top_words = [tokenizer.id_to_word.get(idx.item(), "<unk>") for idx in top_indices]
    
    # Sample completions
    completions = []
    for _ in range(num_completions):
        # Sample from the distribution
        next_id = torch.multinomial(probs, num_samples=1).item()
        predicted_word = tokenizer.id_to_word.get(next_id, "<unk>")
        
        # Create the completed phrase by replacing the mask
        words = prompt.lower().split()
        for i, word in enumerate(words):
            if word == "___":
                words[i] = predicted_word
        
        completed_text = " ".join(words)
        completions.append((completed_text, predicted_word))
    
    return completions, top_words

def main():
    parser = argparse.ArgumentParser(description="Test phrase completion with CuteGPT")
    parser.add_argument("--prompt", type=str, help="Prompt to complete (use ___ for the blank)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("--num", type=int, default=5, help="Number of completions to generate")
    
    args = parser.parse_args()
    
    print("CuteGPT - Phrase Completion Test")
    print("--------------------------------")
    
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
        print(f"✓ Loaded model weights from {model_path}")
    except FileNotFoundError:
        print(f"✗ No saved model found at {model_path}, using untrained model")
    
    # Create tokenizer
    tokenizer = SimpleWordTokenizer(vocab_size=config["vocab_size"])
    
    # Set model to evaluation mode
    model.eval()
    
    # Sample prompts
    sample_prompts = [
        "The cat sat on the ___",
        "The dog is in the ___",
        "I like to eat ___",
        "She went to the ___",
        "They were playing with a ___"
    ]
    
    # If prompt is provided, use it; otherwise use interactive mode
    if args.prompt:
        prompt = args.prompt
        completions, top_words = complete_phrase(
            model, tokenizer, prompt, 
            num_completions=args.num, 
            temperature=args.temperature
        )
        
        print(f"\nPrompt: {prompt}")
        print(f"Top 5 most likely words: {', '.join(top_words)}")
        print("\nCompletions:")
        for i, (completion, word) in enumerate(completions):
            print(f"  {i+1}. {completion} (filled with: '{word}')")
    else:
        # Interactive mode
        print("\nEnter a prompt with ___ to indicate where to complete.")
        print("Example: 'The cat sat on the ___'")
        print("Type 'exit', 'bye', or 'quit' to exit or 'sample' to see sample prompts.")
        
        while True:
            user_input = input("\nEnter prompt> ").strip()
            
            if user_input.lower() in ['exit', 'bye', 'quit']:
                print("Exiting phrase completion test. Goodbye!")
                break
            elif user_input.lower() == 'sample':
                print("\nSample prompts:")
                for i, prompt in enumerate(sample_prompts):
                    print(f"  {i+1}. {prompt}")
                continue
            
            if "___" not in user_input:
                print("Error: Prompt must contain ___ to indicate where to complete.")
                continue
            
            try:
                completions, top_words = complete_phrase(
                    model, tokenizer, user_input, 
                    num_completions=args.num, 
                    temperature=args.temperature
                )
                
                print(f"\nTop 5 most likely words: {', '.join(top_words)}")
                print("\nCompletions:")
                for i, (completion, word) in enumerate(completions):
                    print(f"  {i+1}. {completion} (filled with: '{word}')")
            except Exception as e:
                print(f"Error during completion: {str(e)}")

if __name__ == "__main__":
    main()
