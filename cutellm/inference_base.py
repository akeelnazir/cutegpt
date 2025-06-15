import torch
import torch.nn as nn
from .model_base import CuteLLM

def generate(model, tokenizer, prompt, max_len=20):
    """
    Generate text using the trained model
    
    Args:
        model: The trained CuteLLM model
        tokenizer: A tokenizer object with encode and decode methods
        prompt: The text prompt to start generation from
        max_len: Maximum number of tokens to generate
        
    Returns:
        Generated text including the prompt
    """
    input_ids = tokenizer.encode(prompt)
    for _ in range(max_len):
        logits = model(torch.tensor([input_ids]))
        next_id = torch.argmax(logits[:, -1]).item()
        input_ids.append(next_id)
    return tokenizer.decode(input_ids)

def sample_generate(model, tokenizer, prompt, max_len=20, temperature=1.0):
    """
    Generate text with sampling for more diverse outputs
    
    Args:
        model: The trained CuteLLM model
        tokenizer: A tokenizer object with encode and decode methods
        prompt: The text prompt to start generation from
        max_len: Maximum number of tokens to generate
        temperature: Controls randomness (higher = more random)
        
    Returns:
        Generated text including the prompt
    """
    input_ids = tokenizer.encode(prompt)
    for _ in range(max_len):
        logits = model(torch.tensor([input_ids]))
        # Apply temperature
        logits = logits[:, -1] / temperature
        # Sample from the distribution
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        input_ids.append(next_id)
    return tokenizer.decode(input_ids)
