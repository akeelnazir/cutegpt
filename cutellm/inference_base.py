import torch
import torch.nn as nn
import logging
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
    logging.info("STEP 1: Starting greedy text generation")
    logging.info(f"Prompt: '{prompt}', Max length: {max_len}")
    
    # Tokenize the prompt
    logging.info("STEP 2: Tokenizing the prompt")
    input_ids = tokenizer.encode(prompt)
    logging.info(f"Encoded prompt to {len(input_ids)} tokens: {input_ids}")
    
    # Generate tokens one by one
    logging.info("STEP 3: Generating tokens sequentially")
    for i in range(max_len):
        # Convert input_ids to tensor and get model predictions
        input_tensor = torch.tensor([input_ids])
        logging.debug(f"Input tensor shape: {input_tensor.shape}")
        
        # Forward pass through the model
        logits = model(input_tensor)
        logging.debug(f"Output logits shape: {logits.shape}")
        
        # Select the most likely next token (greedy decoding)
        next_id = torch.argmax(logits[:, -1]).item()
        logging.debug(f"Generated token {i+1}/{max_len}: {next_id} ('{tokenizer.get_word_from_id(next_id)}')")
        
        # Add the predicted token to the sequence
        input_ids.append(next_id)
    
    # Decode the generated sequence back to text
    logging.info("STEP 4: Decoding generated tokens to text")
    generated_text = tokenizer.decode(input_ids)
    logging.info(f"Generated text: '{generated_text}'")
    
    return generated_text

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
    logging.info("STEP 1: Starting sampling-based text generation")
    logging.info(f"Prompt: '{prompt}', Max length: {max_len}, Temperature: {temperature}")
    
    # Tokenize the prompt
    logging.info("STEP 2: Tokenizing the prompt")
    input_ids = tokenizer.encode(prompt)
    logging.info(f"Encoded prompt to {len(input_ids)} tokens: {input_ids}")
    
    # Generate tokens one by one with sampling
    logging.info("STEP 3: Generating tokens with temperature sampling")
    for i in range(max_len):
        # Convert input_ids to tensor and get model predictions
        input_tensor = torch.tensor([input_ids])
        logging.debug(f"Input tensor shape: {input_tensor.shape}")
        
        # Forward pass through the model
        logits = model(input_tensor)
        logging.debug(f"Output logits shape: {logits.shape}")
        
        # Apply temperature scaling
        logits = logits[:, -1] / temperature
        logging.debug(f"Applied temperature {temperature} to logits")
        
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)
        logging.debug(f"Converted to probability distribution")
        
        # Sample from the probability distribution
        next_id = torch.multinomial(probs, num_samples=1).item()
        logging.debug(f"Sampled token {i+1}/{max_len}: {next_id} ('{tokenizer.get_word_from_id(next_id)}')")
        
        # Add the sampled token to the sequence
        input_ids.append(next_id)
    
    # Decode the generated sequence back to text
    logging.info("STEP 4: Decoding generated tokens to text")
    generated_text = tokenizer.decode(input_ids)
    logging.info(f"Generated text: '{generated_text}'")
    
    return generated_text
