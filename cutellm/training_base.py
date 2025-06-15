import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import logging

from cutellm.model_base import CuteLLM
from cutellm.tokenizer import SimpleWordTokenizer
from cutellm.config import get_model_config, get_training_config

def run_training():
    """
    Run a simple training loop for the CuteLLM model
    This is a basic educational implementation for demonstration purposes
    """
    logging.info("STEP 1: Setting up model configuration")
    # Get configuration from environment or use defaults
    config = get_model_config()
    logging.info(f"Model configuration: vocab_size={config['vocab_size']}, d_model={config['d_model']}, "
                f"n_heads={config['n_heads']}, n_layers={config['n_layers']}")
    
    logging.info("STEP 2: Initializing tokenizer")
    # Create a simple word-level tokenizer
    tokenizer = SimpleWordTokenizer(vocab_size=config["vocab_size"])
    logging.info(f"Tokenizer initialized with vocabulary size: {config['vocab_size']}")
    
    logging.info("STEP 3: Creating training dataset")
    # Create dataloader from real training data
    dataloader = create_training_data(config, tokenizer)
    
    logging.info("STEP 4: Initializing model and optimizer")
    # Initialize model
    model = CuteLLM(config)
    logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    logging.info("Optimizer: Adam with learning rate 3e-4")
    
    loss_fn = nn.CrossEntropyLoss()
    logging.info("Loss function: CrossEntropyLoss")
    
    logging.info("STEP 5: Starting training loop")
    
    # Training loop
    training_config = get_training_config()
    num_epochs = training_config["epochs"]
    for epoch in range(num_epochs):  # Number of epochs from config
        logging.info(f"Epoch {epoch+1}/{num_epochs} starting")
        total_loss = 0
        batch_count = 0
        for batch_idx, batch in enumerate(dataloader):
            inputs, targets = batch
            logging.debug(f"Processing batch {batch_idx+1}/{len(dataloader)}, "
                         f"input shape: {inputs.shape}, target shape: {targets.shape}")
            
            # Forward pass
            logits = model(inputs)
            
            # Calculate loss
            loss = loss_fn(logits.view(-1, config["vocab_size"]), targets.view(-1))
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track statistics
            total_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 5 == 0:  # Log every 5 batches
                logging.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                             f"Loss: {loss.item():.4f}")
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / batch_count
        logging.info(f"Epoch {epoch+1}/{num_epochs} complete, Average Loss: {avg_loss:.4f}")
    
    logging.info("STEP 6: Training complete!")
    
    # Save the model weights
    logging.info("STEP 7: Saving model weights")
    os.makedirs("models", exist_ok=True)  # Ensure directory exists
    save_path = "models/cute_llm.pth"
    torch.save(model.state_dict(), save_path)
    logging.info(f"Model saved to {save_path}")
    
    return model

def create_training_data(config, tokenizer, batch_size=None, seq_length=None):
    # Use values from config if not explicitly provided
    training_config = get_training_config()
    batch_size = batch_size or training_config["batch_size"]
    seq_length = seq_length or training_config["seq_length"]
    """
    Create a dataset from real training data in data/training_data.txt
    
    Args:
        config: Model configuration dictionary
        tokenizer: Tokenizer to convert text to token IDs
        batch_size: Number of sequences per batch
        seq_length: Maximum length of each sequence
        
    Returns:
        DataLoader with real training data
    """
    logging.info("Creating training dataset from real data")
    logging.info(f"Parameters: batch_size={batch_size}, seq_length={seq_length}")
    
    # Path to training data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "training_data.txt")
    logging.info(f"Loading training data from {data_path}")
    
    try:
        with open(data_path, "r") as f:
            text = f.read()
            logging.info(f"Successfully loaded {len(text)} characters of training data")
    except FileNotFoundError:
        logging.warning(f"Training data file not found at {data_path}")
        logging.warning("Falling back to dummy data")
        return create_dummy_data(config)
    
    # Remove comment lines (starting with #)
    logging.info("Processing training data - removing comments and empty lines")
    lines = [line for line in text.split("\n") if line and not line.startswith("#")]
    text = " ".join(lines)
    logging.info(f"Processed text length: {len(text)} characters")
    
    # Tokenize the entire text
    logging.info("Tokenizing text data")
    all_tokens = tokenizer.encode(text)
    logging.info(f"Generated {len(all_tokens)} tokens from the text")
    
    # Create sequences of fixed length
    logging.info(f"Creating sequences with length {seq_length} and 50% overlap")
    sequences = []
    for i in range(0, len(all_tokens) - seq_length, seq_length // 2):  # 50% overlap between sequences
        sequences.append(all_tokens[i:i + seq_length])
    
    logging.info(f"Created {len(sequences)} sequences")
    
    if len(sequences) == 0:
        logging.warning("Warning: Not enough training data, falling back to dummy data")
        return create_dummy_data(config)
    
    # Convert to tensors
    logging.info("Creating input-target pairs for next token prediction")
    inputs = []
    targets = []
    for seq in sequences:
        if len(seq) < 2:  # Need at least 2 tokens for input and target
            continue
        inputs.append(seq[:-1])  # All tokens except the last one
        targets.append(seq[1:])  # All tokens except the first one (shifted by 1)
    
    logging.info(f"Created {len(inputs)} input-target pairs")
    
    # Pad sequences to the same length
    logging.info("Padding sequences to uniform length")
    max_len = max(len(seq) for seq in inputs)
    logging.info(f"Maximum sequence length: {max_len}")
    padded_inputs = []
    padded_targets = []
    
    for i in range(len(inputs)):
        if len(inputs[i]) < max_len:
            # Pad with <unk> token
            pad_len = max_len - len(inputs[i])
            padded_inputs.append(inputs[i] + [tokenizer.word_to_id["<unk>"]] * pad_len)
            padded_targets.append(targets[i] + [tokenizer.word_to_id["<unk>"]] * pad_len)
        else:
            padded_inputs.append(inputs[i])
            padded_targets.append(targets[i])
    
    logging.info(f"Padded all sequences to length {max_len}")
    
    # Convert to tensors
    logging.info("Converting data to PyTorch tensors")
    inputs_tensor = torch.tensor(padded_inputs, dtype=torch.long)
    targets_tensor = torch.tensor(padded_targets, dtype=torch.long)
    
    logging.info(f"Created tensors with shapes: inputs {inputs_tensor.shape}, targets {targets_tensor.shape}")
    
    # Create dataset and dataloader
    logging.info(f"Creating DataLoader with batch size {batch_size}")
    dataset = TensorDataset(inputs_tensor, targets_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    logging.info(f"DataLoader created with {len(dataloader)} batches")
    
    return dataloader

# Kept for backwards compatibility
def create_dummy_data(config, seq_length=20, batch_size=8, num_batches=5):
    """
    Create a simple synthetic dataset for training the model
    
    Args:
        config: Model configuration dictionary
        seq_length: Length of each sequence
        batch_size: Number of sequences per batch
        num_batches: Number of batches to generate
        
    Returns:
        DataLoader with synthetic data
    """
    logging.info("Creating synthetic (dummy) training data")
    logging.info(f"Parameters: seq_length={seq_length}, batch_size={batch_size}, num_batches={num_batches}")
    
    # Create random token sequences
    logging.info(f"Generating {batch_size * num_batches} random sequences of length {seq_length}")
    data = np.random.randint(0, config["vocab_size"], size=(batch_size * num_batches, seq_length))
    
    # Input is all tokens except the last one
    inputs = torch.tensor(data[:, :-1], dtype=torch.long)
    logging.info(f"Created input tensor with shape {inputs.shape}")
    
    # Target is all tokens except the first one (shifted by 1)
    targets = torch.tensor(data[:, 1:], dtype=torch.long)
    logging.info(f"Created target tensor with shape {targets.shape}")
    
    # Create dataset and dataloader
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    logging.info(f"Created DataLoader with {len(dataloader)} batches")
    
    return dataloader

# Allow running this module directly for testing
if __name__ == "__main__":
    logging.info("Starting CuteLLM training process")
    run_training()
    logging.info("Training process completed")