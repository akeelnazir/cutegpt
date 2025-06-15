import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

from cutellm.model_base import CuteLLM
from cutellm.tokenizer import SimpleWordTokenizer

def run_training():
    """
    Run a simple training loop for the CuteLLM model
    This is a basic educational implementation for demonstration purposes
    """
    # Define a simple configuration
    config = {
        "vocab_size": 1000,  # Small vocabulary for demonstration
        "d_model": 128,     # Embedding dimension
        "n_heads": 4,       # Number of attention heads
        "n_layers": 2,      # Number of transformer layers
    }
    
    # Create a simple word-level tokenizer
    tokenizer = SimpleWordTokenizer(vocab_size=config["vocab_size"])
    
    # Create dataloader from real training data
    dataloader = create_training_data(config, tokenizer)
    
    # Initialize model
    model = CuteLLM(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    print("Starting training...")
    
    # Training loop
    for epoch in range(10):  # Tiny number of epochs
        total_loss = 0
        batch_count = 0
        for batch in dataloader:
            inputs, targets = batch
            logits = model(inputs)
            loss = loss_fn(logits.view(-1, config["vocab_size"]), targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/10, Loss: {avg_loss:.4f}")
    
    print("Training complete!")
    
    # Save the model weights
    print("Saving model weights...")
    save_path = "models/cute_llm.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    return model

def create_training_data(config, tokenizer, batch_size=8, seq_length=20):
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
    # Path to training data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "training_data.txt")
    
    print(f"Loading training data from {data_path}")
    
    try:
        with open(data_path, "r") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Warning: Training data file not found at {data_path}")
        print("Falling back to dummy data")
        return create_dummy_data(config)
    
    # Remove comment lines (starting with #)
    lines = [line for line in text.split("\n") if line and not line.startswith("#")]
    text = " ".join(lines)
    
    # Tokenize the entire text
    all_tokens = tokenizer.encode(text)
    
    # Create sequences of fixed length
    sequences = []
    for i in range(0, len(all_tokens) - seq_length, seq_length // 2):  # 50% overlap between sequences
        sequences.append(all_tokens[i:i + seq_length])
    
    if len(sequences) == 0:
        print("Warning: Not enough training data, falling back to dummy data")
        return create_dummy_data(config)
    
    # Convert to tensors
    inputs = []
    targets = []
    for seq in sequences:
        if len(seq) < 2:  # Need at least 2 tokens for input and target
            continue
        inputs.append(seq[:-1])
        targets.append(seq[1:])
    
    # Pad sequences to the same length
    max_len = max(len(seq) for seq in inputs)
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
    
    # Convert to tensors
    inputs_tensor = torch.tensor(padded_inputs, dtype=torch.long)
    targets_tensor = torch.tensor(padded_targets, dtype=torch.long)
    
    print(f"Created dataset with {len(inputs_tensor)} sequences")
    
    # Create dataset and dataloader
    dataset = TensorDataset(inputs_tensor, targets_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
    print("Creating dummy training data")
    # Create random token sequences
    data = np.random.randint(0, config["vocab_size"], size=(batch_size * num_batches, seq_length))
    # Input is all tokens except the last one
    inputs = torch.tensor(data[:, :-1], dtype=torch.long)
    # Target is all tokens except the first one (shifted by 1)
    targets = torch.tensor(data[:, 1:], dtype=torch.long)
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size)

# Allow running this module directly for testing
if __name__ == "__main__":
    run_training()