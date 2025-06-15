"""
Custom Fine-Tuning Example with CuteLLM

This example demonstrates how to fine-tune the CuteLLM model on your own custom data.
It shows the process of preparing a custom dataset and training the model further.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cutellm.model_base import CuteLLM
from cutellm.tokenizer import SimpleWordTokenizer

def create_custom_dataset(text, tokenizer, seq_length=20):
    """
    Create a dataset from custom text
    
    Args:
        text: The text to use for training
        tokenizer: Tokenizer to convert text to token IDs
        seq_length: Maximum length of each sequence
        
    Returns:
        DataLoader with the custom training data
    """
    print(f"Creating dataset from custom text ({len(text)} characters)")
    
    # Tokenize the entire text
    all_tokens = tokenizer.encode(text)
    
    # Create sequences of fixed length
    sequences = []
    for i in range(0, len(all_tokens) - seq_length, seq_length // 2):  # 50% overlap between sequences
        sequences.append(all_tokens[i:i + seq_length])
    
    if len(sequences) == 0:
        print("Error: Not enough training data")
        return None
    
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
    return DataLoader(dataset, batch_size=4, shuffle=True)

def main():
    print("\n" + "="*50)
    print("CUSTOM FINE-TUNING EXAMPLE")
    print("="*50)
    
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
    
    # Load the pre-trained model weights
    model_path = "models/cute_llm.pth"
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"✓ Loaded pre-trained model weights from {model_path}")
    except FileNotFoundError:
        print(f"✗ Pre-trained model weights not found at {model_path}")
        print("  Starting with a fresh model (not recommended)")
    
    # Custom training data - this is a simple example
    # In a real scenario, you would load data from a file
    custom_text = """
    the cute cat sat on the mat and looked at the dog
    the dog was happy to see the cat
    they played together in the garden
    the sun was shining and the sky was blue
    it was a perfect day for the cat and dog to be friends
    """
    
    # Create a dataset from the custom text
    dataloader = create_custom_dataset(custom_text, tokenizer)
    if dataloader is None:
        return
    
    # Setup optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Lower learning rate for fine-tuning
    loss_fn = nn.CrossEntropyLoss()
    
    # Fine-tuning loop
    print("\nStarting fine-tuning...")
    
    num_epochs = 20  # More epochs for fine-tuning
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        
        for batch in dataloader:
            inputs, targets = batch
            
            # Forward pass
            logits = model(inputs)
            loss = loss_fn(logits.view(-1, config["vocab_size"]), targets.view(-1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        # Print progress
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    print("\nFine-tuning complete!")
    
    # Save the fine-tuned model
    fine_tuned_path = "models/cute_llm_fine_tuned.pth"
    torch.save(model.state_dict(), fine_tuned_path)
    print(f"Fine-tuned model saved to {fine_tuned_path}")
    
    # Test the fine-tuned model
    print("\nTesting the fine-tuned model:")
    model.eval()
    
    # Test prompts related to the fine-tuning data
    test_prompts = [
        "the cute cat",
        "the dog was",
        "they played",
        "it was a"
    ]
    
    for prompt in test_prompts:
        # Generate text using the model
        input_ids = tokenizer.encode(prompt)
        for _ in range(10):  # Generate 10 tokens
            with torch.no_grad():
                logits = model(torch.tensor([input_ids]))
                next_id = torch.argmax(logits[:, -1]).item()
                input_ids.append(next_id)
        
        generated_text = tokenizer.decode(input_ids)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{generated_text}'")
    
    print("\nNote: You can see how the model has adapted to the new training data.")
    print("The model should now be better at completing sentences related to cats and dogs.")

if __name__ == "__main__":
    main()
