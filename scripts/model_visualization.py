"""
Model Visualization Example with CuteLLM

This example demonstrates how to visualize the CuteLLM model architecture
and its logits when making predictions. It provides graphical representations
to help understand the model's internal workings.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.patches import Rectangle
import seaborn as sns
from collections import defaultdict

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cutellm.model_base import CuteLLM
from cutellm.tokenizer import SimpleWordTokenizer

def visualize_model_architecture(model, save_path="docs/images/model_architecture.png"):
    """
    Create a visual representation of the model architecture
    
    Args:
        model: The CuteLLM model
        save_path: Path to save the visualization
    """
    # Get layer dimensions
    embed_size = model.embed.weight.shape[1]  # Embedding dimension
    vocab_size = model.embed.weight.shape[0]  # Vocabulary size
    n_layers = len(model.transformer.layers)  # Number of transformer layers
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors
    colors = {
        'embedding': '#8dd3c7',
        'transformer': '#ffffb3',
        'attention': '#bebada',
        'feedforward': '#fb8072',
        'output': '#80b1d3'
    }
    
    # Define layer heights and spacing
    layer_height = 0.8
    vertical_spacing = 0.5
    total_height = (n_layers * 2 + 2) * (layer_height + vertical_spacing)
    
    # Draw embedding layer
    y_pos = total_height - layer_height
    ax.add_patch(Rectangle((0, y_pos), 2, layer_height, 
                          facecolor=colors['embedding'], edgecolor='black'))
    ax.text(1, y_pos + layer_height/2, f"Embedding\n({vocab_size}, {embed_size})", 
            ha='center', va='center', fontsize=10)
    
    # Draw transformer layers
    for i in range(n_layers):
        # Attention block
        y_pos -= layer_height + vertical_spacing
        ax.add_patch(Rectangle((0, y_pos), 2, layer_height, 
                              facecolor=colors['attention'], edgecolor='black'))
        ax.text(1, y_pos + layer_height/2, f"Self-Attention Layer {i+1}\n(heads={model.transformer.layers[i].self_attn.num_heads})", 
                ha='center', va='center', fontsize=10)
        
        # Add arrow
        ax.arrow(1, y_pos, 0, -vertical_spacing/2, head_width=0.1, 
                head_length=vertical_spacing/4, fc='black', ec='black')
        
        # Feed-forward block
        y_pos -= layer_height + vertical_spacing
        ax.add_patch(Rectangle((0, y_pos), 2, layer_height, 
                              facecolor=colors['feedforward'], edgecolor='black'))
        ax.text(1, y_pos + layer_height/2, f"Feed-Forward Layer {i+1}\n(dim={model.transformer.layers[i].linear1.out_features})", 
                ha='center', va='center', fontsize=10)
        
        # Add arrow if not last layer
        if i < n_layers - 1:
            ax.arrow(1, y_pos, 0, -vertical_spacing/2, head_width=0.1, 
                    head_length=vertical_spacing/4, fc='black', ec='black')
    
    # Draw output layer
    y_pos -= layer_height + vertical_spacing
    ax.add_patch(Rectangle((0, y_pos), 2, layer_height, 
                          facecolor=colors['output'], edgecolor='black'))
    ax.text(1, y_pos + layer_height/2, f"Linear Output\n({embed_size}, {vocab_size})", 
            ha='center', va='center', fontsize=10)
    
    # Set axis properties
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(y_pos - vertical_spacing, total_height + vertical_spacing)
    ax.axis('off')
    
    # Add title and legend
    plt.title("CuteLLM Model Architecture", fontsize=14)
    
    # Create legend patches
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=colors['embedding'], edgecolor='black', label='Embedding Layer'),
        Rectangle((0, 0), 1, 1, facecolor=colors['attention'], edgecolor='black', label='Self-Attention'),
        Rectangle((0, 0), 1, 1, facecolor=colors['feedforward'], edgecolor='black', label='Feed-Forward'),
        Rectangle((0, 0), 1, 1, facecolor=colors['output'], edgecolor='black', label='Output Layer')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add parameter count
    total_params = sum(p.numel() for p in model.parameters())
    ax.text(1, y_pos - vertical_spacing - 0.2, f"Total Parameters: {total_params:,}", 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Model architecture visualization saved as '{save_path}'")
    return fig

def visualize_logits(model, tokenizer, prompt, top_k=10, save_path="docs/images/logits_visualization.png"):
    """
    Visualize the logits (prediction scores) for each position in the input
    
    Args:
        model: The trained CuteLLM model
        tokenizer: Tokenizer to convert text to token IDs
        prompt: Text prompt to analyze
        top_k: Number of top tokens to show
        save_path: Path to save the visualization
    """
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt)
    input_tokens = prompt.split()
    
    # Get model predictions
    with torch.no_grad():
        logits = model(torch.tensor([input_ids]))
    
    # Convert to probabilities
    probs = torch.softmax(logits[0], dim=-1).numpy()
    
    # Create figure with subplots for each position
    n_positions = min(len(input_ids), 5)  # Limit to 5 positions for clarity
    fig, axes = plt.subplots(n_positions, 1, figsize=(12, 3*n_positions))
    if n_positions == 1:
        axes = [axes]
    
    # For each position, show the top-k predicted next tokens
    for pos in range(n_positions):
        # Get the top-k tokens for this position
        pos_probs = probs[pos]
        top_indices = np.argsort(pos_probs)[-top_k:][::-1]
        top_probs = pos_probs[top_indices]
        
        # Get the token words
        top_tokens = [tokenizer.get_word_from_id(idx) for idx in top_indices]
        
        # Plot horizontal bar chart
        y_pos = np.arange(len(top_tokens))
        axes[pos].barh(y_pos, top_probs, color='skyblue')
        axes[pos].set_yticks(y_pos)
        axes[pos].set_yticklabels(top_tokens)
        axes[pos].set_xlim(0, max(top_probs) * 1.1)
        
        # Add position info
        if pos < len(input_tokens):
            axes[pos].set_title(f"Position {pos+1}: '{input_tokens[pos]}' → Next Token Predictions")
        else:
            axes[pos].set_title(f"Position {pos+1} → Next Token Predictions")
        
        # Add probability values
        for i, v in enumerate(top_probs):
            axes[pos].text(v + 0.01, i, f"{v:.4f}", va='center')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Logits visualization saved as '{save_path}'")
    return fig

def visualize_token_heatmap(model, tokenizer, prompt, save_path="docs/images/token_heatmap.png"):
    """
    Create a heatmap of token prediction probabilities
    
    Args:
        model: The trained CuteLLM model
        tokenizer: Tokenizer to convert text to token IDs
        prompt: Text prompt to analyze
        save_path: Path to save the visualization
    """
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt)
    input_tokens = prompt.split()
    
    # Get model predictions
    with torch.no_grad():
        logits = model(torch.tensor([input_ids]))
    
    # Convert to probabilities
    probs = torch.softmax(logits[0], dim=-1).numpy()
    
    # Select a subset of common tokens for the heatmap
    common_words = ["the", "a", "is", "was", "cat", "dog", "house", "car", "book", 
                   "run", "walk", "big", "small", "good", "bad", "in", "on", "at"]
    token_ids = [tokenizer.word_to_id.get(word, tokenizer.word_to_id["<unk>"]) for word in common_words]
    
    # Create a matrix of probabilities for these tokens at each position
    heatmap_data = probs[:len(input_tokens), token_ids]
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, 
                xticklabels=common_words,
                yticklabels=input_tokens,
                cmap="YlGnBu",
                annot=True,
                fmt=".3f",
                cbar_kws={'label': 'Probability'})
    
    plt.title("Token Prediction Probabilities")
    plt.xlabel("Predicted Tokens")
    plt.ylabel("Input Position")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Token heatmap visualization saved as '{save_path}'")
    return plt.gcf()

def main():
    print("\n" + "="*50)
    print("MODEL VISUALIZATION")
    print("="*50)
    
    # Ensure the docs/images directory exists
    import os
    os.makedirs("docs/images", exist_ok=True)
    
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
    
    # 1. Visualize the model architecture
    print("\nVisualizing model architecture...")
    visualize_model_architecture(model)
    
    # 2. Visualize logits for a sample prompt
    print("\nVisualizing model logits...")
    test_prompt = "the cat sat on the"
    visualize_logits(model, tokenizer, test_prompt)
    
    # 3. Create a token prediction heatmap
    print("\nCreating token prediction heatmap...")
    visualize_token_heatmap(model, tokenizer, test_prompt)
    
    print("\nExplanation:")
    print("- The model architecture diagram shows the layers and connections in the model")
    print("- The logits visualization shows the model's predictions at each position")
    print("- The token heatmap shows prediction probabilities across different tokens")
    print("- These visualizations help understand how the model processes and predicts text")

if __name__ == "__main__":
    main()
