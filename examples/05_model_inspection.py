"""
Model Inspection Example with CuteLLM

This example demonstrates how to inspect the internals of a trained CuteLLM model.
It shows how to visualize the embeddings, attention patterns, and other model components.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cutellm.model_base import CuteLLM
from cutellm.tokenizer import SimpleWordTokenizer

def visualize_embeddings(model, tokenizer, words):
    """
    Visualize word embeddings for a list of words
    
    Args:
        model: The trained CuteLLM model
        tokenizer: Tokenizer to convert text to token IDs
        words: List of words to visualize
    """
    # Get word IDs
    word_ids = [tokenizer.word_to_id.get(word, tokenizer.word_to_id["<unk>"]) for word in words]
    
    # Get embeddings
    with torch.no_grad():
        embeddings = model.embed(torch.tensor(word_ids)).numpy()
    
    # Use PCA to reduce dimensionality to 2D for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Plot the embeddings
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], marker='o')
    
    # Add word labels
    for i, word in enumerate(words):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    
    plt.title("Word Embeddings Visualization (2D PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("word_embeddings.png")
    print(f"Embeddings visualization saved as 'word_embeddings.png'")

def analyze_attention(model, tokenizer, sentence):
    """
    Analyze and visualize attention patterns for a sentence
    
    Args:
        model: The trained CuteLLM model
        tokenizer: Tokenizer to convert text to token IDs
        sentence: Sentence to analyze
    """
    # Tokenize the sentence
    input_ids = tokenizer.encode(sentence)
    input_tensor = torch.tensor([input_ids])
    
    # Get the words for visualization
    words = sentence.split()
    
    # Forward pass with attention hooks
    attentions = []
    
    def get_attention(module, input, output):
        # Extract attention weights from the output
        # This assumes the attention weights are available in the output tuple
        attention = output[1] if isinstance(output, tuple) else None
        if attention is not None:
            attentions.append(attention.detach())
    
    # Register hooks to capture attention weights
    hooks = []
    for name, module in model.named_modules():
        if "self_attn" in name:
            hook = module.register_forward_hook(get_attention)
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # If we captured attention weights, visualize them
    if attentions:
        print(f"Captured {len(attentions)} attention layers")
        
        # Plot attention heatmaps
        fig, axes = plt.subplots(len(attentions), 1, figsize=(10, len(attentions) * 3))
        if len(attentions) == 1:
            axes = [axes]
        
        for i, attn in enumerate(attentions):
            # Get attention weights for the first head of the first batch
            attn_weights = attn[0, 0].numpy()
            
            # Plot heatmap
            im = axes[i].imshow(attn_weights, cmap="viridis")
            axes[i].set_title(f"Attention Layer {i+1}")
            
            # Set x and y labels to be the words
            if len(words) <= len(input_ids):
                axes[i].set_xticks(np.arange(len(input_ids)))
                axes[i].set_yticks(np.arange(len(input_ids)))
                axes[i].set_xticklabels(words)
                axes[i].set_yticklabels(words)
            
            # Rotate x labels for better readability
            plt.setp(axes[i].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add colorbar
            fig.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.savefig("attention_patterns.png")
        print(f"Attention visualization saved as 'attention_patterns.png'")
    else:
        print("No attention weights were captured. This might be due to the model architecture.")

def main():
    print("\n" + "="*50)
    print("MODEL INSPECTION EXAMPLE")
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
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Visualize word embeddings
    print("\nVisualizing word embeddings...")
    words_to_visualize = [
        "cat", "dog", "house", "car", "book", 
        "run", "walk", "eat", "sleep", "talk",
        "big", "small", "good", "bad", "happy"
    ]
    try:
        visualize_embeddings(model, tokenizer, words_to_visualize)
    except Exception as e:
        print(f"Error visualizing embeddings: {e}")
        print("You may need to install scikit-learn: pip install scikit-learn")
    
    # Analyze attention patterns
    print("\nAnalyzing attention patterns...")
    test_sentence = "the cat sat on the mat"
    try:
        analyze_attention(model, tokenizer, test_sentence)
    except Exception as e:
        print(f"Error analyzing attention: {e}")
        print("This might be due to the model architecture or missing matplotlib.")
    
    print("\nExplanation:")
    print("- Word embeddings show how words are represented in the model's vector space")
    print("- Similar words should be closer together in the embedding space")
    print("- Attention patterns show which words the model focuses on when processing text")
    print("- Brighter colors in the attention heatmap indicate stronger attention")

if __name__ == "__main__":
    main()
