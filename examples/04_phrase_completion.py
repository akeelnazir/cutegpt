"""
Phrase Completion Example with CuteLLM

This example demonstrates how to use CuteLLM for phrase completion tasks.
It shows how to use the mask token to fill in blanks in sentences.
"""

import torch
import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cutellm.model_base import CuteLLM
from cutellm.tokenizer import SimpleWordTokenizer

def complete_phrase(model, tokenizer, phrase, top_k=5):
    """
    Complete a phrase with a masked token (represented by ___)
    
    Args:
        model: The trained CuteLLM model
        tokenizer: Tokenizer to convert text to token IDs
        phrase: The phrase with a ___ where a word should be predicted
        top_k: Number of top predictions to return
        
    Returns:
        List of (word, probability) tuples for the top predictions
    """
    # Tokenize the phrase
    input_ids = tokenizer.encode(phrase)
    
    # Find the position of the mask token
    mask_pos = tokenizer.get_mask_position(input_ids)
    if mask_pos == -1:
        print("Error: No mask token (___ or <mask>) found in the phrase")
        return []
    
    # Get model predictions
    with torch.no_grad():
        logits = model(torch.tensor([input_ids]))
        
    # Get probabilities for the mask position
    probs = torch.softmax(logits[0, mask_pos], dim=-1)
    
    # Get top-k predictions
    top_probs, top_indices = torch.topk(probs, top_k)
    
    # Convert to words and probabilities
    results = []
    for i in range(top_k):
        word_id = top_indices[i].item()
        word = tokenizer.get_word_from_id(word_id)
        probability = top_probs[i].item()
        results.append((word, probability))
    
    return results

def main():
    print("\n" + "="*50)
    print("PHRASE COMPLETION EXAMPLE")
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
    
    # Example phrases with blanks to fill
    test_phrases = [
        "the cat sat on the ___",
        "the ___ was shining bright",
        "she ___ to the store",
        "they played in the ___"
    ]
    
    print("\nPredicting words to fill in the blanks:")
    
    for phrase in test_phrases:
        print(f"\nPhrase: '{phrase}'")
        predictions = complete_phrase(model, tokenizer, phrase, top_k=5)
        
        if predictions:
            print("Top predictions:")
            for i, (word, prob) in enumerate(predictions):
                print(f"  {i+1}. '{word}' ({prob:.4f})")
            
            # Show the top prediction in context
            best_word = predictions[0][0]
            completed = phrase.replace("___", best_word)
            print(f"\nBest completion: '{completed}'")
    
    print("\nExplanation:")
    print("- The model predicts the most likely word to replace the ___ token")
    print("- Predictions are ranked by probability")
    print("- This is similar to how masked language modeling works in larger models")
    print("- The quality of predictions depends on the training data and model size")

if __name__ == "__main__":
    main()
