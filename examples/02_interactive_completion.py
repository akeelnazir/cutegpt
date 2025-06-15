"""
Interactive Text Completion Example with CuteLLM

This example provides an interactive command-line interface where users can
type prompts and see the model complete them in real-time. It's a great way
to experiment with the model's capabilities.
"""

import torch
import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cutellm.model_base import CuteLLM
from cutellm.tokenizer import SimpleWordTokenizer
from cutellm.inference_base import sample_generate

def main():
    print("\n" + "="*50)
    print("INTERACTIVE TEXT COMPLETION")
    print("="*50)
    print("\nLoading CuteLLM model...")
    
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
    
    print("\nCuteLLM Interactive Text Completion")
    print("Type a prompt and press Enter to see the model complete it.")
    print("Type 'exit' to quit.")
    print("Type 'settings' to adjust generation parameters.")
    
    # Default generation settings
    max_len = 15
    temperature = 0.8
    
    while True:
        print("\n" + "-"*50)
        prompt = input("Your prompt: ")
        
        if prompt.lower() == 'exit':
            print("Goodbye!")
            break
        
        if prompt.lower() == 'settings':
            try:
                max_len = int(input("Max tokens to generate (current: {}): ".format(max_len)) or max_len)
                temperature = float(input("Temperature (current: {}): ".format(temperature)) or temperature)
                print(f"Settings updated: max_len={max_len}, temperature={temperature}")
            except ValueError:
                print("Invalid input. Using previous settings.")
            continue
        
        if not prompt:
            continue
        
        print("\nGenerating completion...")
        
        # Generate text with the current settings
        generated_text = sample_generate(model, tokenizer, prompt, max_len=max_len, temperature=temperature)
        
        # Highlight the prompt in the generated text
        highlighted_text = generated_text
        if generated_text.startswith(prompt):
            completion = generated_text[len(prompt):]
            highlighted_text = f"{prompt}[{completion}]"
        
        print(f"Completion: {highlighted_text}")

if __name__ == "__main__":
    main()
