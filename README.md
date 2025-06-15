# CuteGPT

A custom small LLM for understanding the process of creating, training and using. This project demonstrates the basic concepts of language model creation, training, and inference in a simplified form.

## Overview

CuteGPT is a minimal implementation of a transformer-based language model (CuteLLM) that can be trained on simple text data and used for basic text generation and phrase completion tasks. It's designed to be educational rather than practical, focusing on clarity and simplicity over performance.

## Project Structure

```
cutellm/            # Core model implementation
  ├── __init__.py   # Package exports
  ├── model_base.py # CuteLLM model definition
  ├── inference_base.py # Text generation functions
  ├── tokenizer.py  # Tokenization functionality
  └── training_base.py  # Training functions

data/               # Training data
  └── training_data.txt # Simple text for training

examples/           # Example implementations

models/             # Saved model weights
  └── cute_llm.pth  # Trained model weights

scripts/            # Utility scripts
  ├── generate_text.py          # Text generation demo
  ├── inspect_model.py          # Model inspection tool
  └── evaluate_phrase_completion.py # Evaluation script

tests/              # Test scripts
  ├── test_completion.py        # Interactive phrase completion test
  └── test_phrase_completion.py # Basic phrase completion test

main.py            # Main entry point
setup.py           # Package setup script
```

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/akeelnazir/cutegpt.git
cd cutegpt

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Training the Model

```bash
# Train the model using the provided training data
make run
```

### Testing Phrase Completion

```bash
# Test the model's phrase completion abilities
make test
```

### Generating Text

```bash
# Generate text using the trained model
python scripts/generate_text.py
```

## Makefile Commands

- `make run`: Train the model
- `make generate`: Generate text using the trained model
- `make inspect`: Inspect the model architecture
- `make test`: Test phrase completion

## Educational Value

This project demonstrates:
- Basic transformer model architecture
- Simple tokenization for NLP tasks
- Training loop implementation
- Text generation with and without temperature sampling
- Phrase completion testing

## License

This project is for educational purposes only.
