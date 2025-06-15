# CuteLLM Examples

This directory contains example implementations to help you understand how to use the CuteLLM model. These examples are designed for educational purposes and demonstrate various aspects of working with a small language model.

## Available Examples

1. **01_basic_text_generation.py**
   - Demonstrates how to load a trained model and generate text
   - Shows both greedy decoding and sampling with different temperatures
   - Great starting point for understanding the basics

2. **02_interactive_completion.py**
   - Provides an interactive command-line interface for text completion
   - Allows you to experiment with different prompts and generation settings
   - Useful for exploring the model's capabilities in real-time

3. **03_custom_fine_tuning.py**
   - Shows how to fine-tune the model on your own custom data
   - Demonstrates creating a custom dataset and training process
   - Explains how to save and test the fine-tuned model

4. **04_phrase_completion.py**
   - Demonstrates how to use the model for filling in blanks in sentences
   - Shows how to use the mask token for word prediction tasks
   - Provides probability scores for different word predictions

5. **05_model_inspection.py**
   - Helps you understand the inner workings of the model
   - Visualizes word embeddings and attention patterns
   - Provides insights into the model's parameters and architecture

## Running the Examples

Before running these examples, make sure:

1. You have installed all required dependencies (`pip install -r ../requirements.txt`)
2. You have trained the model or have a pre-trained model file at `models/cute_llm.pth`

To run any example, use:

```bash
python examples/01_basic_text_generation.py
```

## Learning Path

If you're new to language models, we recommend following this learning path:

1. Start with `01_basic_text_generation.py` to understand the basics
2. Try `02_interactive_completion.py` to experiment with the model
3. Explore `04_phrase_completion.py` to see a different use case
4. Use `05_model_inspection.py` to peek inside the model
5. Finally, try `03_custom_fine_tuning.py` to adapt the model to your own data

## Additional Resources

For more information about the CuteLLM model:
- Check the main README.md file in the project root
- Look at the implementation in the `cutellm/` directory
- Explore the training data in the `data/` directory

Happy learning!
