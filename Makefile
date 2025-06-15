# Default target that displays help information
.PHONY: help
help:
	@echo "CuteGPT - A custom small LLM for educational purposes"
	@echo ""
	@echo "Available targets:"
	@echo "  help        - Display this help message"
	@echo "  venv        - Create a virtual environment"
	@echo "  setup       - Complete setup for development (creates venv, installs dependencies)"
	@echo "  install     - Install the package and its dependencies"
	@echo "  dev         - Install the package in development mode"
	@echo "  run         - Train the model"
	@echo "  generate    - Generate text using the trained model"
	@echo "  inspect     - Inspect the model architecture"
	@echo "  visualize   - Generate model visualizations"
	@echo "  test        - Run tests for interactive completion"
	@echo "  test-phrase - Test phrase completion"
	@echo "  evaluate    - Evaluate phrase completion"
	@echo ""
	@echo "Example commands:"
	@echo "  examples    - Run all examples"
	@echo "  example1    - Run basic text generation example"
	@echo "  example2    - Run interactive completion example"
	@echo "  example3    - Run custom fine-tuning example"
	@echo "  example4    - Run phrase completion example"
	@echo "  example5    - Run model inspection example"
	@echo "  visualize   - Run model visualization script"
	@echo "  examples    - Run all example scripts"
	@echo ""

# Create virtual environment
.PHONY: venv
venv:
	python3 -m venv .venv
	@echo "Virtual environment created. Activate it with: source .venv/bin/activate.fish"

# Complete setup for development
.PHONY: setup
setup: venv install dev
	@echo "Development environment setup complete!"

# Install the package and its dependencies
.PHONY: install
install:
	@echo "Installing dependencies..."
	@.venv/bin/pip install --upgrade pip setuptools wheel
	@.venv/bin/pip install -r requirements.txt || (echo "Error installing dependencies. Please check requirements.txt for compatibility issues."; exit 1)

# Install the package in development mode
.PHONY: dev
dev:
	.venv/bin/pip install -e .

# Run the model training
.PHONY: run
run:
	.venv/bin/python main.py

# Generate text with the model
.PHONY: generate
generate:
	.venv/bin/python scripts/generate_text.py

# Inspect the model
.PHONY: inspect
inspect:
	.venv/bin/python scripts/inspect_model.py

# Test interactive completion
.PHONY: test
test:
	.venv/bin/python tests/test_completion.py

# Test phrase completion
.PHONY: test-phrase
test-phrase:
	.venv/bin/python tests/test_phrase_completion.py

# Evaluate phrase completion
.PHONY: evaluate
evaluate:
	.venv/bin/python scripts/evaluate_phrase_completion.py

# Example commands
# Run all examples
.PHONY: examples
examples: example1 example2 example3 example4 example5
	@echo "All examples have been run!"

# Run basic text generation example
.PHONY: example1
example1:
	@echo "Running basic text generation example..."
	.venv/bin/python examples/01_basic_text_generation.py

# Run interactive completion example
.PHONY: example2
example2:
	@echo "Running interactive completion example..."
	.venv/bin/python examples/02_interactive_completion.py

# Run custom fine-tuning example
.PHONY: example3
example3:
	@echo "Running custom fine-tuning example..."
	.venv/bin/python examples/03_custom_fine_tuning.py

# Run phrase completion example
.PHONY: example4
example4:
	@echo "Running phrase completion example..."
	.venv/bin/python examples/04_phrase_completion.py

# Run model inspection example
.PHONY: example5
example5:
	@echo "Running model inspection example..."
	.venv/bin/python examples/05_model_inspection.py

# Run model visualization script
.PHONY: visualize
visualize:
	@echo "Running model visualization script..."
	.venv/bin/python scripts/model_visualization.py