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
	@echo "  run         - Run the model training"
	@echo "  generate    - Generate text with the model"
	@echo "  inspect     - Inspect the model"
	@echo "  test        - Test interactive completion"
	@echo "  test-phrase - Test phrase completion"
	@echo "  evaluate    - Evaluate phrase completion"
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