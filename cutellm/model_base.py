import torch
import torch.nn as nn
import logging

class CuteLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        logging.info("Initializing CuteLLM model components")
        
        logging.info(f"Creating embedding layer with vocab_size={config['vocab_size']}, d_model={config['d_model']}")
        self.embed = nn.Embedding(config["vocab_size"], config["d_model"])
        
        logging.info(f"Creating transformer encoder with {config['n_layers']} layers and {config['n_heads']} attention heads")
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config["d_model"],
                nhead=config["n_heads"],
                dim_feedforward=512,
                batch_first=True,  # Add batch_first=True to fix the warning
            ),
            num_layers=config["n_layers"],
        )
        
        logging.info(f"Creating language model head (output layer) with d_model={config['d_model']}, vocab_size={config['vocab_size']}")
        self.lm_head = nn.Linear(config["d_model"], config["vocab_size"])
        
        total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"Model initialized with {total_params} total parameters")

    def forward(self, x):
        logging.debug(f"Forward pass with input shape: {x.shape}")
        
        logging.debug("Step 1: Embedding tokens")
        x = self.embed(x)
        logging.debug(f"Embedded shape: {x.shape}")
        
        logging.debug("Step 2: Passing through transformer layers")
        x = self.transformer(x)
        logging.debug(f"Transformer output shape: {x.shape}")
        
        logging.debug("Step 3: Projecting to vocabulary size with language model head")
        logits = self.lm_head(x)
        logging.debug(f"Output logits shape: {logits.shape}")
        
        return logits