import torch
import torch.nn as nn

class CuteLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config["vocab_size"], config["d_model"])
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config["d_model"],
                nhead=config["n_heads"],
                dim_feedforward=512,
                batch_first=True,  # Add batch_first=True to fix the warning
            ),
            num_layers=config["n_layers"],
        )
        self.lm_head = nn.Linear(config["d_model"], config["vocab_size"])

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        return self.lm_head(x)