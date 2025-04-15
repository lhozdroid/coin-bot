import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initializes the positional encoding

        Args:
            d_model:
            max_len:
        """
        super().__init__()

        # Creates the positional encoding full of zeroes [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # Generates positions [max_len, 1]
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # Computes the frequency scaling factor [d_model/2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Applies sine to even dimensions [max_len, d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)

        # Applies cosine to odd dimensions [max_len, d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)

        # Adds the batch dimension [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward processing

        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len].to(x.device)
        return x
