import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        """
        Initializes the attention pooling mechanism

        Args:
            d_model:
        """
        super().__init__()

        # Defines a learnable attention vector
        self.attention_vector = nn.Parameter(torch.randn(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for attention pooling

        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            [batch_size, d_model]
        """
        # Calculates raw attention scores
        attn_scores = torch.matmul(x, self.attention_vector)  # [batch_size, seq_len]

        # Applies softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # [batch_size, seq_len, 1]

        # Applies weighted sum over the sequence dimension
        pooled = torch.sum(x * attn_weights, dim=1)  # [batch_size, d_model]

        return pooled
