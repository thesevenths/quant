import math
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention


def get_causal_mask(seq_len, device):
    # Generate upper-triangular mask for causal attention
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers,
                 output_dim, dropout=0.1, max_len=5000):
        super().__init__()
        # Input projection and normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Positional encoding
        self.pos_encoder = PositionalEncoding(model_dim, max_len)

        # Transformer encoder with dropout and causal mask
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(model_dim)
        )

        # Final regression head
        self.fc = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
            nn.Linear(model_dim, output_dim)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        # Create causal mask
        seq_len = x.size(1)
        mask = get_causal_mask(seq_len, x.device)
        # Transformer encoder
        x = self.transformer_encoder(x, mask=mask)
        # Pooling: use last time step
        x = x[:, -1, :]
        out = self.fc(x)
        return out

    def get_attention(self, module, input, output):
        print("Output:", output)
        if len(output) > 1:
            self.attn_weights = output[1]
        else:
            self.attn_weights = None
