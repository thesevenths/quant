
import torch.nn as nn
from torch.nn import MultiheadAttention

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_dim)
        self.attn = MultiheadAttention(model_dim, num_heads, batch_first=True)
        self.attn_weights = None

    def forward(self, x):
        x = self.input_proj(x)
        # Custom attention application
        attn_output, attn_weights = self.attn(x, x, x)  # Self-attention
        self.attn_weights = attn_weights
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

    def get_attention(self, module, input, output):
        print("Output:", output)
        if len(output) > 1:
            self.attn_weights = output[1]
        else:
            self.attn_weights = None