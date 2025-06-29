import math
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution


def get_causal_mask(seq_len, device):
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class BTCTransformer(nn.Module):
    def __init__(self, input_dim=7, d_model=32, nhead=4, nlayers=1, action_dim=3, dropout=0.1, max_len=5000):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(  # 单个transformer block的结构
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(  # transformer block个数
            encoder_layer,
            num_layers=nlayers,
            norm=nn.LayerNorm(d_model)
        )

        self.action_head = nn.Linear(d_model, action_dim)  # action output
        self.value_head = nn.Linear(d_model, 1)  # critical output

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        seq_len = x.size(1)
        mask = get_causal_mask(seq_len, x.device)
        x = self.transformer_encoder(x, mask=mask)
        x = x[:, -1, :]  # Take the last timestep's features: (batch, d_model)
        logits = self.action_head(x)  # (batch, action_dim)
        value = self.value_head(x).squeeze(-1)  # (batch,)
        return logits, value  # actor和critical共享底层网络（主要是提取state信息），在head这里分叉


class TransformerPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, input_dim=5, d_model=32, nhead=4, nlayers=1,
                 dropout=0.1, max_len=5000, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, ortho_init=False, **kwargs)
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        self.dropout = dropout
        self.max_len = max_len
        # Disable default extractors
        self.features_extractor = None
        self.mlp_extractor = None
        self._build_network()

    def _build_network(self):
        self.transformer = BTCTransformer(
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            nlayers=self.nlayers,
            action_dim=self.action_space.n,
            dropout=self.dropout,
            max_len=self.max_len
        )
        self.action_net = self.transformer.action_head
        self.value_net = self.transformer.value_head
        self.action_dist = CategoricalDistribution(self.action_space.n)

    def forward(self, obs, deterministic=False):
        # Ensure obs is in correct shape: (batch, seq_len, input_dim)
        if len(obs.shape) == 2:  # (batch, seq_len * input_dim)
            batch_size = obs.shape[0]
            obs = obs.view(batch_size, -1, self.input_dim)  # Reshape to (batch, seq_len, input_dim)
        logits, value = self.transformer(obs)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        actions = distribution.get_actions(deterministic=deterministic)
        # Ensure actions is at least 1D, even for batch_size=1
        if actions.dim() == 0:
            actions = actions.unsqueeze(0)  # Add batch dimension if scalar
        log_prob = distribution.log_prob(actions)
        return actions, value, log_prob

    def get_distribution(self, obs):
        # Override to use transformer directly
        if len(obs.shape) == 2:
            batch_size = obs.shape[0]
            obs = obs.view(batch_size, -1, self.input_dim)
        logits, _ = self.transformer(obs)
        return self.action_dist.proba_distribution(action_logits=logits)

    def extract_features(self, obs):
        if len(obs.shape) == 2:
            batch_size = obs.shape[0]
            obs = obs.view(batch_size, -1, self.input_dim)
        logits, _ = self.transformer(obs)
        return logits

    def evaluate_actions(self, obs, actions):
        if len(obs.shape) == 2:
            batch_size = obs.shape[0]
            obs = obs.view(batch_size, -1, self.input_dim)
        logits, values = self.transformer(obs)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        log_prob = distribution.log_prob(actions)  # 计算new/old policy ratio，更新policy gradient
        entropy = distribution.entropy() # 高entropy高exploration，加入loss防止policy过早收敛
        return values, log_prob, entropy #

    def predict_values(self, obs):
        if len(obs.shape) == 2:
            batch_size = obs.shape[0]
            obs = obs.view(batch_size, -1, self.input_dim)
        _, values = self.transformer(obs)  # 用于更新critical net
        return values
