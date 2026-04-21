"""
Temporal Fusion Transformer (TFT) — Variant Model
==================================================
Adapted for fixed-horizon RUL regression on C-MAPSS.

Reference
---------
Bryan Lim et al., "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting," NeurIPS 2020.
https://arxiv.org/abs/1912.09363

Key components
--------------
1. Variable Selection Network (VSN)     — learns which sensors matter most
2. LSTM encoder                         — local temporal state
3. Gated skip connections + LayerNorm  — training stability
4. Multi-head Self-Attention            — long-range dependency modelling
5. Gated Residual Network (GRN)         — context-aware nonlinear processing
6. FC regression head                   — scalar RUL output

Why this differs from the LSTM baseline
----------------------------------------
- Explicit variable selection: the network produces interpretable per-feature
  importance weights, revealing which sensors drive predictions.
- Attention over the full window: the LSTM baseline compresses history into a
  single hidden state; TFT can attend back to any time step.
- Gated residual paths: information bypasses sublayers when not needed,
  reducing vanishing gradients on longer windows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinearUnit(nn.Module):
    """Splits last dim in half; applies element-wise sigmoid gate."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GatedResidualNetwork(nn.Module):
    """
    GRN(x) = LayerNorm(x + GLU(W2 · ELU(W1 · x)))

    Optionally conditions on a context vector c (same shape as x).
    """
    def __init__(self, d_model: int, d_hidden: int = None,
                 dropout: float = 0.1, use_context: bool = False):
        super().__init__()
        d_hidden = d_hidden or d_model
        in_dim = 2 * d_model if use_context else d_model
        self.use_context = use_context
        self.fc1  = nn.Linear(in_dim, d_hidden)
        self.fc2  = nn.Linear(d_hidden, d_model * 2)
        self.glu  = GatedLinearUnit()
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        residual = x
        inp = torch.cat([x, c], dim=-1) if (self.use_context and c is not None) else x
        h = F.elu(self.fc1(inp))
        h = self.drop(self.glu(self.fc2(h)))
        return self.norm(residual + h)


class VariableSelectionNetwork(nn.Module):
    """
    Learns a soft attention weight over input features at each time step,
    then produces a weighted blend of per-feature GRN embeddings.

    Input : (B, T, n_features)
    Output: processed (B, T, d_model), weights (B, T, n_features)
    """
    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_features  = n_features
        self.d_model     = d_model

        self.feature_proj = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(n_features)]
        )
        self.feature_grn = nn.ModuleList(
            [GatedResidualNetwork(d_model, dropout=dropout) for _ in range(n_features)]
        )
        self.weight_net = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_features),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        B, T, F = x.shape

        weights = torch.softmax(self.weight_net(x), dim=-1)   # (B, T, F)

        feat_out = []
        for i in range(self.n_features):
            fi = self.feature_proj[i](x[..., i:i+1])          # (B, T, d_model)
            fi = self.feature_grn[i](fi)
            feat_out.append(fi)

        stacked = torch.stack(feat_out, dim=2)                 # (B, T, F, d_model)
        out = (stacked * weights.unsqueeze(-1)).sum(dim=2)     # (B, T, d_model)
        return self.norm(out), weights


class TemporalFusionTransformer(nn.Module):
    """
    Simplified TFT for C-MAPSS RUL regression.

    Parameters
    ----------
    n_features    : int   number of input channels (17 for C-MAPSS)
    d_model       : int   internal embedding dimension
    n_heads       : int   attention heads (d_model must be divisible by n_heads)
    n_lstm_layers : int   depth of LSTM encoder
    dropout       : float
    """

    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 n_lstm_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.vsn = VariableSelectionNetwork(n_features, d_model, dropout)

        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model,
                            num_layers=n_lstm_layers, batch_first=True,
                            dropout=dropout if n_lstm_layers > 1 else 0.0)
        self.lstm_gate = nn.Linear(d_model, d_model)
        self.lstm_norm = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        self.attn_grn  = GatedResidualNetwork(d_model, dropout=dropout)
        self.attn_norm = nn.LayerNorm(d_model)

        self.ff_grn = GatedResidualNetwork(d_model, d_hidden=d_model * 2, dropout=dropout)

        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_features)  →  (B, 1)"""
        vsn_out, _ = self.vsn(x)

        lstm_out, _ = self.lstm(vsn_out)
        gate     = torch.sigmoid(self.lstm_gate(lstm_out))
        lstm_out = self.lstm_norm(vsn_out + gate * lstm_out)

        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_grn(attn_out)
        attn_out = self.attn_norm(lstm_out + attn_out)

        ff_out = self.ff_grn(attn_out)
        last   = ff_out[:, -1, :]
        return self.head(last)

    def get_variable_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Variable selection weights: (B, T, n_features) — for interpretability plots."""
        _, weights = self.vsn(x)
        return weights


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TemporalFusionTransformer(n_features=17)
    dummy = torch.randn(8, 30, 17)
    out   = model(dummy)
    w     = model.get_variable_weights(dummy)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert out.shape == (8, 1)
    assert w.shape   == (8, 30, 17)
    print(f"✓ Output: {out.shape}  Weights: {w.shape}  Params: {n_p:,}")
    print(f"✓ Weight sum (first sample, first step): {w[0,0].sum():.4f} ≈ 1.0")
