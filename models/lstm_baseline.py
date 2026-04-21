"""
LSTM Baseline Model
===================
Stacked LSTM with dropout for C-MAPSS RUL regression.
Covered in lecture → eligible as the course baseline.
"""

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """
    Stacked LSTM → FC head for RUL regression.

    Architecture
    ------------
    Input  : (batch, seq_len, n_features)
    LSTM   : num_layers stacked LSTM, hidden_size units each
    Dropout: between layers and before head
    FC head: hidden_size → 64 → 1
    Output : (batch, 1)  scalar RUL prediction
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, n_features)
        returns : (batch, 1)
        """
        out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        last = out[:, -1, :]           # take final time step
        return self.head(last)         # (batch, 1)


if __name__ == "__main__":
    model = LSTMBaseline(n_features=17)
    dummy = torch.randn(8, 30, 17)
    print("Output shape:", model(dummy).shape)   # (8, 1)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
