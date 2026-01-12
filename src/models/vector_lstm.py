import torch
from torch import nn


class VectorLSTM(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 512, num_layers: int = 1) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.activation = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        x_flat = x.view(b * t, d)
        h = self.encoder(x_flat)
        h = self.activation(h)
        h = h.view(b, t, -1)
        out, _ = self.lstm(h)
        out_flat = out.contiguous().view(b * t, -1)
        y_flat = self.decoder(out_flat)
        y = y_flat.view(b, t, d)
        return y

