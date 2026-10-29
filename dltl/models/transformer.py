"""Transformer baseline for time-series forecasting."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, inputs):
        return inputs + self.pe[:, : inputs.size(1), :]


class TransformerTimeSeries(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=64,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        output_dim=1,
    ):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, inputs):
        embedded = self.input_embedding(inputs)
        encoded = self.transformer_encoder(self.positional_encoding(embedded))
        return self.output_layer(encoded[:, -1, :])
