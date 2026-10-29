"""CNN-LSTM-Attention baseline model."""

import torch
import torch.nn as nn


class CLA(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        output_dim,
        cnn_filters=64,
        kernel_size=3,
        dropout=0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv1 = nn.Conv1d(
            input_dim, cnn_filters, kernel_size=kernel_size, padding=1
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(
            cnn_filters,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def attention(self, lstm_output):
        scores = self.attention_weights(lstm_output)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_output, dim=1)
        return context, weights

    def forward(self, inputs):
        features = self.pool(self.relu(self.conv1(inputs.permute(0, 2, 1))))
        features = features.permute(0, 2, 1)
        h0 = torch.zeros(self.num_layers, features.size(0), self.hidden_dim).to(
            features.device
        )
        c0 = torch.zeros(self.num_layers, features.size(0), self.hidden_dim).to(
            features.device
        )
        lstm_output, _ = self.lstm(features, (h0, c0))
        context, weights = self.attention(lstm_output)
        return self.fc(self.dropout(context)), weights


def is_valid_combination(seq_len, kernel_size, pooling_size=2):
    padding = (kernel_size - 1) // 2
    effective_seq_len = seq_len + 2 * padding - (kernel_size - 1)
    pooled_seq_len = effective_seq_len // pooling_size
    return effective_seq_len > 0 and pooled_seq_len > 0
