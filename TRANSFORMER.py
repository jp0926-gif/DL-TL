import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd
import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
pd.set_option('display.max_columns', None)
#import seaborn as sns
import os
import plotly.graph_objects as go
import glob
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from datetime import timedelta

from dltl.baseline import train_model_with_flag_masking
from dltl.common import (
    TimeSeriesDataset,
    create_sequences_with_flags,
    kalman_filter_interpolation,
    log10_1p,
)

# Set the font properties
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
#### Set seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available)
print(torch.version.cuda)
print(device)

import torch
import torch.nn as nn
import numpy as np
import math

# Positional Encoding for Sequential Data
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_encoder_layers=4, dim_feedforward=256, dropout=0.1, output_dim=1):
        super(TransformerTimeSeries, self).__init__()

        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, src):
        """
        src: (batch_size, seq_len, input_dim) - Input time-series data
        """
        src = self.input_embedding(src)
        src = self.positional_encoding(src)

        src = src.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        encoded_output = self.transformer_encoder(src)

        output = self.output_layer(encoded_output[-1])  # Take last time-step representation
        return output
    
# Shared preprocessing and training utilities are imported from dltl.
