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
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from matplotlib import font_manager, rc
import os
import pandas as pd
import numpy as np
from pykalman import KalmanFilter
pd.set_option('display.max_columns', None)
import glob

# Set the font properties
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.unicode_minus'] = False  # 한글 폰트 사용시 마이너스 기호 깨짐 방지

#### Set seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CLA(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, cnn_filters=64, kernel_size=3, dropout=0.0):
        super(CLA, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # CNN for feature extraction
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=cnn_filters, kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        # LSTM for temporal sequence modeling
        self.lstm = nn.LSTM(cnn_filters, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # Attention mechanism
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def attention(self, lstm_output):
        """
        Attention mechanism: Compute attention weights and context vector.
        lstm_output: [batch_size, seq_len, hidden_dim]
        """
        attention_scores = self.attention_weights(lstm_output)  # [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)  # Normalize scores
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # Weighted sum
        return context_vector, attention_weights
    
    def forward(self, x):
        # Reshape for CNN: [batch_size, input_dim, seq_len]
        x = x.permute(0, 2, 1)

        # Apply CNN layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Reshape for LSTM: [batch_size, seq_len, features]
        x = x.permute(0, 2, 1)

        # Initialize LSTM hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Apply LSTM
        lstm_output, _ = self.lstm(x, (h0, c0))  # lstm_output: [batch_size, seq_len, hidden_dim]

        # Apply attention mechanism
        context_vector, attention_weights = self.attention(lstm_output)

        # Apply dropout and pass through fully connected layer
        context_vector = self.dropout(context_vector)
        out = self.fc(context_vector)
        return out, attention_weights
    
# Shared preprocessing utilities are imported from dltl.common.

def train_model_with_flag_masking(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, EPOCHS, random_seed=42, patience=50):
    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    best_validation_loss = float('inf')
    no_improve = 0                                                
    last_lr = optimizer.param_groups[0]['lr']     
                    
    for epoch in range(1, EPOCHS + 1):  # Epoch starts from 1 for readability
        # Training loop
        model.train()
        total_train_loss = 0
        count_actual_values_train = 0
        for batch_idx, (train_inputs, train_targets, train_flags) in enumerate(train_dataloader):
            optimizer.zero_grad()
            train_outputs, _ = model(train_inputs)
            
            # Element-wise masking for training
            mask = train_flags == 0
            mask = mask.expand_as(train_outputs)  # Ensure mask matches train_outputs shape
            train_outputs_masked = train_outputs * mask.float()
            train_targets_masked = train_targets * mask.float()

            # If valid, calculate the training loss
            if mask.sum() > 0:
                train_loss = criterion(train_outputs_masked, train_targets_masked)
                train_loss.backward()
                optimizer.step()

                # Accumulate total training loss for logging purposes
                total_train_loss += train_loss.item()
                count_actual_values_train += 1

        # Calculate average training loss
        avg_train_loss = total_train_loss / count_actual_values_train if count_actual_values_train > 0 else float('inf')

        # Validation loop
        model.eval()
        validation_loss = 0
        count_actual_values = 0
        with torch.no_grad():
            for batch_idx, (valid_inputs, valid_targets, valid_flags) in enumerate(valid_dataloader):
                valid_outputs, _ = model(valid_inputs)
                
                # Element-wise masking for validation
                mask = valid_flags == 0
                mask = mask.expand_as(valid_outputs)  # Ensure mask matches valid_outputs shape
                valid_outputs_masked = valid_outputs * mask.float()
                valid_targets_masked = valid_targets * mask.float()

                # If valid, calculate the validation loss
                if mask.sum() > 0:
                    valid_loss = criterion(valid_outputs_masked, valid_targets_masked)
                    validation_loss += valid_loss.item()
                    count_actual_values += 1

        # Calculate average validation loss
        if count_actual_values > 0:
            validation_loss /= count_actual_values
        
        # Print progress every epoch
        current_lr = optimizer.param_groups[0]['lr']
        if epoch % 5 == 0 or current_lr != last_lr:
            print(f"Epoch {epoch}/{EPOCHS}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {validation_loss:.4f}, LR: {current_lr:.6f}")
        
        # Check if learning rate has changed
        if current_lr != last_lr:
            print(f"Learning rate changed to {current_lr:.6f} at epoch {epoch}")
            last_lr = current_lr

        scheduler.step(validation_loss)

        # Early stopping check
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch}, best validation loss: {best_validation_loss:.4f}')
            return model

    return model

def is_valid_combination(seq_len, kernel_size, pooling_size=2):
    padding = (kernel_size - 1) // 2
    effective_seq_len = seq_len + 2 * padding - (kernel_size - 1)
    pooled_seq_len = effective_seq_len // pooling_size
    return effective_seq_len > 0 and pooled_seq_len > 0
