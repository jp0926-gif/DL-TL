import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
#import optuna
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
#import seaborn as sns
from matplotlib import font_manager, rc
import platform
import os
import pandas as pd
import numpy as np
from pykalman import KalmanFilter
import matplotlib.dates as mdates
pd.set_option('display.max_columns', None)
#import seaborn as sns
#import plotly.graph_objects as go
import glob
from sklearn.metrics import r2_score, mean_squared_error

from dltl.common import (
    TimeSeriesDataset,
    create_sequences,
    filter_columns,
    kalman_filter_interpolation,
    log10_1p,
    train_model_with_flag_masking,
)
from dltl.tft import (
    GLU,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    QuantileLoss,
    ScaledDotProductAttention,
    TFT,
    TemporalLayer,
    VariableSelectionNetwork,
)

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

print(torch.cuda.is_available)
print(torch.version.cuda)
print(device)

# Shared preprocessing and training utilities are imported from dltl.common.
