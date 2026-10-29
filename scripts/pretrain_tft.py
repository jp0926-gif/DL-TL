import os
import glob
import pandas as pd
import numpy as np
from pykalman import KalmanFilter
import matplotlib.dates as mdates
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from matplotlib import font_manager, rc

from dltl.common import (
    TimeSeriesDataset,
    create_sequences,
    filter_columns,
    kalman_filter_interpolation,
    log10_1p,
    train_model_with_flag_masking,
)
from dltl.config import load_experiment_config_from_cli
from dltl.models.tft import (
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
pd.set_option('display.max_columns', None)
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

# Load paths and experiment parameters from JSON.
config = load_experiment_config_from_cli()
data_config = config["data"]
training_config = config["training"]
river_data_info = data_config["river_data_info"]
Input_Variables = data_config["input_variables"]
quantiles = training_config["quantiles"]
batch_size = training_config["batch_size"]
Input_Sequence = training_config["input_sequence"]
Output_Sequence = training_config["output_sequence"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 반복문을 통해 각 데이터프레임에 대해 모델 훈련 및 저장
Sc_tr_x = []
Sc_tr_y = []
Sc_vl_x = []
Sc_vl_y = []
Sc_te_x = []
Sc_te_y = []
Sc_tr_flags = []
Sc_vl_flags = []
Sc_te_flags = []

# Loop over each river and its locations
for river_name, river_info in river_data_info.items():
  
    for location_name, filepath in river_info['locations'].items():
        print(f"Starting training for River: {river_name}, Location: {location_name}")

        df = pd.read_csv(
            filepath,
            encoding=data_config["encoding"],
            index_col=data_config["index_column"],
        )
        
        df.index = pd.to_datetime(df.index).normalize()

        interpolated_data, interpolation_flags = kalman_filter_interpolation(df)
        # Concatenate interpolated data and flags for processing
        temp_df = filter_columns(interpolated_data, Input_Variables)
        flags_df = filter_columns(interpolation_flags, Input_Variables)
        temp_df2 = temp_df.copy()

        temp_df2.loc[:, 'Month'] = temp_df2.index.month
        temp_df2.loc[:, 'season'] = temp_df2['Month'].apply(lambda x: '여름' if x in [6, 7, 8, 9, 10] else '이외')
        temp_df2=temp_df2.drop(columns=['Month'])
        transform_df = temp_df2.copy()
        
        # Convert categprocal variables
        categorical_scalers = {}
        categorical_columns = ['season']
        category_counts = {'season':2}
        
        for col in categorical_columns:
            srs = transform_df[col].apply(str) 
            categorical_scalers[col] = LabelEncoder().fit(srs.values)
            transform_df[col] = categorical_scalers[col].transform(transform_df[col].apply(str))

        # Create input
        new_input_columns = []
        for col in temp_df2.columns:
            if col == 'date':
                continue
            new_input_columns.append(col)
            
        # select where you want to put your inputs in. If not categoricals, variables go to time dependent continuous with the unknown variables.    
        static_covariates = ['season']
        time_dependent_categorical = []
        time_dependent_continuous = []

        for col in new_input_columns:
            if col in static_covariates or col in time_dependent_categorical:
                continue
            time_dependent_continuous.append(col)
        
        # Select known time-dependent variables as columns containing 'Forecast'
        known_time_dependent = [col for col in new_input_columns if 'Forecast' in col or col == 'Month']
        
        # Select observed time-dependent variables
        observed_time_dependent = []
        for col in new_input_columns:
            if col in static_covariates or col in known_time_dependent:
                continue
            observed_time_dependent.append(col)
        
        col_to_idx = {col: idx for idx, col in enumerate(new_input_columns)}  
        
        # Select output variable
        out_var = [col for col in temp_df2.columns if 'Cyanocell' in col][0]
        Output_Variables = [out_var]
        target_column = out_var
        X = transform_df[new_input_columns]
        y = transform_df[Output_Variables]
        
        # Create dataset
        start_date = transform_df.index.min()
        X.index = X.index.normalize()
        y.index = y.index.normalize()

        X_train = X.truncate(before=start_date, after='2021-12-31')
        y_train = y.truncate(before=start_date, after='2021-12-31')
        flags_train = flags_df.truncate(before=start_date, after='2021-12-31')

        X_valid = X.truncate(before='2022-01-01', after='2022-12-31')
        y_valid = y.truncate(before='2022-01-01', after='2022-12-31')
        flags_valid = flags_df.truncate(before='2022-01-01', after='2022-12-31')

        X_test = X.truncate(before='2023-01-01', after='2023-12-31')
        y_test = y.truncate(before='2023-01-01', after='2023-12-31')
        flags_test = flags_df.truncate(before='2023-01-01', after='2023-12-31')

        real_columns = [col for col in X.columns if any(var in col for var in Input_Variables)]
        other_columns = [col for col in X.columns if col not in real_columns]
        
        X_train_real = X_train[real_columns]
        X_train_other = X_train[other_columns]
        X_valid_real = X_valid[real_columns]
        X_valid_other = X_valid[other_columns]
        X_test_real = X_test[real_columns]
        X_test_other = X_test[other_columns]
        
        ss_train = MinMaxScaler()
        scaled_y_train = log10_1p(y_train.apply(pd.to_numeric, errors='coerce'))
        X_train_real.loc[:, out_var] = log10_1p(X_train_real[out_var])
        scaled_X_train_real = ss_train.fit_transform(X_train_real)
        scaled_X_train_real = pd.DataFrame(scaled_X_train_real, index = X_train_real.index, columns=X_train_real.columns)
        X_train_scaled = pd.concat([scaled_X_train_real, X_train_other], axis=1)
        cols = [out_var] + [col for col in X_train_scaled if col != out_var]
        X_train_scaled = X_train_scaled[cols]

        scaled_y_valid = log10_1p(y_valid.apply(pd.to_numeric, errors='coerce'))
        X_valid_real.loc[:, out_var] = log10_1p(X_valid_real[out_var])
        scaled_X_valid_real = ss_train.transform(X_valid_real)
        scaled_X_valid_real = pd.DataFrame(scaled_X_valid_real, index = X_valid_real.index, columns=X_valid_real.columns)
        X_valid_scaled = pd.concat([scaled_X_valid_real, X_valid_other], axis=1)
        cols = [out_var] + [col for col in X_valid_scaled if col != out_var]
        X_valid_scaled = X_valid_scaled[cols]

        scaled_y_test = log10_1p(y_test.apply(pd.to_numeric, errors='coerce'))
        X_test_real.loc[:, out_var] = log10_1p(X_test_real[out_var])
        scaled_X_test_real = ss_train.transform(X_test_real)
        scaled_X_test_real = pd.DataFrame(scaled_X_test_real, index = X_test_real.index, columns=X_test_real.columns)
        X_test_scaled = pd.concat([scaled_X_test_real, X_test_other], axis=1)
        cols = [out_var] + [col for col in X_test_scaled if col != out_var]
        X_test_scaled = X_test_scaled[cols]

        # Create sequences
        XX_train, yy_train, flags_train_seq = create_sequences(X_train_scaled, out_var, Input_Sequence, Output_Sequence, scaled_y_train, flags=flags_train)
        XX_valid, yy_valid, flags_valid_seq = create_sequences(X_valid_scaled, out_var, Input_Sequence, Output_Sequence, scaled_y_valid, flags=flags_valid)
        XX_test, yy_test, flags_test_seq = create_sequences(X_test_scaled, out_var, Input_Sequence, Output_Sequence, scaled_y_test, flags=flags_test)

        Sc_tr_x.append(XX_train)
        Sc_tr_y.append(yy_train)
        Sc_tr_flags.append(flags_train_seq)
        
        Sc_vl_x.append(XX_valid)
        Sc_vl_y.append(yy_valid)
        Sc_vl_flags.append(flags_valid_seq)

        Sc_te_x.append(XX_test)
        Sc_te_y.append(yy_test)
        Sc_te_flags.append(flags_test_seq)

        print(f"Train sequences shape for {location_name}:", XX_train.shape, yy_train.shape)
        print(f"Validation sequences shape for {location_name}:", XX_valid.shape, yy_valid.shape)
        print(f"Test sequences shape for {location_name}:", XX_test.shape, yy_test.shape)

# numpy array to tensor
Sc_tr_x = np.concatenate(Sc_tr_x, axis=0)
Sc_tr_y = np.concatenate(Sc_tr_y, axis=0)
Sc_tr_flags = np.concatenate(Sc_tr_flags, axis=0)

Sc_vl_x = np.concatenate(Sc_vl_x, axis=0)
Sc_vl_y = np.concatenate(Sc_vl_y, axis=0)
Sc_vl_flags = np.concatenate(Sc_vl_flags, axis=0)

Sc_te_x = np.concatenate(Sc_te_x, axis=0)
Sc_te_y = np.concatenate(Sc_te_y, axis=0)
Sc_te_flags = np.concatenate(Sc_te_flags, axis=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create datasets and dataloaders
train_dataset = TimeSeriesDataset(Sc_tr_x, Sc_tr_y, Sc_tr_flags, device)
valid_dataset = TimeSeriesDataset(Sc_vl_x, Sc_vl_y, Sc_vl_flags, device)
test_dataset = TimeSeriesDataset(Sc_te_x, Sc_te_y, Sc_te_flags, device)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Print shapes to verify
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(valid_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")   

# Select hyperparameters from the JSON config.
hyperparameters = training_config["hyperparameters"]
Dropout_list = hyperparameters["dropout"]
lr_list = hyperparameters["learning_rate"]
Num_LSTM_Layers_list = hyperparameters["num_lstm_layers"]
Hidden_Layer_Size_list = hyperparameters["hidden_layer_size"]
Embedding_Dim_list = hyperparameters["embedding_dim"]
Num_Attention_list = hyperparameters["num_attention_heads"]
EPOCHS = training_config["pretrain_epochs"]
pickle_dir = config["paths"]["pretrained_models"]
os.makedirs(pickle_dir, exist_ok=True)

for Dropout in Dropout_list:
    for learning_rate in lr_list:
        for Num_LSTM_Layers in Num_LSTM_Layers_list:
            for Hidden_Layer_Size in Hidden_Layer_Size_list:
                for Embedding_Dim in Embedding_Dim_list:
                    for Num_Attention_Heads in Num_Attention_list:
                        print("Dropout: ", Dropout)
                        print("learning_rate: ", learning_rate)
                        print("Num_LSTM_Layers: ", Num_LSTM_Layers)
                        print("Hidden_Layer_Size: ", Hidden_Layer_Size)
                        print("Embedding_Dim: ", Embedding_Dim)
                        print("Num_Attention_Heads: ", Num_Attention_Heads)
                     
                        # Model
                        model = TFT(
                            quantiles=quantiles, 
                            dropout=Dropout, 
                            device=device, 
                            hidden_layer_size=Hidden_Layer_Size, 
                            num_lstm_layers=Num_LSTM_Layers, 
                            embedding_dim=Embedding_Dim, 
                            encoder_steps=Input_Sequence, 
                            num_attention_heads=Num_Attention_Heads, 
                            col_to_idx=col_to_idx, 
                            static_covariates=static_covariates, 
                            time_dependent_categorical=time_dependent_categorical, 
                            time_dependent_continuous=time_dependent_continuous, 
                            category_counts=category_counts, 
                            known_time_dependent=known_time_dependent, 
                            observed_time_dependent=observed_time_dependent, 
                        )
                        model = model.to(device)
                        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                        criterion = QuantileLoss(quantiles=quantiles)
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15)

                        model = train_model_with_flag_masking(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, EPOCHS)
                        final_lr = optimizer.param_groups[0]['lr']
                        model.eval()
                        model.to("cpu")

                        model_save_path = os.path.join(
                            pickle_dir, 
                            f"dr{Dropout}_lr{learning_rate}_nl{Num_LSTM_Layers}_hid{Hidden_Layer_Size}_emb{Embedding_Dim}_atten{Num_Attention_Heads}_final_lr{final_lr:.6f}.pt"
                        )
                        # Save model
                        torch.save(model, model_save_path)

                        # Count number of pickle files saved
                        saved_pickles = [f for f in os.listdir(pickle_dir) if f.endswith('.pt')]
                        print(f"Total number of pickle files saved in {pickle_dir}: {len(saved_pickles)}")
