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
from dltl.config import load_experiment_config_from_cli
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

# Loop over each river and its locations
for river_name, river_info in river_data_info.items():

    for location_name, filepath in river_info['locations'].items():
        print(f"Starting fine-tuning for River: {river_name}, Location: {location_name}")
        # Load and preprocess the data
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
        
        # 카테고리컬 컬럼 변환 TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
        categorical_scalers = {}
        categorical_columns = ['season']
        category_counts = {'season':2}
        
        for col in categorical_columns:
            srs = transform_df[col].apply(str) 
            categorical_scalers[col] = LabelEncoder().fit(srs.values)
            transform_df[col] = categorical_scalers[col].transform(transform_df[col].apply(str))

        # 추가 변수 생성
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

        real_columns = [col for col in X.columns if any(var in col for var in Input_Variables)]
        other_columns = [col for col in X.columns if col not in real_columns]
        
        X_train_real = X_train[real_columns]
        X_train_other = X_train[other_columns]
        X_valid_real = X_valid[real_columns]
        X_valid_other = X_valid[other_columns]
        
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

        # Create sequences
        XX_train, yy_train, flags_train_seq = create_sequences(X_train_scaled, out_var, Input_Sequence, Output_Sequence, scaled_y_train, flags=flags_train)
        XX_valid, yy_valid, flags_valid_seq = create_sequences(X_valid_scaled, out_var, Input_Sequence, Output_Sequence, scaled_y_valid, flags=flags_valid)

        print(f"Train sequences shape for {location_name}:", XX_train.shape, yy_train.shape)
        print(f"Validation sequences shape for {location_name}:", XX_valid.shape, yy_valid.shape)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        train_dataset = TimeSeriesDataset(XX_train,yy_train, flags_train_seq, device)
        valid_dataset = TimeSeriesDataset(XX_valid,yy_valid, flags_valid_seq, device)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        hyperparameters = training_config["hyperparameters"]
        Dropout_list = hyperparameters["dropout"]
        lr_list = hyperparameters["learning_rate"]
        Num_LSTM_Layers_list = hyperparameters["num_lstm_layers"]
        Hidden_Layer_Size_list = hyperparameters["hidden_layer_size"]
        Embedding_Dim_list = hyperparameters["embedding_dim"]
        Num_Attention_list = hyperparameters["num_attention_heads"]
        EPOCHS = training_config["fine_tune_epochs"]
        model_dir = config["paths"]["pretrained_models"]
        # Iterate through all hyperparameter combinations
        for Dropout in Dropout_list:
            for learning_rate in lr_list:
                for Num_LSTM_Layers in Num_LSTM_Layers_list:
                    for Hidden_Layer_Size in Hidden_Layer_Size_list:
                        for Embedding_Dim in Embedding_Dim_list:
                            for Num_Attention_Heads in Num_Attention_list:
                                
                                # Define the pre-trained model path for the specific river and location
                                best_model_path = os.path.join(
                                    model_dir,
                                    f'dr{Dropout}_lr{learning_rate}_nl{Num_LSTM_Layers}_hid{Hidden_Layer_Size}_emb{Embedding_Dim}_atten{Num_Attention_Heads}_final_lr*.pt',
                                )
                                                         
                                model_paths = config["paths"]["fine_tune_models"]
                                scheme_a_save_dir = os.path.join(model_paths["scheme_a"], river_name, location_name)
                                scheme_b_save_dir = os.path.join(model_paths["scheme_b"], river_name, location_name)
                                scheme_c_save_dir = os.path.join(model_paths["scheme_c"], river_name, location_name)
                                scheme_d_save_dir = os.path.join(model_paths["scheme_d"], river_name, location_name)
                                os.makedirs(scheme_a_save_dir, exist_ok=True)  # Create directory if it doesn't exist                               
                                os.makedirs(scheme_b_save_dir, exist_ok=True)                                
                                os.makedirs(scheme_c_save_dir, exist_ok=True)
                                os.makedirs(scheme_d_save_dir, exist_ok=True)                                

                                weight_paths = config["paths"]["fine_tune_weights"]
                                scheme_a_save_dir1 = os.path.join(weight_paths["scheme_a"], river_name, location_name)
                                scheme_b_save_dir1 = os.path.join(weight_paths["scheme_b"], river_name, location_name)
                                scheme_c_save_dir1 = os.path.join(weight_paths["scheme_c"], river_name, location_name)
                                scheme_d_save_dir1 = os.path.join(weight_paths["scheme_d"], river_name, location_name)
                                os.makedirs(scheme_a_save_dir1, exist_ok=True)  # Create directory if it doesn't exist                               
                                os.makedirs(scheme_b_save_dir1, exist_ok=True)                                
                                os.makedirs(scheme_c_save_dir1, exist_ok=True)
                                os.makedirs(scheme_d_save_dir1, exist_ok=True)      
                                
                                scheme_a_save_model_path = f'{scheme_a_save_dir}/dr{Dropout}_lr{learning_rate}_nl{Num_LSTM_Layers}_hid{Hidden_Layer_Size}_emb{Embedding_Dim}_atten{Num_Attention_Heads}_final_lr*.pt'
                                scheme_b_save_model_path = f'{scheme_b_save_dir}/dr{Dropout}_lr{learning_rate}_nl{Num_LSTM_Layers}_hid{Hidden_Layer_Size}_emb{Embedding_Dim}_atten{Num_Attention_Heads}_final_lr*.pt'
                                scheme_c_save_model_path = f'{scheme_c_save_dir}/dr{Dropout}_lr{learning_rate}_nl{Num_LSTM_Layers}_hid{Hidden_Layer_Size}_emb{Embedding_Dim}_atten{Num_Attention_Heads}_final_lr*.pt' 
                                scheme_d_save_model_path = f'{scheme_d_save_dir}/dr{Dropout}_lr{learning_rate}_nl{Num_LSTM_Layers}_hid{Hidden_Layer_Size}_emb{Embedding_Dim}_atten{Num_Attention_Heads}_final_lr*.pt'
                                                                
                                # Check if files for Scheme A and Scheme B already exist
                                scheme_a_existing = glob.glob(scheme_a_save_model_path)
                                scheme_b_existing = glob.glob(scheme_b_save_model_path)
                                scheme_c_existing = glob.glob(scheme_c_save_model_path)
                                scheme_d_existing = glob.glob(scheme_d_save_model_path)

                                if scheme_a_existing and scheme_b_existing and scheme_c_existing and scheme_d_existing:
                                    print(f"Skipping Dropout: {Dropout}, LR: {learning_rate}, LSTM Layers: {Num_LSTM_Layers}, "
                                        f"Hidden: {Hidden_Layer_Size}, Embedding: {Embedding_Dim}, Attention: {Num_Attention_Heads} - Models already exist.")
                                    continue

                                # If pre-trained model does not exist, skip training
                                pre_trained_model = glob.glob(best_model_path)
                                if not pre_trained_model:
                                    print(f"No pre-trained model found for Dropout: {Dropout}, LR: {learning_rate}, "
                                            f"LSTM Layers: {Num_LSTM_Layers}, Hidden: {Hidden_Layer_Size}, "
                                            f"Embedding: {Embedding_Dim}, Attention: {Num_Attention_Heads}. Skipping...")
                                    continue
                                
                                # Load the pre-trained model
                                best_model_path = pre_trained_model[0]
                                print(f"Found pre-trained model: {best_model_path}")

                                # If not, perform training
                                print(f"Training Dropout: {Dropout}, LR: {learning_rate}, LSTM Layers: {Num_LSTM_Layers}, "
                                    f"Hidden: {Hidden_Layer_Size}, Embedding: {Embedding_Dim}, Attention: {Num_Attention_Heads}")                                                                
                      
                                # Scheme A: Full fine-tuning TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
                                # Save to directory SCHA
                                model = torch.load(best_model_path)
                                model = model.to(device)

                                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                                criterion = QuantileLoss(quantiles=quantiles)
                                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15)

                                # Train the model
                                model = train_model_with_flag_masking(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, EPOCHS)
                                final_lr = optimizer.param_groups[0]['lr']

                                # Save Scheme A model
                                torch.save(model, f'{scheme_a_save_dir}/dr{Dropout}_lr{learning_rate}_nl{Num_LSTM_Layers}_hid{Hidden_Layer_Size}_emb{Embedding_Dim}_atten{Num_Attention_Heads}_final_lr{final_lr:.6f}.pt')
                                print(f"Saved Scheme A model to {scheme_a_save_dir}")
                                
                                final_model_filename = f'dr{Dropout}_lr{learning_rate}_nl{Num_LSTM_Layers}_hid{Hidden_Layer_Size}_emb{Embedding_Dim}_atten{Num_Attention_Heads}_final_lr{final_lr:.6f}.pt'
                                final_model_path1 = os.path.join(scheme_a_save_dir1, final_model_filename)
                                torch.save({
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'scheduler_state_dict': scheduler.state_dict(),
                                    'final_lr': final_lr,
                                }, final_model_path1)
                                print(f"Model weights saved at {final_model_path1}")  

                                # Variables to save
                                preprocessing_inf = {
                                    'col_to_idx': col_to_idx,
                                    'time_dependent_continuous': time_dependent_continuous,
                                    'time_dependent_categorical': time_dependent_categorical,
                                    'static_covariates': static_covariates,
                                    'known_time_dependent': known_time_dependent,
                                    'observed_time_dependent': observed_time_dependent,
                                    'category_counts': category_counts,
                                    'categorical_scalers': categorical_scalers
                                }

                                # Save the dictionary as a pickle file
                                metadata_filename = f'preprocessing_inf_dr{Dropout}_lr{learning_rate}_nl{Num_LSTM_Layers}_hid{Hidden_Layer_Size}_emb{Embedding_Dim}_atten{Num_Attention_Heads}.pkl'
                                metadata_save_path = os.path.join(scheme_a_save_dir1, metadata_filename)
                                with open(metadata_save_path, 'wb') as f:
                                    pickle.dump(preprocessing_inf, f)
                                print(f"Preprocessing metadata saved to {metadata_save_path}")
                                                                 
                                del model  # Free memory
                                
                                # After saving, count how many pickles are in the directory
                                pickle_files = [f for f in os.listdir(scheme_a_save_dir) if f.endswith('.pt')]
                                print(f"Number of pickles saved for {location_name}: {len(pickle_files)}")


                                # Scheme B: Freeze all except output layer TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
                                # Save to directory SCHB
                                model = torch.load(best_model_path)
                                model = model.to(device)

                                # Freeze all layers except for the output layer
                                for name, param in model.named_parameters():
                                    if 'output' not in name:
                                        param.requires_grad = False
                                    else:
                                        param.requires_grad = True

                                # Set optimizer (only output layer trainable)
                                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
                                criterion = QuantileLoss(quantiles=quantiles)
                                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15)

                                # Train the model
                                model = train_model_with_flag_masking(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, EPOCHS)
                                final_lr = optimizer.param_groups[0]['lr']

                                # Save Scheme B model
                                torch.save(model, f'{scheme_b_save_dir}/dr{Dropout}_lr{learning_rate}_nl{Num_LSTM_Layers}_hid{Hidden_Layer_Size}_emb{Embedding_Dim}_atten{Num_Attention_Heads}_final_lr{final_lr:.6f}.pt')
                                print(f"Saved Scheme B model to {scheme_b_save_dir}")
                                
                                final_model_filename = f'dr{Dropout}_lr{learning_rate}_nl{Num_LSTM_Layers}_hid{Hidden_Layer_Size}_emb{Embedding_Dim}_atten{Num_Attention_Heads}_final_lr{final_lr:.6f}.pt'
                                final_model_path1 = os.path.join(scheme_b_save_dir1, final_model_filename)
                                torch.save({
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'scheduler_state_dict': scheduler.state_dict(),
                                    'final_lr': final_lr,
                                }, final_model_path1)
                                print(f"Model weights saved at {final_model_path1}")  

                                # Variables to save
                                preprocessing_inf = {
                                    'col_to_idx': col_to_idx,
                                    'time_dependent_continuous': time_dependent_continuous,
                                    'time_dependent_categorical': time_dependent_categorical,
                                    'static_covariates': static_covariates,
                                    'known_time_dependent': known_time_dependent,
                                    'observed_time_dependent': observed_time_dependent,
                                    'category_counts': category_counts,
                                    'categorical_scalers': categorical_scalers
                                }

                                # Save the dictionary as a pickle file
                                metadata_filename = f'preprocessing_inf_dr{Dropout}_lr{learning_rate}_nl{Num_LSTM_Layers}_hid{Hidden_Layer_Size}_emb{Embedding_Dim}_atten{Num_Attention_Heads}.pkl'
                                metadata_save_path = os.path.join(scheme_b_save_dir1, metadata_filename)
                                with open(metadata_save_path, 'wb') as f:
                                    pickle.dump(preprocessing_inf, f)
                                print(f"Preprocessing metadata saved to {metadata_save_path}")
                                                                
                                del model  # Free memory
                                
                                # After saving, count how many pickles are in the directory
                                pickle_files = [f for f in os.listdir(scheme_b_save_dir) if f.endswith('.pt')]
                                print(f"Number of pickles saved for {location_name}: {len(pickle_files)}")


                                # Scheme C: Reinitialize the output layer without freezing TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
                                # Save to directory SCHC
                                model = torch.load(best_model_path)
                                model = model.to(device)
                                
                                # Reinitialize the output layer
                                if hasattr(model.output.module, 'weight'):
                                    nn.init.xavier_uniform_(model.output.module.weight)  # Xavier initialization
                                if hasattr(model.output.module, 'bias'):
                                    nn.init.zeros_(model.output.module.bias)  # Set biases to zero

                                # Set optimizer (all layers trainable)
                                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                                criterion = QuantileLoss(quantiles=quantiles)
                                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15)

                                # Train the model
                                model = train_model_with_flag_masking(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, EPOCHS)
                                final_lr = optimizer.param_groups[0]['lr']

                                # Save Scheme C model                            
                                torch.save(model, f'{scheme_c_save_dir}/dr{Dropout}_lr{learning_rate}_nl{Num_LSTM_Layers}_hid{Hidden_Layer_Size}_emb{Embedding_Dim}_atten{Num_Attention_Heads}_final_lr{final_lr:.6f}.pt')
                                print(f"Saved Scheme C model to {scheme_c_save_dir}")
                                
                                final_model_filename = f'dr{Dropout}_lr{learning_rate}_nl{Num_LSTM_Layers}_hid{Hidden_Layer_Size}_emb{Embedding_Dim}_atten{Num_Attention_Heads}_final_lr{final_lr:.6f}.pt'
                                final_model_path1 = os.path.join(scheme_c_save_dir1, final_model_filename)
                                torch.save({
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'scheduler_state_dict': scheduler.state_dict(),
                                    'final_lr': final_lr,
                                }, final_model_path1)
                                print(f"Model weights saved at {final_model_path1}")  

                                # Variables to save
                                preprocessing_inf = {
                                    'col_to_idx': col_to_idx,
                                    'time_dependent_continuous': time_dependent_continuous,
                                    'time_dependent_categorical': time_dependent_categorical,
                                    'static_covariates': static_covariates,
                                    'known_time_dependent': known_time_dependent,
                                    'observed_time_dependent': observed_time_dependent,
                                    'category_counts': category_counts,
                                    'categorical_scalers': categorical_scalers
                                }

                                # Save the dictionary as a pickle file
                                metadata_filename = f'preprocessing_inf_dr{Dropout}_lr{learning_rate}_nl{Num_LSTM_Layers}_hid{Hidden_Layer_Size}_emb{Embedding_Dim}_atten{Num_Attention_Heads}.pkl'
                                metadata_save_path = os.path.join(scheme_c_save_dir1, metadata_filename)
                                with open(metadata_save_path, 'wb') as f:
                                    pickle.dump(preprocessing_inf, f)
                                print(f"Preprocessing metadata saved to {metadata_save_path}")
                                                                
                                del model  # Free memory

                                # After saving, count how many pickles are in the directory
                                pickle_files = [f for f in os.listdir(scheme_c_save_dir) if f.endswith('.pt')]
                                print(f"Number of pickles saved for {location_name}: {len(pickle_files)}")

                                # Scheme D: Freeze all except output layer and reinitialize the output layer TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
                                # Save to directory SCHB
                                model = torch.load(best_model_path)
                                model = model.to(device)

                                # Freeze all layers except for the output layer
                                for name, param in model.named_parameters():
                                    if 'output' not in name:
                                        param.requires_grad = False
                                    else:
                                        param.requires_grad = True

                                # Reinitialize the output layer
                                if hasattr(model.output.module, 'weight'):
                                    nn.init.xavier_uniform_(model.output.module.weight)  # Xavier initialization
                                if hasattr(model.output.module, 'bias'):
                                    nn.init.zeros_(model.output.module.bias)  # Set biases to zero

                                # Set optimizer (only output layer trainable)
                                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
                                criterion = QuantileLoss(quantiles=quantiles)
                                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15)

                                # Train the model
                                model = train_model_with_flag_masking(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, EPOCHS)
                                final_lr = optimizer.param_groups[0]['lr']

                                # Save Scheme D model                           
                                torch.save(model, f'{scheme_d_save_dir}/dr{Dropout}_lr{learning_rate}_nl{Num_LSTM_Layers}_hid{Hidden_Layer_Size}_emb{Embedding_Dim}_atten{Num_Attention_Heads}_final_lr{final_lr:.6f}.pt')
                                print(f"Saved Scheme D model to {scheme_d_save_dir}")
                                
                                final_model_filename = f'dr{Dropout}_lr{learning_rate}_nl{Num_LSTM_Layers}_hid{Hidden_Layer_Size}_emb{Embedding_Dim}_atten{Num_Attention_Heads}_final_lr{final_lr:.6f}.pt'
                                final_model_path1 = os.path.join(scheme_d_save_dir1, final_model_filename)
                                torch.save({
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'scheduler_state_dict': scheduler.state_dict(),
                                    'final_lr': final_lr,
                                }, final_model_path1)
                                print(f"Model weights saved at {final_model_path1}")  

                                # Variables to save
                                preprocessing_inf = {
                                    'col_to_idx': col_to_idx,
                                    'time_dependent_continuous': time_dependent_continuous,
                                    'time_dependent_categorical': time_dependent_categorical,
                                    'static_covariates': static_covariates,
                                    'known_time_dependent': known_time_dependent,
                                    'observed_time_dependent': observed_time_dependent,
                                    'category_counts': category_counts,
                                    'categorical_scalers': categorical_scalers
                                }

                                # Save the dictionary as a pickle file
                                metadata_filename = f'preprocessing_inf_dr{Dropout}_lr{learning_rate}_nl{Num_LSTM_Layers}_hid{Hidden_Layer_Size}_emb{Embedding_Dim}_atten{Num_Attention_Heads}.pkl'
                                metadata_save_path = os.path.join(scheme_d_save_dir1, metadata_filename)
                                with open(metadata_save_path, 'wb') as f:
                                    pickle.dump(preprocessing_inf, f)
                                print(f"Preprocessing metadata saved to {metadata_save_path}") 
                                                                                                
                                del model  # Free memory
                                
                                # After saving, count how many pickles are in the directory
                                pickle_files = [f for f in os.listdir(scheme_d_save_dir) if f.endswith('.pt')]
                                print(f"Number of pickles saved for {location_name}: {len(pickle_files)}")

                            else:
                                print('Pickle file not found:', best_model_path)
