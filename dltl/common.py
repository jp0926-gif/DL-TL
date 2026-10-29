"""Shared data preparation and training utilities for TFT experiments."""

import numpy as np
import pandas as pd
import torch
from pykalman import KalmanFilter
from torch.utils.data import Dataset


def round_columns(df, keyword, decimals):
    """Round columns whose prefix matches ``keyword``."""
    for column in df.columns:
        if keyword in column.split("_")[0]:
            df[column] = np.round(df[column].astype(float), decimals)
    return df


class TimeSeriesDataset(Dataset):
    """Move time-series samples, targets and interpolation flags to a device."""

    def __init__(self, x, y, flags, device):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)
        self.flags = torch.tensor(flags, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.flags[index]


def create_sequences(data, target, seq_len, out_seq, target_values, flags=None):
    """Create TFT input windows, targets and optional interpolation flags."""
    period = list(data.index)
    start = period.index(data.loc[~data[target].isna(), :].index[0])
    end = period.index(data.loc[~data[target].isna(), :].index[-1])

    values = np.array(data)[start : end + 1, :]
    targets = np.array(target_values)[start : end + 1]
    flag_values = np.array(flags)[start : end + 1, :] if flags is not None else None

    sequences, sequence_targets, sequence_flags = [], [], []
    for index in range(len(values) - seq_len - out_seq + 1):
        sequences.append(values[index : index + seq_len + out_seq, :])
        sequence_targets.append(targets[index + seq_len + out_seq - 1])
        if flags is not None:
            target_index = flags.columns.get_loc(target)
            sequence_flags.append(flag_values[index + seq_len + out_seq - 1, target_index])

    x_array = np.array(sequences)
    y_array = np.array(sequence_targets).reshape(-1, 1)
    flags_array = np.array(sequence_flags).reshape(-1, 1) if flags is not None else None
    return x_array, y_array, flags_array


def create_sequences_with_flags(data, flags, seq_len, forecast_horizon, target_idx):
    """Create baseline-model windows with target interpolation flags."""
    sequences, targets, target_flags = [], [], []
    for index in range(len(data) - seq_len - forecast_horizon + 1):
        sequences.append(data[index : index + seq_len, :])
        target_position = index + seq_len + forecast_horizon - 1
        targets.append(data[target_position, target_idx])
        target_flags.append(flags[target_position, target_idx])
    return (
        np.array(sequences),
        np.array(targets).reshape(-1, 1),
        np.array(target_flags).reshape(-1, 1),
    )


def train_model_with_flag_masking(
    model,
    train_dataloader,
    valid_dataloader,
    criterion,
    optimizer,
    scheduler,
    EPOCHS,
    random_seed=42,
    patience=50,
):
    """Train a TFT model while excluding interpolated target observations."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    best_validation_loss = float("inf")
    no_improve = 0
    last_lr = optimizer.param_groups[0]["lr"]

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0
        count_actual_values_train = 0
        for train_inputs, train_targets, train_flags in train_dataloader:
            optimizer.zero_grad()
            train_outputs = model(train_inputs)
            mask = train_flags == 0
            mask_expanded = mask.unsqueeze(-1).expand_as(train_outputs)
            train_outputs_masked = train_outputs * mask_expanded.float()
            train_targets_masked = train_targets * mask.float()

            if mask_expanded.sum() > 0:
                train_loss = criterion(train_outputs_masked, train_targets_masked)
                train_loss.backward()
                optimizer.step()
                total_train_loss += train_loss.item()
                count_actual_values_train += 1

        avg_train_loss = (
            total_train_loss / count_actual_values_train
            if count_actual_values_train > 0
            else float("inf")
        )

        model.eval()
        validation_loss = 0
        count_actual_values = 0
        with torch.no_grad():
            for valid_inputs, valid_targets, valid_flags in valid_dataloader:
                valid_outputs = model(valid_inputs)
                mask = valid_flags == 0
                mask_expanded = mask.unsqueeze(-1).expand_as(valid_outputs)
                valid_outputs_masked = valid_outputs * mask_expanded.float()
                valid_targets_masked = valid_targets * mask.float()

                if mask_expanded.sum() > 0:
                    valid_loss = criterion(valid_outputs_masked, valid_targets_masked)
                    validation_loss += valid_loss.item()
                    count_actual_values += 1

        if count_actual_values > 0:
            validation_loss /= count_actual_values

        current_lr = optimizer.param_groups[0]["lr"]
        if epoch % 5 == 0 or current_lr != last_lr:
            print(
                f"Epoch {epoch}/{EPOCHS}, Training Loss: {avg_train_loss:.4f}, "
                f"Validation Loss: {validation_loss:.4f}, LR: {current_lr:.6f}"
            )

        if current_lr != last_lr:
            print(f"Learning rate changed to {current_lr:.6f} at epoch {epoch}")
            last_lr = current_lr

        scheduler.step(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(
                f"Early stopping at epoch {epoch}, "
                f"best validation loss: {best_validation_loss:.4f}"
            )
            return model

    return model


def log10_1p(values):
    """Apply the log10(1 + x) transform used by the experiments."""
    return np.log10(1 + values)


def filter_columns(df, variables):
    """Select columns containing at least one configured variable name."""
    return df[
        [column for column in df.columns if any(variable in column for variable in variables)]
    ]


def classify_stage(value):
    """Map a numeric value to the four stages used by baseline experiments."""
    if value < 4:
        return 1
    if value < 5:
        return 2
    if value < 6:
        return 3
    return 4


def kalman_filter_interpolation(df):
    """Fill missing numeric values and return flags marking filled cells."""
    filled = df.copy()
    interpolation_flags = pd.DataFrame(0, index=filled.index, columns=filled.columns)

    for column in filled.columns:
        values = filled[column].values
        if np.isnan(values).any():
            masked_values = np.ma.masked_invalid(values)
            kalman_filter = KalmanFilter(
                initial_state_mean=np.nanmean(values),
                n_dim_obs=1,
                transition_matrices=[1],
                observation_matrices=[1],
                transition_covariance=np.eye(1),
                observation_covariance=np.eye(1),
            )
            kalman_filter = kalman_filter.em(masked_values, n_iter=10)
            filled_values, _ = kalman_filter.smooth(masked_values)
            filled[column] = np.where(np.isnan(values), filled_values.flatten(), values)
            interpolation_flags[column] = np.where(np.isnan(values), 1, 0)

    return filled, interpolation_flags
