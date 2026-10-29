# DL-TL

Deep learning and transfer-learning experiments for forecasting cyanobacterial cell counts (`Cyanocell`) from water-quality and weather time-series data.

이 저장소는 수질·수문·기상 시계열을 이용해 남조류 세포 수를 예측하는 연구 코드를 정리한 것입니다. LSTM, GRU, BiLSTM, Transformer, CLA와 Temporal Fusion Transformer(TFT)를 비교하고, TFT 사전학습 모델을 여러 fine-tuning 방식으로 전이하는 실험을 포함합니다.

## Project overview

- **Target**: cyanobacterial cell count (`Cyanocell`)
- **Inputs**: water temperature, chlorophyll-a, total nitrogen/phosphorus, water level, discharge, precipitation, temperature forecast and related variables
- **Models**: LSTM, GRU, BiLSTM, Transformer, CLA, TFT
- **Preprocessing**: Kalman-filter interpolation, log transform and min-max scaling
- **Evaluation metrics**: RMSE, MAE and R²
- **Data split used by the TFT scripts**: training through 2021, validation in 2022 and testing in 2023

## Repository contents

| Path | Description |
| --- | --- |
| `LSTM.py` | LSTM baseline experiment |
| `GRU.py` | GRU baseline experiment |
| `BILSTM.py` | Bidirectional LSTM baseline experiment |
| `TRANSFORMER.py` | Transformer encoder baseline experiment |
| `CLA.py` | CLA architecture experiment |
| `BEFORE_TL.py` | TFT experiment before transfer learning |
| `PRETRAIN_TFT.py` | TFT pretraining workflow |
| `FINE_TUNE_S1TO4.py` | Fine-tuning schemes A–D |
| `dltl/common.py` | Shared preprocessing, sequence creation and TFT training utilities |
| `dltl/baseline.py` | Shared baseline-model training loop |
| `dltl/recurrent.py` | Shared LSTM, GRU and BiLSTM model definitions |
| `dltl/tft.py` | Shared TFT layers, loss and model definition |
| `dltl/config.py` | JSON configuration loader and path normalization |
| `config.example.json` | Example data, hyperparameter and output-path configuration |
| `DATA/` | Time-series CSV files used by the baseline experiments |
| `TFT_DATA/` | Time-series CSV files prepared for TFT experiments |

## Environment setup

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/jp0926-gif/DL-TL.git
cd DL-TL
python -m venv .venv
```

Activate the virtual environment and install the dependencies:

```bash
pip install -r requirements.txt
```

PyTorch automatically uses `cuda:0` when CUDA is available and otherwise falls back to the CPU.

## Running an experiment

The TFT files use a JSON configuration so paths and experiment parameters do not need to be edited inside the Python source.

1. Copy `config.example.json` to `config.json`.
2. Set the data locations and output directories in `config.json`.
3. Fill the hyperparameter arrays under `training.hyperparameters`.
4. Confirm that the expected columns exist in the selected CSV file.

For example, after configuring the paths and hyperparameters:

```bash
python PRETRAIN_TFT.py --config config.json
python FINE_TUNE_S1TO4.py --config config.json
```

Relative paths in the config file are resolved from the config file's directory. The included CSV files use a `Date` index and may require CP949 decoding; both the index column and encoding are configurable.

## Reproducibility notes

- Random seeds are set in the experiment scripts.
- Model checkpoints and generated artifacts (`*.pt`, `*.pth`, `*.pkl`) are intentionally excluded from Git.
- The original environment versions were not recorded, so `requirements.txt` lists compatible minimum versions rather than an exact lock file.
- Hyperparameter arrays are empty in `config.example.json`. Add the values used for a specific experiment before running it.

## Data and license

Before reusing or redistributing the included data, verify its original source, license and any attribution requirements. A software license has not yet been selected for this repository; reuse of the code requires permission from the repository owner until a license is added.
