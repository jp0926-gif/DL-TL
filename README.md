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

The files are research scripts rather than a packaged command-line application. Before running one of them:

1. Set the CSV path in `river_data_info`.
2. Fill the hyperparameter lists such as `Dropout_list`, `lr_list` and `Hidden_Layer_Size_list`.
3. Replace the `/PATH/TO/YOUR/...` output directories with writable local paths.
4. Confirm that the expected columns exist in the selected CSV file.

For example, after configuring the paths and hyperparameters:

```bash
python PRETRAIN_TFT.py
python FINE_TUNE_S1TO4.py
```

The included CSV files use a `Date` index and may require CP949 decoding, as shown in the scripts.

## Reproducibility notes

- Random seeds are set in the experiment scripts.
- Model checkpoints and generated artifacts (`*.pt`, `*.pth`, `*.pkl`) are intentionally excluded from Git.
- The original environment versions were not recorded, so `requirements.txt` lists compatible minimum versions rather than an exact lock file.
- Hyperparameter arrays are empty in the public snapshot. Add the values used for a specific experiment before running it.

## Data and license

Before reusing or redistributing the included data, verify its original source, license and any attribution requirements. A software license has not yet been selected for this repository; reuse of the code requires permission from the repository owner until a license is added.

## Future cleanup

- Move repeated preprocessing and training utilities into reusable modules.
- Replace hard-coded paths with a YAML/TOML configuration or command-line arguments.
- Record exact experiment parameters and results.
- Add automated tests for sequence creation, interpolation and model tensor shapes.
