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

## Repository structure

```text
DL-TL/
├─ configs/
│  └─ config.example.json
├─ data/
│  └─ README.md
├─ dltl/
│  ├─ models/
│  │  ├─ cla.py
│  │  ├─ recurrent.py
│  │  ├─ tft.py
│  │  └─ transformer.py
│  ├─ baseline.py
│  ├─ common.py
│  └─ config.py
├─ scripts/
│  ├─ pretrain_tft.py
│  └─ fine_tune_tft.py
├─ README.md
└─ requirements.txt
```

- `dltl/models/` contains the model architectures.
- `dltl/common.py` contains shared preprocessing, sequence and TFT training utilities.
- `dltl/baseline.py` contains the baseline-model training loop.
- `scripts/` contains the runnable pretraining and fine-tuning workflows.
- `configs/` contains the example experiment configuration.
- `data/` documents the expected dataset without redistributing raw CSV files.

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

1. Copy `configs/config.example.json` to `configs/config.json`.
2. Place the CSV files locally and set their paths in `configs/config.json`.
3. Fill the hyperparameter arrays under `training.hyperparameters`.
4. Confirm that the expected columns exist in the selected CSV file.

For example, after configuring the paths and hyperparameters:

```bash
python -m scripts.pretrain_tft --config configs/config.json
python -m scripts.fine_tune_tft --config configs/config.json
```

Relative paths are resolved from the config file's directory. The scripts expect a `Date` index by default and support configurable CSV encoding and index columns.

## Reproducibility notes

- Random seeds are set in the experiment scripts.
- Model checkpoints and generated artifacts (`*.pt`, `*.pth`, `*.pkl`) are intentionally excluded from Git.
- The original environment versions were not recorded, so `requirements.txt` lists compatible minimum versions rather than an exact lock file.
- Hyperparameter arrays are empty in `configs/config.example.json`. Add the values used for a specific experiment before running it.

## Data and license

Raw CSV files are not distributed because their original redistribution license has not been verified. See `data/README.md` for the expected schema and document the source, download date, license and preprocessing steps for any dataset you use.

A software license has not yet been selected for this repository; reuse of the code requires permission from the repository owner until a license is added.
