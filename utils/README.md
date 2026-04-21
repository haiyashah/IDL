# RUL Prediction: LSTM vs. Temporal Fusion Transformer
24-788 Introduction to Deep Learning, Spring 2026 Mini-Project

## Overview

This project compares two models for Remaining Useful Life (RUL) prediction on the
NASA C-MAPSS turbofan engine dataset. The baseline is a stacked LSTM and the variant
is a Temporal Fusion Transformer (TFT), which adds variable selection and self-attention
on top of an LSTM encoder. I run both models on FD001 and FD003 to see how each handles
single vs. two-fault-mode degradation.

Contribution 1: LSTM vs. TFT on FD001
Contribution 2: Cross-subset comparison on FD001 vs. FD003

---

## Repository Layout

```
cmapss_project/
├── data/                        # put CMAPSSData files here
├── models/
│   ├── lstm_baseline.py         # stacked LSTM
│   └── tft.py                   # Temporal Fusion Transformer
├── utils/
│   ├── data.py                  # loading, preprocessing, sliding window
│   └── metrics.py               # RMSE and asymmetric PHM score
├── train.py                     # trains either model on any subset
├── run_all_experiments.py       # runs all 4 combinations in sequence
├── reproduce_results.ipynb      # loads checkpoints and generates figures
├── checkpoints/                 # saved model weights
├── results/                     # JSON files with metrics and predictions
└── figures/                     # output PDF figures
```

---

## Setup

```bash
pip install torch numpy pandas scikit-learn matplotlib wandb
```

---

## Data Download

Download from the NASA Prognostics Data Repository and unzip:

```
https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
```

The inner CMAPSSData.zip needs a second unzip. The flat directory containing
train_FD001.txt, test_FD001.txt, RUL_FD001.txt etc. is what you pass as --data_dir.

---

## Training

Single model:
```bash
python train.py --model lstm --subset FD001 --epochs 50 --data_dir /path/to/data
python train.py --model tft  --subset FD001 --epochs 50 --data_dir /path/to/data
```

All four combinations:
```bash
python run_all_experiments.py --data_dir /path/to/data --epochs 50 --no_wandb
```

---

## Reproducing Results

Open reproduce_results.ipynb and run all cells. It loads the saved checkpoints
and result JSONs and regenerates all figures without retraining.

Update DATA_DIR at the top of the notebook to match your local data path.

---

## Hyperparameters

| Setting | LSTM | TFT |
|---|---|---|
| Hidden size | 128 | d_model=64 |
| Layers | 2 | 2 LSTM + 4-head attention |
| Dropout | 0.2 | 0.1 |
| Learning rate | 1e-3 | 5e-4 |
| Batch size | 64 | 64 |
| Window length | 30 | 30 |
| RUL cap | 125 | 125 |
| Optimizer | Adam + ReduceLROnPlateau | Adam + ReduceLROnPlateau |

---

## Metrics

RMSE (primary): sqrt(1/N * sum((y_pred - y_true)^2))

Asymmetric PHM score (secondary, lower is better):
  d = y_pred - y_true
  s = exp(-d/13) - 1  if d < 0  (early prediction)
  s = exp(d/10)  - 1  if d >= 0 (late prediction, penalized more heavily)
  Score = sum(s_i)

---

## References

- Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series
  Forecasting," NeurIPS 2020. https://arxiv.org/abs/1912.09363
- Hochreiter and Schmidhuber, "Long Short-Term Memory," Neural Computation, 1997.
- Saxena et al., "Damage Propagation Modeling for Aircraft Engine Run-to-Failure
  Simulation," PHMAP 2008.

---

## AI Tool Use

Claude (Anthropic) was used to help write the initial code structure. All model
architecture choices, hyperparameter decisions, experimental analysis, and the
written report are my own work.
