# Chapter 5 – Daily Marine Heatwave Archive (Zarr) for Web Visualisation

This repository contains the workflow used in Chapter 5 to generate a daily Marine Heatwave (MHW) archive from gridded Sea Surface Temperature (SST) data, saved as **Zarr** for fast time slicing in a Flask/web dashboard.

## What this produces

Given an SST dataset (`sst(time, lat, lon)`), the script computes daily fields:

- **`mhw_flag`**: binary mask (1 = day belongs to a detected MHW event, else 0)
- **`anomaly`**: SST anomaly relative to the seasonal climatology  
  \> `anomaly = sst - seas`
- **`intensity`**: exceedance above the MHW threshold, truncated at 0  
  \> `intensity = max(sst - thresh, 0)`

Outputs are written to a **Zarr store** (directory), enabling quick queries like:
- “Give me maps for 2024-08-15”
- “Subset around Ireland, load only one day”
- “Plot anomaly/intensity without reading the full 42-year cube”

## Detection method

MHW events are detected per grid cell using `detect()` in `alternew_hobday1.py`, an adaptation of the Hobday marine heatwave approach with configurable:
- climatology baseline period (`climatologyPeriod`)
- percentile threshold (`pctile`, 99th percentile)
- minimum duration (default in detector)
- optional merging of events separated by small gaps (`joinAcrossGaps`, `maxGap`)

See `alternew_hobday1.py` for the detailed implementation and assumptions.

# 2024 Test – CNN2D-LSTM Forecast + MHW Products Pipeline

This folder contains the reproducible workflow used in Chapter 5 to:
1) train a 1-step CNN2D-LSTM model (trained on 1982–2023),
2) replay/pseudo-realtime forecasts for 2024 (iterative rollout, lead 0..6),
3) evaluate forecasts against observations (metrics by lead and by horizon),
4) compute daily MHW products (anomaly, intensity, MHW flag) from forecast SST.

The CNN2D-LSTM architecture is described in Chapter 3.  
The MHW definition and climatology/threshold framework are described in Chapter 2.

---

## Scripts

### 1) Training (1982–2023)
**`chap5_2024train1.py`**
Trains a 1-step CNN2D-LSTM with sequence length 15 (iterative forecasting setup).

**Outputs**
- `CNN2DLSTM_TRAIN_SEQ15_1STEP/model_save/model_cnn2dlstm_seq15_1step_best.pth`
- `CNN2DLSTM_TRAIN_SEQ15_1STEP/model_save/ckpt_cnn2dlstm_seq15_1step.pt`  
  (includes scaler parameters + valid pixel mask + config)

---

### 2) Precompute pseudo-realtime predictions for 2024 (lead 0..6)
**`chap5_2024precompute2.py`**  
Loads the checkpoint and generates predictions for each init_date in 2024 by:
- using the previous 15 observed days as input,
- producing lead0 for init_date,
- rolling forward iteratively to lead6.

**Output**
- `CNN2DLSTM_PRECOMP_2024_SEQ15/cnn2dlstm_pred_2024_lead0to6.zarr`  
  Variable: `sst_pred(init_time, lead, lat, lon)`

---

### 3) Evaluate 2024 forecasts (metrics by lead and horizon)
**`chap5_2024evaluate3.py`** 
Compares `sst_pred(init_time,lead)` against observed SST at `target_time = init_time + lead`.

Computes:
- RMSE, MAE, Bias (obs − pred), R²
- By lead (0..6)
- By horizon (H = 3,5,7), aggregated over leads [0..H-1]

**Outputs**
- `CNN2DLSTM_EVAL_2024_SEQ15/metrics_by_lead.csv`
- `CNN2DLSTM_EVAL_2024_SEQ15/metrics_by_horizon.csv`

---

### 4) MHW products computed on forecast SST
**`chap5_aftermodel_nhw2024.py`**  
Computes daily:
- `anomaly(init_time,lead,lat,lon)`
- `intensity(init_time,lead,lat,lon)` = max(sst_pred − threshold, 0)
- `mhw_flag(init_time,lead,lat,lon)` (event-day flag)

Key design choice:
- Detection uses the full 7 forecast days (lead 0..6) plus an observed-history buffer
  before init_time (default 30 days) to support minimum duration and gap-merge logic.

**Output**
- `CNN2DLSTM_MHW_2024_SEQ15/cnn2dlstm_mhw_2024_lead0to6.zarr`

---

# 2025 Demo – CNN2D-LSTM Forecast + MHW Products

This folder reproduces the Chapter 5 “demo/live-style” pipeline for 2025:
1) train a 1-step CNN2D-LSTM model (trained through 2024),
2) precompute pseudo-realtime forecasts for 2025 (iterative rollout; lead 0..6),
3) evaluate forecasts vs observed SST (metrics by lead and horizon),
4) compute forecast-based MHW fields (anomaly, intensity, MHW flag) for web display.

**Model architecture**: see Chapter 3.  
**MHW framework**: see Chapter 2.

---

## Scripts

### 1) Train model (1982–2024)
**`chap5_train2025.py`**
Trains a 1-step CNN2D-LSTM with `seq_len=15` for iterative rollout.
- Adam lr=1e-4, batch=32, dropout=0.1
- early stopping patience=12 (max 500 epochs)
- masked loss over valid ocean pixels
- saves scaler + valid mask + config into checkpoint

**Outputs**
- `CNN2DLSTM_TRAIN_SEQ15_1STEP_1982_2024/model_save/model_cnn2dlstm_seq15_1step_best.pth`
- `CNN2DLSTM_TRAIN_SEQ15_1STEP_1982_2024/model_save/ckpt_cnn2dlstm_seq15_1step.pt`

---

### 2) Precompute 2025 forecasts (lead 0..6)
**`chap5_precompute2025.py`**  
Loads the checkpoint and generates pseudo-realtime forecasts for 2025 using iterative rollout.
Key setting:
- `INIT_START` (default `2025-03-15`) — start date for initialisations
End date is automatically chosen so target dates exist for lead 0..6.

**Output**
- `CNN2DLSTM_PRECOMP_2025_SEQ15/cnn2dlstm_pred_2025_lead0to6.zarr`
  - `sst_pred(init_time, lead, lat, lon)`

---

### 3) Evaluate 2025 forecasts
**`chap5_evaluate2025.py`**  
Evaluates predictions against observed SST at `target_time = init_time + lead`.
Metrics:
- RMSE, MAE, Bias (obs − pred), R²
Reports:
- by lead (0..6)
- by horizon (H = 3,5,7) pooling leads 0..H-1

**Outputs**
- `CNN2DLSTM_EVAL_2025_SEQ15/metrics_by_lead.csv`
- `CNN2DLSTM_EVAL_2025_SEQ15/metrics_by_horizon.csv`

---

### 4) MHW products from 2025 forecasts
**`chap5_mhw2025.py`**   
Computes daily:
- `anomaly(init_time,lead,lat,lon)` = sst_pred − seasonal climatology
- `intensity(init_time,lead,lat,lon)` = max(sst_pred − threshold, 0)
- `mhw_flag(init_time,lead,lat,lon)` = event-day flag

Important implementation details:
- climatology baseline: 1991–2024
- percentile threshold: 99th
- DOY mapping uses a leap-year reference; Feb 29 is interpolated
- event rules: minDuration=5, optional gap-join with maxGap=4
- observed buffer before init_time (default 30 days) is used for robust event detection
  near the forecast start; only forecast flags are retained

**Output**
- `CNN2DLSTM_MHW_2025_SEQ15/cnn2dlstm_mhw_2025_lead0to6.zarr`

---

## Requirements

Python 3.9+ recommended.

Packages:
- numpy, pandas, xarray, zarr, netCDF4 (or h5netcdf)
- torch
- scikit-learn
- tqdm (for MHW post-processing progress)

Example install (conda):
```bash
conda create -n chap5 python=3.10 -y
conda activate chap5
conda install -c conda-forge numpy pandas xarray zarr netcdf4 scikit-learn tqdm -y
pip install torch


