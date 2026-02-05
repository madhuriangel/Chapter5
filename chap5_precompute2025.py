"""
chap5_precompute2025.py

Precompute pseudo-realtime 2025 predictions using trained 1-step CNN2D-LSTM (iterative rollout).
Outputs Zarr: sst_pred(init_time, lead, lat, lon) with lead=0..6
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# =========================
# ONLY PLACE TO EDIT
# =========================
DATA_PATH = "noaa_icesmi_combinefile_FINAL_1res1982_2025.nc"
CKPT_PATH = "CNN2DLSTM_TRAIN_SEQ15_1STEP_1982_2024/model_save/ckpt_cnn2dlstm_seq15_1step.pt"

OUT_DIR   = "CNN2DLSTM_PRECOMP_2025_SEQ15"
OUT_ZARR  = os.path.join(OUT_DIR, "cnn2dlstm_pred_2025_lead0to6.zarr")

SEQ_LEN     = 15
MAX_HORIZON = 7   # store 0..6

INIT_START = "2025-03-15"   # mid-March (edit if needed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================

class CNN2D_LSTM_Model(nn.Module):
    def __init__(self, H, W, hidden_units=(256, 128, 64), dropout_rate=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((H // 4, W // 4))
        )
        enc_h, enc_w = H // 4, W // 4
        feat = 32 * enc_h * enc_w
        self.lstm1 = nn.LSTM(feat, hidden_units[1], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_units[1], hidden_units[2], batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_units[2], H * W)
        self.H, self.W = H, W

    def forward(self, x):
        B, T, H, W = x.shape
        x = x.unsqueeze(2).reshape(B*T, 1, H, W)
        z = self.encoder(x).view(B, T, -1)
        z, _ = self.lstm1(z)
        z, _ = self.lstm2(z)
        z = self.dropout(z[:, -1, :])
        out = self.fc(z)
        return out.view(B, self.H, self.W)

def rebuild_full_frames(valid_norm, mask_hw, H, W):
    Tn, Nv = valid_norm.shape
    out = np.zeros((Tn, H, W), dtype=np.float32)
    out.reshape(Tn, -1)[:, mask_hw.ravel()] = valid_norm
    return out

def fill_nans_timewise(arr_tn):
    x = arr_tn.copy()
    for t in range(1, x.shape[0]):
        m = ~np.isfinite(x[t])
        x[t, m] = x[t-1, m]
    for t in range(x.shape[0]-2, -1, -1):
        m = ~np.isfinite(x[t])
        x[t, m] = x[t+1, m]
    return x

def to_norm_full(frames_phys, valid_idx_flat, scaler, mask_hw, H, W):
    flat = frames_phys.reshape(frames_phys.shape[0], -1)
    valid = flat[:, valid_idx_flat]
    if not np.isfinite(valid).all():
        valid = fill_nans_timewise(valid)
    if not np.isfinite(valid).all():
        mean = scaler.mean_.astype(np.float32)
        valid = np.where(np.isfinite(valid), valid, mean)
    valid_norm = scaler.transform(valid).astype(np.float32)
    return rebuild_full_frames(valid_norm, mask_hw, H, W)

def inv_to_phys_frame(pred_norm_full, valid_idx_flat, scaler, H, W):
    flat = pred_norm_full.reshape(-1)
    pv = flat[valid_idx_flat].reshape(1, -1)
    pv_phys = scaler.inverse_transform(pv)[0].astype(np.float32)
    out = np.full((H, W), np.nan, dtype=np.float32)
    out.reshape(-1)[valid_idx_flat] = pv_phys
    return out

def find_time_index(time64, dt64):
    idx = np.where(time64 == dt64)[0]
    if len(idx) == 0:
        raise ValueError(f"Date {dt64} not found in dataset time axis.")
    return int(idx[0])

os.makedirs(OUT_DIR, exist_ok=True)

# Load checkpoint
ckpt = torch.load(CKPT_PATH, map_location="cpu")
H = int(ckpt["H"]); W = int(ckpt["W"])
hidden_units = tuple(ckpt["hidden_units"])
dropout_rate = float(ckpt["dropout_rate"])
valid_idx_flat = ckpt["valid_idx_flat"].astype(bool)
mask_hw = ckpt["mask_hw"].astype(bool)

scaler = StandardScaler()
scaler.mean_  = ckpt["scaler_mean"].astype(np.float64)
scaler.scale_ = ckpt["scaler_scale"].astype(np.float64)
scaler.var_   = scaler.scale_ ** 2
scaler.n_features_in_ = scaler.mean_.shape[0]

model = CNN2D_LSTM_Model(H, W, hidden_units=hidden_units, dropout_rate=dropout_rate)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE)
model.eval()

print(f"[INFO] Loaded ckpt: {CKPT_PATH} | device={DEVICE}")

# Load SST
ds = xr.open_dataset(DATA_PATH)
sst = ds["sst"].values.astype(np.float32)
lat = ds["lat"].values
lon = ds["lon"].values
time64 = ds["time"].values.astype("datetime64[ns]")

# Auto INIT_END = last available obs day - (MAX_HORIZON-1)
last_obs = pd.to_datetime(time64[-1]).normalize()
init_end = last_obs - pd.Timedelta(days=(MAX_HORIZON - 1))
INIT_END = init_end.strftime("%Y-%m-%d")

init_times = pd.date_range(INIT_START, INIT_END, freq="D")
init_times64 = init_times.values.astype("datetime64[ns]")
n_init = len(init_times)

print(f"[INFO] Init times: {init_times[0].date()} .. {init_times[-1].date()} (N={n_init})")

pred_out = np.full((n_init, MAX_HORIZON, H, W), np.nan, dtype=np.float32)

for ii, init_dt in enumerate(init_times64):
    i_init = find_time_index(time64, init_dt)

    i_hist0 = i_init - SEQ_LEN
    if i_hist0 < 0:
        raise RuntimeError(f"Not enough history for init={str(init_dt)} (need {SEQ_LEN} days).")

    hist_phys = sst[i_hist0:i_init]  # observed up to init-1
    roll_window = to_norm_full(hist_phys, valid_idx_flat, scaler, mask_hw, H, W)  # (SEQ_LEN,H,W)

    for lead in range(MAX_HORIZON):
        x_in = torch.from_numpy(roll_window[None, ...]).to(DEVICE)
        with torch.no_grad():
            pred_norm = model(x_in).cpu().numpy()[0].astype(np.float32)

        pred_phys = inv_to_phys_frame(pred_norm, valid_idx_flat, scaler, H, W)
        pred_out[ii, lead] = pred_phys

        roll_window = np.concatenate([roll_window[1:], pred_norm[None, ...]], axis=0)

    if (ii + 1) % 30 == 0 or ii == 0:
        print(f"[PROGRESS] {ii+1}/{n_init} (init={pd.to_datetime(init_dt).date()})")

pred_da = xr.DataArray(
    pred_out,
    coords={"init_time": init_times, "lead": np.arange(MAX_HORIZON, dtype=int), "lat": lat, "lon": lon},
    dims=("init_time", "lead", "lat", "lon"),
    name="sst_pred"
)
pred_ds = xr.Dataset({"sst_pred": pred_da})
pred_ds["lead"].attrs["description"] = "Forecast lead in days (0..6). Target date = init_time + lead days."
pred_ds.attrs["model"] = "CNN2D-LSTM iterative 1-step rollout"
pred_ds.attrs["seq_len"] = str(SEQ_LEN)

if os.path.exists(OUT_ZARR):
    import shutil
    shutil.rmtree(OUT_ZARR)

pred_ds.to_zarr(OUT_ZARR, mode="w")
print(f"[DONE] Saved Zarr: {OUT_ZARR}")
