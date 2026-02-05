"""
chap5_train2025.py

Train 1-step CNN2D-LSTM (iterative rollout) with seq_len=15.
Uses SST file 1982-2025 but trains ONLY on 1982-2024.
"""

import os
import json
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# =========================
# ONLY PLACE TO EDIT
# =========================
DATA_PATH = "noaa_icesmi_combinefile_FINAL_1res1982_2025.nc"

OUT_DIR   = "CNN2DLSTM_TRAIN_SEQ15_1STEP_1982_2024"
MODEL_DIR = os.path.join(OUT_DIR, "model_save")

SEQ_LEN   = 15

TRAIN_START = np.datetime64("1982-01-01")
TRAIN_END   = np.datetime64("2024-12-31")  # train to 2024 for 2025 demo

LR          = 1e-4
BATCH_SIZE  = 32
EPOCHS      = 500
PATIENCE    = 12
DROPOUT     = 0.1
HIDDEN_UNITS = (256, 128, 64)
VAL_FRACTION = 0.2
# =========================

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device = {device}")

def rebuild_full_frames(valid_norm, mask_hw, H, W):
    Tn, Nv = valid_norm.shape
    out = np.zeros((Tn, H, W), dtype=np.float32)
    out.reshape(Tn, -1)[:, mask_hw.ravel()] = valid_norm
    return out

def create_sequences_2d(arr_norm_full, seq_len):
    X, y = [], []
    T = arr_norm_full.shape[0]
    for i in range(T - seq_len):
        X.append(arr_norm_full[i:i+seq_len])
        y.append(arr_norm_full[i+seq_len])
    return np.asarray(X, np.float32), np.asarray(y, np.float32)

def masked_mse(pred, target, mask_hw_tensor):
    diff2 = (pred - target) ** 2
    wdiff2 = diff2 * mask_hw_tensor
    denom = torch.clamp(mask_hw_tensor.mean(), min=1e-8)
    return wdiff2.mean() / denom

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
        # x: (B,T,H,W)
        B, T, H, W = x.shape
        x = x.unsqueeze(2).reshape(B*T, 1, H, W)
        z = self.encoder(x).view(B, T, -1)
        z, _ = self.lstm1(z)
        z, _ = self.lstm2(z)
        z = self.dropout(z[:, -1, :])
        out = self.fc(z)
        return out.view(B, self.H, self.W)

print("[1/4] Load data")
ds = xr.open_dataset(DATA_PATH)
sst = ds["sst"].values.astype(np.float32)
lat = ds["lat"].values
lon = ds["lon"].values
time64 = ds["time"].values

T, H, W = sst.shape
print(f"[INFO] SST shape: {sst.shape}")

train_mask_time = (time64 >= TRAIN_START) & (time64 <= TRAIN_END)
train_data = sst[train_mask_time]
print(f"[INFO] Train frames: {train_data.shape[0]}")

mask_hw = ~np.isnan(train_data).any(axis=0)
valid_idx_flat = mask_hw.ravel()
Nv = int(valid_idx_flat.sum())
print(f"[INFO] Valid pixels: {Nv} / {H*W}")
if Nv == 0:
    raise RuntimeError("No valid pixels found. Check NaNs.")

print("[2/4] Fit scaler (train only) + build sequences")
train_flat = train_data.reshape(train_data.shape[0], -1)
train_valid = train_flat[:, valid_idx_flat]

scaler = StandardScaler()
train_valid_norm = scaler.fit_transform(train_valid).astype(np.float32)
train_norm_full = rebuild_full_frames(train_valid_norm, mask_hw, H, W)

X_all, y_all = create_sequences_2d(train_norm_full, SEQ_LEN)
N = X_all.shape[0]
n_val = int(np.floor(N * VAL_FRACTION))
n_train = N - n_val

X_train, y_train = X_all[:n_train], y_all[:n_train]
X_val,   y_val   = X_all[n_train:], y_all[n_train:]

train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                          batch_size=BATCH_SIZE, shuffle=False)

print("[3/4] Train")
model = CNN2D_LSTM_Model(H, W, hidden_units=HIDDEN_UNITS, dropout_rate=DROPOUT).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
mask_hw_tensor = torch.from_numpy(mask_hw.astype(np.float32)).to(device)

best_val = float("inf")
pat = 0

best_pth  = os.path.join(MODEL_DIR, "model_cnn2dlstm_seq15_1step_best.pth")
best_ckpt = os.path.join(MODEL_DIR, "ckpt_cnn2dlstm_seq15_1step.pt")
cfg_path  = os.path.join(MODEL_DIR, "config_cnn2dlstm_seq15_1step.json")

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = masked_mse(pred, yb, mask_hw_tensor)
        loss.backward()
        optimizer.step()
        tr_losses.append(loss.item())

    model.eval()
    va_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = masked_mse(pred, yb, mask_hw_tensor)
            va_losses.append(loss.item())

    tr = float(np.mean(tr_losses))
    va = float(np.mean(va_losses))
    print(f"[E{epoch:03d}] train={tr:.6f} | val={va:.6f}")
    scheduler.step(va)

    if va < best_val:
        best_val = va
        pat = 0
        torch.save(model.state_dict(), best_pth)

        ckpt = {
            "seq_len": int(SEQ_LEN),
            "H": int(H), "W": int(W),
            "hidden_units": tuple(HIDDEN_UNITS),
            "dropout_rate": float(DROPOUT),
            "model_state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "mask_hw": mask_hw.astype(np.bool_),
            "valid_idx_flat": valid_idx_flat.astype(np.bool_),
            "lat": lat.astype(np.float32),
            "lon": lon.astype(np.float32),
            "train_start": str(TRAIN_START),
            "train_end": str(TRAIN_END),
            "data_path": DATA_PATH,
        }
        torch.save(ckpt, best_ckpt)

        cfg = {
            "seq_len": int(SEQ_LEN),
            "iterative_1step": True,
            "max_horizon_supported": 7,
            "train_start": str(TRAIN_START),
            "train_end": str(TRAIN_END),
            "data_path": DATA_PATH,
            "notes": "Scaler fitted on 1982–2024 only. Mask from training years only."
        }
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        print(f"  [SAVE] best val={best_val:.6f} -> {best_pth}")
        print(f"  [SAVE] checkpoint -> {best_ckpt}")
    else:
        pat += 1
        if pat >= PATIENCE:
            print(f"[EARLY STOP] patience={PATIENCE} reached.")
            break

print(f"[DONE] Best val loss: {best_val:.6f}")
