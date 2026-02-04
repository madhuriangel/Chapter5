
"""
evaluate_forecast_2024_horizons.py

Evaluate precomputed 2024 predictions (sst_pred(init_time, lead, lat, lon))
against observed SST for 2024.
Computes metrics:
- RMSE, MAE, Bias, R2
for each lead (0..6), and for horizons H=3,5,7 (aggregate over leads).

Writes:
- OUT_DIR/metrics_by_lead.csv
- OUT_DIR/metrics_by_horizon.csv
"""

import os
import numpy as np
import xarray as xr
import pandas as pd

# =========================
# ONLY PLACE TO EDIT
# =========================
DATA_PATH = "noaa_icesmi_combinefile_FINAL_1res1982_2024.nc"
PRED_ZARR = "CNN2DLSTM_PRECOMP_2024_SEQ15/cnn2dlstm_pred_2024_lead0to6.zarr"

OUT_DIR = "CNN2DLSTM_EVAL_2024_SEQ15"
HORIZONS = [3, 5, 7]  # compare

# =========================
os.makedirs(OUT_DIR, exist_ok=True)

def rmse(x):
    return float(np.sqrt(np.nanmean(x**2)))

def mae(x):
    return float(np.nanmean(np.abs(x)))

def r2_score(y_true, y_pred):
    numer = float(np.nansum((y_true - y_pred) ** 2))
    denom = float(np.nansum((y_true - np.nanmean(y_true)) ** 2))
    return float(1 - numer / denom) if denom != 0 else np.nan

# load data
ds_obs = xr.open_dataset(DATA_PATH)
sst_obs = ds_obs["sst"]  # (time,lat,lon)
time_obs = pd.DatetimeIndex(pd.to_datetime(ds_obs["time"].values))

ds_pred = xr.open_zarr(PRED_ZARR, consolidated=False)
sst_pred = ds_pred["sst_pred"]  # (init_time,lead,lat,lon)

init_times = pd.DatetimeIndex(pd.to_datetime(sst_pred["init_time"].values))
leads = sst_pred["lead"].values.astype(int)
max_lead = int(leads.max())

# determine valid evaluation mask: use non-NaN of observed & predicted (per comparison)
# We'll compute metrics on overlapping finite values only.

# ==============
# Metrics by lead
# ==============
rows = []
for lead in leads:
    # target date = init_time + lead
    target_times = init_times + pd.to_timedelta(int(lead), unit="D")

    # align with obs time index
    # keep only those targets that exist in obs time
    m = target_times.isin(time_obs)
    if m.sum() == 0:
        continue

    it_sel = init_times[m]
    tt_sel = target_times[m]

    pred_sel = sst_pred.sel(init_time=it_sel, lead=int(lead))  # (init,lat,lon)
    obs_sel  = sst_obs.sel(time=tt_sel)                        # (init,lat,lon) aligned by order

    # convert to numpy and flatten
    p = pred_sel.values.reshape(len(it_sel), -1)
    o = obs_sel.values.reshape(len(it_sel), -1)

    # finite mask
    mm = np.isfinite(p) & np.isfinite(o)
    if mm.sum() == 0:
        continue

    diff = (o - p)[mm]
    y_true = o[mm]
    y_pred = p[mm]

    rows.append({
        "lead": int(lead),
        "n": int(mm.sum()),
        "RMSE": rmse(diff),
        "MAE": mae(diff),
        "Bias(mean_o_minus_p)": float(np.nanmean(diff)),
        "R2": r2_score(y_true, y_pred)
    })

df_lead = pd.DataFrame(rows).sort_values("lead")
df_lead.to_csv(os.path.join(OUT_DIR, "metrics_by_lead.csv"), index=False)
print("[DONE] metrics_by_lead.csv written")

# ==================
# Metrics by horizon
# ==================
rows_h = []
for H in HORIZONS:
    use_leads = [l for l in leads if l < H]
    if len(use_leads) == 0:
        continue

    diffs = []
    yts = []
    yps = []
    n_total = 0

    for lead in use_leads:
        target_times = init_times + pd.to_timedelta(int(lead), unit="D")
        m = target_times.isin(time_obs)
        if m.sum() == 0:
            continue

        it_sel = init_times[m]
        tt_sel = target_times[m]

        pred_sel = sst_pred.sel(init_time=it_sel, lead=int(lead))
        obs_sel  = sst_obs.sel(time=tt_sel)

        p = pred_sel.values.reshape(len(it_sel), -1)
        o = obs_sel.values.reshape(len(it_sel), -1)

        mm = np.isfinite(p) & np.isfinite(o)
        if mm.sum() == 0:
            continue

        diff = (o - p)[mm]
        diffs.append(diff)
        yts.append(o[mm])
        yps.append(p[mm])
        n_total += int(mm.sum())

    if n_total == 0:
        continue

    diff_all = np.concatenate(diffs)
    yt_all = np.concatenate(yts)
    yp_all = np.concatenate(yps)

    rows_h.append({
        "horizon": int(H),
        "n": int(n_total),
        "RMSE": rmse(diff_all),
        "MAE": mae(diff_all),
        "Bias(mean_o_minus_p)": float(np.nanmean(diff_all)),
        "R2": r2_score(yt_all, yp_all)
    })

df_h = pd.DataFrame(rows_h).sort_values("horizon")
df_h.to_csv(os.path.join(OUT_DIR, "metrics_by_horizon.csv"), index=False)
print("[DONE] metrics_by_horizon.csv written")

print("\n===== Summary =====")
print(df_h.to_string(index=False))
