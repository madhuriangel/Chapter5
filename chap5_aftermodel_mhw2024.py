"""
chap5_aftermodel_mhw2024.py

Compute daily anomaly / intensity / mhw_flag for CNN2D-LSTM forecasts.

Inputs:
- Observed SST netCDF (for climatology baseline + buffer history)
- Predicted SST Zarr: sst_pred(init_time, lead, lat, lon), lead=0..6

Outputs:
- Zarr with:
    anomaly(init_time, lead, lat, lon)
    intensity(init_time, lead, lat, lon)
    mhw_flag(init_time, lead, lat, lon)

Key design:
- MHW detection uses 7 predicted days (lead 0..6) + a small observed-history buffer
  so we can detect events that begin inside the forecast window.
- Website can show only lead 0..2, but detection is run on all 7 leads.

Method matches your Darmaraki-style settings:
- pctile=99
- windowHalfWidth=5
- smoothPercentileWidth=31
- minDuration=5
- joinAcrossGaps=True
- maxGap=4 with 6-day mean checks around gaps
"""

import os
import numpy as np
import xarray as xr
import pandas as pd

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =========================
# ONLY PLACE TO EDIT
# =========================
OBS_SST_NC  = "noaa_icesmi_combinefile_FINAL_1res1982_2024.nc"

PRED_ZARR   = "CNN2DLSTM_PRECOMP_2024_SEQ15/cnn2dlstm_pred_2024_lead0to6.zarr"
PRED_VAR    = "sst_pred"  # variable inside PRED_ZARR

OUT_DIR     = "CNN2DLSTM_MHW_2024_SEQ15"
OUT_ZARR    = os.path.join(OUT_DIR, "cnn2dlstm_mhw_2024_lead0to6.zarr")

# climatology baseline
CLIM_START_YEAR = 1990
CLIM_END_YEAR   = 2023

# Darmaraki / altered-Hobday-style parameters
PCTILE = 99
WINDOW_HALF_WIDTH = 5
SMOOTH_WIDTH = 31
MIN_DURATION = 5
JOIN_ACROSS_GAPS = True
MAX_GAP = 4

# observed buffer before init_time (helps gap-merge checks & minDuration robustness)
# must be >= 6 to support "6-day mean" check;
BUFFER_BEFORE_DAYS = 30

def build_leap_doy_map(leap_year=2012):
    """Return dict mapping (month,day) -> doy(1..366) using a reference leap year."""
    idx = pd.date_range(f"{leap_year}-01-01", f"{leap_year}-12-31", freq="D")
    m = {(d.month, d.day): int(d.dayofyear) for d in idx}
    return m

LEAP_DOY = build_leap_doy_map(2012)
FEB29_DOY = LEAP_DOY[(2, 29)]  # 60

def dates_to_leap_doy(dts: pd.DatetimeIndex) -> np.ndarray:
    """Map real dates -> leap-year DOY (1..366) via month/day."""
    return np.array([LEAP_DOY[(d.month, d.day)] for d in dts], dtype=np.int16)

# =========================
# Climatology builder (vectorised)
# =========================
def periodic_runavg(arr_doy_hw, w):
    """
    arr_doy_hw: (366, H, W)
    periodic running average along doy dimension.
    """
    assert w % 2 == 1, "SMOOTH_WIDTH must be odd"
    pad = w // 2
    a = np.concatenate([arr_doy_hw[-pad:], arr_doy_hw, arr_doy_hw[:pad]], axis=0)  # (366+2pad, H, W)
    kernel = np.ones(w, dtype=np.float32) / w
    # convolve along axis=0
    out = np.empty_like(arr_doy_hw, dtype=np.float32)
    for j in range(arr_doy_hw.shape[1]):
        for i in range(arr_doy_hw.shape[2]):
            out[:, j, i] = np.convolve(a[:, j, i], kernel, mode="valid")
    return out

def compute_climatology_threshold(sst_da: xr.DataArray,
                                 start_year: int,
                                 end_year: int,
                                 pctile: float,
                                 window_half_width: int,
                                 smooth_width: int):
    """
    Builds:
      seas_doy(366, lat, lon)
      thresh_doy(366, lat, lon)
    using a +/- window_half_width across DOY, pooled across baseline years.
    Feb 29 is interpolated
    """
    time = pd.DatetimeIndex(pd.to_datetime(sst_da["time"].values))
    base_mask = (time.year >= start_year) & (time.year <= end_year)
    sst_base = sst_da.isel(time=np.where(base_mask)[0])

    time_base = pd.DatetimeIndex(pd.to_datetime(sst_base["time"].values))
    doy = dates_to_leap_doy(time_base)  # 1..366

    H = sst_base.sizes["lat"]
    W = sst_base.sizes["lon"]

    seas = np.full((366, H, W), np.nan, dtype=np.float32)
    thr  = np.full((366, H, W), np.nan, dtype=np.float32)

    # Pre-store indices by doy for speed
    by_doy = {}
    for d in range(1, 367):
        by_doy[d] = np.where(doy == d)[0]

    # Build each doy using a window of DOYs (circular)
    for d in range(1, 367):
        if d == FEB29_DOY:
            continue

        win = []
        for k in range(-window_half_width, window_half_width + 1):
            dd = d + k
            if dd < 1:
                dd += 366
            if dd > 366:
                dd -= 366
            win.append(dd)

        idxs = np.concatenate([by_doy[dd] for dd in win if len(by_doy[dd]) > 0], axis=0)
        if idxs.size == 0:
            continue

        vals = sst_base.isel(time=idxs).values  # (N, H, W)
        seas[d - 1] = np.nanmean(vals, axis=0).astype(np.float32)
        thr[d - 1]  = np.nanpercentile(vals, pctile, axis=0).astype(np.float32)

    # Feb 29 interpolation
    thr[FEB29_DOY - 1]  = 0.5 * thr[FEB29_DOY - 2]  + 0.5 * thr[FEB29_DOY]
    seas[FEB29_DOY - 1] = 0.5 * seas[FEB29_DOY - 2] + 0.5 * seas[FEB29_DOY]

    # Smooth (periodic) detect() runavg()
    thr_s  = periodic_runavg(thr,  smooth_width)
    seas_s = periodic_runavg(seas, smooth_width)

    return seas_s.astype(np.float32), thr_s.astype(np.float32)

def clim_for_dates(seas_doy, thr_doy, dates: pd.DatetimeIndex):
    """Return (seas_t, thr_t) each shape (T, H, W) for given dates."""
    doy = dates_to_leap_doy(dates)  # 1..366
    seas_t = seas_doy[doy - 1]
    thr_t  = thr_doy[doy - 1]
    return seas_t.astype(np.float32), thr_t.astype(np.float32)

# =========================
# Darmaraki-style flag detection in 1D series
# (minDuration + optional gap merge with 6-day mean checks) 
# =========================
def detect_mhw_flag_1d(temp, thr,
                       min_duration=5,
                       join_across_gaps=True,
                       max_gap=4):
    """
    temp, thr: 1D arrays (L,)
    returns flag: 0/1 array (L,)
    """
    exceed = (temp - thr) > 0
    exceed = np.where(np.isfinite(exceed), exceed, False)

    # find contiguous True segments
    x = exceed.astype(np.int8)
    dx = np.diff(np.r_[0, x, 0])
    starts = np.where(dx == 1)[0]
    ends   = np.where(dx == -1)[0] - 1
    if starts.size == 0:
        return np.zeros_like(x, dtype=np.uint8)

    # filter by duration
    dur = ends - starts + 1
    keep = dur >= min_duration
    starts = starts[keep]
    ends   = ends[keep]
    if starts.size == 0:
        return np.zeros_like(x, dtype=np.uint8)

    # gap-merge with 6-day mean checks
    if join_across_gaps and starts.size > 1:
        merged_starts = [int(starts[0])]
        merged_ends   = [int(ends[0])]

        k = 1
        while k < len(starts):
            gap = int(starts[k] - merged_ends[-1] - 1)
            if gap <= max_gap:
                # need 6-day mean pre-gap: [start_current-5 .. start_current]
                pre_end = merged_starts[-1]
                pre_start = pre_end - 5

                # and post-gap mean: [end_next .. end_next+5]
                post_start = int(ends[k])
                post_end = post_start + 5

                # bounds safety (if not enough points, do not merge)
                if pre_start < 0 or post_end >= len(temp):
                    can_merge = False
                else:
                    pre_mean = np.mean(temp[pre_start:pre_end + 1])
                    post_mean = np.mean(temp[post_start:post_end + 1])

                    can_merge = (pre_mean > thr[pre_start]) and (post_mean > thr[post_start])

                if can_merge:
                    # merge: extend end to next end
                    merged_ends[-1] = int(ends[k])
                    k += 1
                    continue

            # no merge
            merged_starts.append(int(starts[k]))
            merged_ends.append(int(ends[k]))
            k += 1

        starts = np.array(merged_starts, dtype=int)
        ends   = np.array(merged_ends, dtype=int)

    flag = np.zeros_like(x, dtype=np.uint8)
    for s, e in zip(starts, ends):
        flag[s:e+1] = 1
    return flag

# =========================
# Main
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1/5] Load observed SST")
    ds_obs = xr.open_dataset(OBS_SST_NC)
    sst_obs = ds_obs["sst"]  # (time,lat,lon)

    #lon in [-180,180]
    if float(ds_obs["lon"].max()) > 180:
        ds_obs = ds_obs.assign_coords(lon=(((ds_obs.lon + 180) % 360) - 180)).sortby("lon")
        sst_obs = ds_obs["sst"]

    time_obs = pd.DatetimeIndex(pd.to_datetime(ds_obs["time"].values))

    print("[2/5] Compute climatology (seas/thresh) baseline "
          f"{CLIM_START_YEAR}-{CLIM_END_YEAR} (pct={PCTILE}, w={WINDOW_HALF_WIDTH}, smooth={SMOOTH_WIDTH})")
    seas_doy, thr_doy = compute_climatology_threshold(
        sst_obs, CLIM_START_YEAR, CLIM_END_YEAR, PCTILE, WINDOW_HALF_WIDTH, SMOOTH_WIDTH
    )  # (366,H,W) each
    
    # Compute spatial mean over (lat, lon) for each DOY
    seas_mean_doy = np.nanmean(seas_doy, axis=(1, 2))  # shape (366,)
    thr_mean_doy  = np.nanmean(thr_doy, axis=(1, 2))   # shape (366,)

# Print or save the first few values for check
    print("[INFO] Seasonal mean (°C) for DOY 1–5:", np.round(seas_mean_doy[:5], 2))
    print("[INFO] 99th percentile threshold (°C) for DOY 1–5:", np.round(thr_mean_doy[:5], 2))

    
    # Save seasonal mean and 99th percentile threshold to CSV with month/day labels
    idx = pd.date_range("2012-01-01", "2012-12-31", freq="D")  # Leap year ensures 366 days
    df_clim = pd.DataFrame({
    "DOY": np.arange(1, 367),
    "Month": idx.month,
    "Day": idx.day,
    "seasonal_mean_C": np.round(seas_mean_doy, 3),
    "threshold_99pct_C": np.round(thr_mean_doy, 3),
})
    df_clim.to_csv(os.path.join(OUT_DIR, "doy_climatology_mean_threshold.csv"), index=False)
    print(f"[INFO] Saved DOY climatology to {os.path.join(OUT_DIR, 'doy_climatology_mean_threshold.csv')}")


    # valid ocean mask: finite across baseline years
    base_mask = (time_obs.year >= CLIM_START_YEAR) & (time_obs.year <= CLIM_END_YEAR)
    mask_hw = ~np.isnan(sst_obs.isel(time=np.where(base_mask)[0]).values).any(axis=0)  # (H,W)
    H, W = mask_hw.shape
    valid_flat = mask_hw.ravel()
    idx_valid = np.where(valid_flat)[0]
    Nv = idx_valid.size
    print(f"[INFO] Valid pixels for MHW: {Nv}/{H*W}")

    print("[3/5] Load predictions Zarr")
    ds_pred = xr.open_zarr(PRED_ZARR, consolidated=False)
    if PRED_VAR not in ds_pred:
        raise KeyError(f"'{PRED_VAR}' not found in {PRED_ZARR}. Vars: {list(ds_pred.data_vars)}")
    pred = ds_pred[PRED_VAR]  # (init_time, lead, lat, lon)

    if float(pred["lon"].max()) > 180:
        ds_pred = ds_pred.assign_coords(lon=(((ds_pred.lon + 180) % 360) - 180)).sortby("lon")
        pred = ds_pred[PRED_VAR]

    init_times = pd.DatetimeIndex(pd.to_datetime(pred["init_time"].values))
    leads = pred["lead"].values.astype(int)
    max_lead = int(leads.max())
    if max_lead < 6:
        raise ValueError("This script expects lead 0..6 (7 days).")

    # Pre-allocate outputs (init, lead, H, W)
    n_init = len(init_times)
    out_anom = np.full((n_init, len(leads), H, W), np.nan, dtype=np.float32)
    out_int  = np.full((n_init, len(leads), H, W), np.nan, dtype=np.float32)
    out_flag = np.zeros((n_init, len(leads), H, W), dtype=np.uint8)

    print("[4/5] Loop init_times and compute anomaly/intensity/mhw_flag")
    for ii, init_dt in enumerate(tqdm(init_times, desc="init_time")):
        # forecast target dates for lead 0..6
        target_dates = init_dt + pd.to_timedelta(leads, unit="D")

        # anomaly/intensity per lead uses clim for those target dates
        seas_t, thr_t = clim_for_dates(seas_doy, thr_doy, target_dates)  # (7,H,W)
        sst_fc = pred.sel(init_time=init_dt).values.astype(np.float32)   # (7,H,W)

        out_anom[ii] = sst_fc - seas_t
        out_int[ii]  = np.maximum(0.0, sst_fc - thr_t)

        # build detection series: observed buffer before init + 7 predicted days
        # buffer dates = init_dt - BUFFER_BEFORE_DAYS .. init_dt-1
        buf_start = init_dt - pd.Timedelta(days=BUFFER_BEFORE_DAYS)
        buf_end   = init_dt - pd.Timedelta(days=1)
        buf_dates = pd.date_range(buf_start, buf_end, freq="D")

        # select observed buffer (nearest exact daily)
        try:
            obs_buf = sst_obs.sel(time=buf_dates).values.astype(np.float32)  # (B,H,W)
        except Exception:
            # fallback: nearest per date if indexing not exact
            obs_buf = sst_obs.sel(time=buf_dates, method="nearest").values.astype(np.float32)

        # thresholds for buffer dates + forecast dates
        seas_buf, thr_buf = clim_for_dates(seas_doy, thr_doy, buf_dates)       # (B,H,W)
        thr_series = np.concatenate([thr_buf, thr_t], axis=0)                  # (B+7,H,W)
        temp_series = np.concatenate([obs_buf, sst_fc], axis=0)                # (B+7,H,W)

        # compute flags only on valid pixels
        B = obs_buf.shape[0]
        L = B + len(leads)
        temp_flat = temp_series.reshape(L, -1)[:, idx_valid]   # (L,Nv)
        thr_flat  = thr_series.reshape(L, -1)[:, idx_valid]    # (L,Nv)

        flags_fc_flat = np.zeros((len(leads), Nv), dtype=np.uint8)

        for k in range(Nv):
            t1 = temp_flat[:, k]
            th = thr_flat[:, k]

            # if any all-nan, skip
            if not np.isfinite(t1).any():
                continue

            flag_full = detect_mhw_flag_1d(
                t1, th,
                min_duration=MIN_DURATION,
                join_across_gaps=JOIN_ACROSS_GAPS,
                max_gap=MAX_GAP
            )  # (L,)

            # keep only forecast portion (last 7 days)
            flags_fc_flat[:, k] = flag_full[B:B+len(leads)]

        # write back to grid
        fc_grid = np.zeros((len(leads), H*W), dtype=np.uint8)
        fc_grid[:, idx_valid] = flags_fc_flat
        out_flag[ii] = fc_grid.reshape(len(leads), H, W)

    print("[5/5] Save output Zarr")
    ds_out = xr.Dataset(
        {
            "anomaly": xr.DataArray(
                out_anom,
                coords={"init_time": init_times, "lead": leads, "lat": pred["lat"].values, "lon": pred["lon"].values},
                dims=("init_time", "lead", "lat", "lon"),
            ),
            "intensity": xr.DataArray(
                out_int,
                coords={"init_time": init_times, "lead": leads, "lat": pred["lat"].values, "lon": pred["lon"].values},
                dims=("init_time", "lead", "lat", "lon"),
            ),
            "mhw_flag": xr.DataArray(
                out_flag,
                coords={"init_time": init_times, "lead": leads, "lat": pred["lat"].values, "lon": pred["lon"].values},
                dims=("init_time", "lead", "lat", "lon"),
            ),
        }
    )

    ds_out["lead"].attrs["description"] = "Forecast lead in days (0..6). Target date = init_time + lead."
    ds_out.attrs["climatology_baseline"] = f"{CLIM_START_YEAR}-{CLIM_END_YEAR}"
    ds_out.attrs["pctile"] = str(PCTILE)
    ds_out.attrs["windowHalfWidth"] = str(WINDOW_HALF_WIDTH)
    ds_out.attrs["smoothPercentileWidth"] = str(SMOOTH_WIDTH)
    ds_out.attrs["minDuration"] = str(MIN_DURATION)
    ds_out.attrs["joinAcrossGaps"] = str(JOIN_ACROSS_GAPS)
    ds_out.attrs["maxGap"] = str(MAX_GAP)
    ds_out.attrs["buffer_before_days"] = str(BUFFER_BEFORE_DAYS)

    if os.path.exists(OUT_ZARR):
        import shutil
        shutil.rmtree(OUT_ZARR)

    ds_out.to_zarr(OUT_ZARR, mode="w")
    print(f"[DONE] Saved: {OUT_ZARR}")

if __name__ == "__main__":
    main()
