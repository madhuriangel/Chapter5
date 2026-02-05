"""
Create a daily Marine Heatwave (MHW) archive for fast web visualisation.

This script computes, for every grid cell in a gridded SST dataset:

  1) mhw_flag  : daily binary flag (1 = day is within a detected MHW event, else 0)
  2) anomaly   : SST anomaly relative to the seasonal climatology (sst - seas)
  3) intensity : positive exceedance above the MHW threshold (max(sst - thresh, 0))

The resulting daily fields are saved as a Zarr store for efficient slicing by
time and subsetting by lat/lon (useful for Flask / web dashboards).

Detection method
----------------
MHW events are detected using mhw.detect() from `alternew_hobday1.py` 
an adaptation of the Hobday marine heatwave definition with:

- percentile threshold defined by `pctile` (here: 99th percentile)
- day-of-year climatology computed using a moving window (`windowHalfWidth`)
- optional percentile smoothing
- minimum event duration (`minDuration`)
- optional joining across small gaps (`joinAcrossGaps`, `maxGap`)

This script calls detect() independently for each grid point, then expands event
start/end indices into a daily flag time series (mhw_flag).

Inputs
------
- NetCDF file containing:
    - sst(time, lat, lon) in degrees C
    - time coordinate interpretable by pandas/xarray
    - lat, lon coordinates

Outputs
-------
- Zarr directory (OUT_ZARR) containing an xarray Dataset with variables:
    - mhw_flag(time, lat, lon)   uint8
    - anomaly(time, lat, lon)    float32
    - intensity(time, lat, lon)  float32

Notes / assumptions
-------------------
- The climatology period is defined by year integers [start_year, end_year]
  and must exist within the dataset time range.

Project: Chapter 5 – Web-enabled SST/MHW monitoring around Ireland
"""

import numpy as np
import xarray as xr
import pandas as pd
from datetime import date
import alternew_hobday1 as mhw  

DATA_PATH = "noaa_icesmi_combinefile_FINAL_1res1982_2024.nc"
OUT_ZARR  = "mhw_daily_1982_2024.zarr"

CLIM_PERIOD = [1991, 2024]   
PCTILE = 99

ds = xr.open_dataset(DATA_PATH)
sst = ds["sst"].values  # (time, lat, lon)
time64 = pd.to_datetime(ds["time"].values)
lat = ds["lat"].values
lon = ds["lon"].values

# `alternew_hobday1.detect()` expects time in Python ordinal format:
#   date.toordinal() = integer day count used internally by the algorithm.
# We convert each timestamp to an ordinal day to match the detector API.
# ordinal time required by detect()
time_o = np.array([date(t.year, t.month, t.day).toordinal() for t in time64.to_pydatetime()], dtype=int)

T, H, W = sst.shape
mhw_flag = np.zeros((T, H, W), dtype=np.uint8)
anomaly  = np.full((T, H, W), np.nan, dtype=np.float32)
intensity = np.full((T, H, W), np.nan, dtype=np.float32)

for j in range(H):
    print(f"lat row {j+1}/{H}")
    for i in range(W):
        sst1 = sst[:, j, i]

        # skip invalid cells 
        if (not np.isfinite(np.nanmean(sst1))) or ((sst1 < -1).sum() > 0):
            continue

        mhws, clim = mhw.detect(
            time_o, sst1,
            climatologyPeriod=CLIM_PERIOD,
            pctile=PCTILE,
            Ly=False
        )

        # anomaly vs seasonal climatology (seas) 
        anom1 = sst1 - clim["seas"]
        anomaly[:, j, i] = anom1.astype(np.float32)

        # intensity above threshold
        inten1 = sst1 - clim["thresh"]
        intensity[:, j, i] = np.maximum(inten1, 0).astype(np.float32)

        # event-days flag (minDuration/gap merging in detect)
        flag1 = np.zeros(T, dtype=np.uint8)
        for ev in range(mhws["n_events"]):
            s = mhws["index_start"][ev]
            e = mhws["index_end"][ev]
            flag1[s:e+1] = 1
        mhw_flag[:, j, i] = flag1

out = xr.Dataset(
    data_vars=dict(
        mhw_flag=(("time", "lat", "lon"), mhw_flag),
        anomaly=(("time", "lat", "lon"), anomaly),
        intensity=(("time", "lat", "lon"), intensity),
    ),
    coords=dict(time=time64, lat=lat, lon=lon),
    attrs=dict(
        climatology_period=f"{CLIM_PERIOD[0]}-{CLIM_PERIOD[1]}",
        pctile=str(PCTILE),
        note="Daily archive for web visualisation (MHW flag, anomaly, intensity).",
    )
)

# Zarr is best for web-style slicing by date (fast reads)
out.to_zarr(OUT_ZARR, mode="w", consolidated=True)
print("Saved:", OUT_ZARR)
