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
- percentile threshold (`pctile`, e.g., 99th percentile)
- minimum duration (default in detector)
- optional merging of events separated by small gaps (`joinAcrossGaps`, `maxGap`)

See `alternew_hobday1.py` for the detailed implementation and assumptions. :contentReference[oaicite:2]{index=2}

## Repository structure (suggested)

