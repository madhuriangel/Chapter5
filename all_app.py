# app_all_mhw_2024_2025.py
# Single Flask app hosting:
#  1) Historical MHW Explorer (1982–2024) + HAB overlay (date-filtered)
#  2) CNN2D-LSTM Forecast + MHW Explorer (2024 test) showing only lead 0..2
#  3) CNN2D-LSTM Forecast + MHW Explorer (2025 demo/live-style) showing only lead 0..2
#
# Run:
#   python app_all_mhw_2024_2025.py
# Open:
#   http://127.0.0.1:5000

import io
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from flask import Flask, render_template_string, send_file, request

# =========================
# ONLY PLACE TO EDIT
# =========================

# ---------- Historical (1982–2024) ----------
HIST_ZARR = "web_mhw_analysis_alterhob/mhw_daily_1982_2024.zarr"

# HAB overlay CSVs (date-ranged)
HAB_2021_CSV = "mhw_analysis_alterhob/hab2021_conv.csv"
HAB_2023_CSV = "mhw_analysis_alterhob/hab2023_conv.csv"

# ---------- Forecast configs ----------
# IMPORTANT: these Zarr folders must exist (precomputed earlier)

FORECAST_CONFIGS = {
    # internal key -> config
    "fc2024": {
        "label": "Forecast (2024 test)",
        "page_title": "SST & MHW Prediction (2024 test)",
        "pred_zarr": "CNN2DLSTM_PRECOMP_2024_SEQ15/cnn2dlstm_pred_2024_lead0to6.zarr",
        "mhw_zarr":  "CNN2DLSTM_MHW_2024_SEQ15/cnn2dlstm_mhw_2024_lead0to6.zarr",
        "pred_var": "sst_pred",
        "baseline_text": "1990–2023",
    },
    "fc2025": {
        "label": "Forecast (2025 demo)",
        "page_title": "SST & MHW Prediction (2025 demo live)",
        "pred_zarr": "CNN2DLSTM_PRECOMP_2025_SEQ15/cnn2dlstm_pred_2025_lead0to6.zarr",
        "mhw_zarr":  "CNN2DLSTM_MHW_2025_SEQ15/cnn2dlstm_mhw_2025_lead0to6.zarr",
        "pred_var": "sst_pred",
        "baseline_text": "1991–2024",
    },
}

# Website should show only these leads (even though detection used lead 0..6)
LEADS_TO_SHOW = [0, 1, 2]

# Plot sizing
FIGSIZE_HIST = (7, 5)
FIGSIZE_FCST = (7, 5)

# Display size on page (CSS max-width)
MAX_WIDTH_PX = 850

# Binary mhw_flag colors
NO_MHW_COLOR = "#1f78b4"
MHW_COLOR    = "#ff6f00"

# =========================
# Flask app
# =========================
app = Flask(__name__)

# =========================
# Utilities
# =========================
def _wrap_lon_360_to_180(lon):
    return ((lon + 180) % 360) - 180

def _standardize_lon(ds, lon_name="lon"):
    if lon_name in ds.coords:
        if float(ds[lon_name].max()) > 180:
            ds = ds.assign_coords(**{lon_name: _wrap_lon_360_to_180(ds[lon_name])}).sortby(lon_name)
    return ds

def _set_geo_style(ax, lon0, lon1, lat0, lat1):
    ax.add_feature(cfeature.LAND.with_scale("10m"), zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.5, zorder=2)
    ax.set_extent([lon0, lon1, lat0, lat1], crs=ccrs.PlateCarree())

    xticks = np.arange(np.floor(lon0), np.ceil(lon1) + 1, 1)
    yticks = np.arange(np.floor(lat0), np.ceil(lat1) + 1, 1)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f"))

def _add_colorbar(fig, ax, pcm, *, is_flag=False, label=""):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15, axes_class=plt.Axes)
    if is_flag:
        cbar = fig.colorbar(pcm, cax=cax, ticks=[0, 1])
        cbar.ax.set_yticklabels(["No MHW", "MHW"])
    else:
        cbar = fig.colorbar(pcm, cax=cax)
        if label:
            cbar.set_label(label)
    return cbar

# =========================
# Load datasets once (startup)
# =========================

# ----- Historical -----
ds_hist = xr.open_zarr(HIST_ZARR, consolidated=True)
ds_hist = _standardize_lon(ds_hist)

HIST_VAR_NAMES = ["mhw_flag", "anomaly", "intensity"]
missing = [v for v in HIST_VAR_NAMES if v not in ds_hist.data_vars]
if missing:
    raise KeyError(f"[HIST] Missing vars {missing}. Available: {list(ds_hist.data_vars)}")

hist_time = pd.DatetimeIndex(pd.to_datetime(ds_hist["time"].values))
HIST_MIN_DATE = hist_time.min().strftime("%Y-%m-%d")
HIST_MAX_DATE = hist_time.max().strftime("%Y-%m-%d")

HIST_LABEL_TO_VAR = {"MHW Flag": "mhw_flag", "Anomaly": "anomaly", "Intensity": "intensity"}
HIST_VAR_TO_LABEL = {v: k for k, v in HIST_LABEL_TO_VAR.items()}

# ----- Forecasts (2024 + 2025) -----
FC_LAYER_LABEL_TO_KEY = {
    "SST Forecast": "sst",
    "Anomaly": "anomaly",
    "Intensity": "intensity",
    "MHW Flag": "mhw_flag",
}
FC_KEY_TO_LAYER_LABEL = {v: k for k, v in FC_LAYER_LABEL_TO_KEY.items()}

fc_data = {}
for fc_id, cfg in FORECAST_CONFIGS.items():
    ds_pred = xr.open_zarr(cfg["pred_zarr"], consolidated=False)
    ds_mhw  = xr.open_zarr(cfg["mhw_zarr"],  consolidated=False)

    ds_pred = _standardize_lon(ds_pred)
    ds_mhw  = _standardize_lon(ds_mhw)

    init_times = pd.DatetimeIndex(pd.to_datetime(ds_pred["init_time"].values))
    fc_data[fc_id] = {
        "cfg": cfg,
        "ds_pred": ds_pred,
        "ds_mhw": ds_mhw,
        "init_times": init_times,
        "min_init": init_times.min().strftime("%Y-%m-%d"),
        "max_init": init_times.max().strftime("%Y-%m-%d"),
    }

# =========================
# HAB loader (date-ranged)
# =========================
def _load_hab_csv(path: str) -> pd.DataFrame:
    """
    Expected columns in hab*_conv.csv:
      Latitude, Longitude, periodstart_date, periodend_date
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["lat", "lon", "start", "end"])

    required = {"Latitude", "Longitude", "periodstart_date", "periodend_date"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame(columns=["lat", "lon", "start", "end"])

    out = pd.DataFrame()
    out["lat"] = pd.to_numeric(df["Latitude"], errors="coerce")
    out["lon"] = pd.to_numeric(df["Longitude"], errors="coerce")
    out["start"] = pd.to_datetime(df["periodstart_date"], errors="coerce", dayfirst=True)
    out["end"]   = pd.to_datetime(df["periodend_date"],   errors="coerce", dayfirst=True)
    out = out.dropna(subset=["lat", "lon", "start", "end"]).reset_index(drop=True)
    out["lon"] = out["lon"].apply(_wrap_lon_360_to_180)
    return out

def _hab_active_on_date(hab_df: pd.DataFrame, dt: pd.Timestamp) -> pd.DataFrame:
    if hab_df.empty:
        return hab_df
    m = (hab_df["start"] <= dt) & (dt <= hab_df["end"])
    return hab_df.loc[m].copy()

HAB_SITES = {
    2021: _load_hab_csv(HAB_2021_CSV),
    2023: _load_hab_csv(HAB_2023_CSV),
}
HIST_OVERLAY_OPTIONS = ["None", "HAB/Toxicity sites"]

# =========================
# HTML Template
# =========================
MODES = ["Historical (1982-2024)"] + [FORECAST_CONFIGS[k]["label"] for k in FORECAST_CONFIGS.keys()]

TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>{{ page_title }}</title>
  <meta charset="utf-8"/>
</head>
<body style="font-family: Arial, sans-serif; margin: 20px;">
  <h1>{{ page_title }}</h1>

  <form action="/" method="get">
    <label for="mode"><b>Mode:</b></label>
    <select id="mode" name="mode" onchange="this.form.submit()">
      {% for m in modes %}
        <option value="{{m}}" {% if m==mode %}selected{% endif %}>{{m}}</option>
      {% endfor %}
    </select>

    {% if is_hist %}
      <label for="date" style="margin-left: 12px;"><b>Choose date:</b></label>
      <input type="date" id="date" name="date"
             value="{{ hist_date_value }}"
             min="{{ hist_min_date }}"
             max="{{ hist_max_date }}"
             onchange="this.form.submit()">

      <label for="layer" style="margin-left: 12px;"><b>Layer:</b></label>
      <select id="layer" name="layer" onchange="this.form.submit()">
        {% for lab in hist_layer_labels %}
          <option value="{{lab}}" {% if lab==hist_layer_label %}selected{% endif %}>{{lab}}</option>
        {% endfor %}
      </select>

      <label for="overlay" style="margin-left: 12px;"><b>Overlay:</b></label>
      <select id="overlay" name="overlay" onchange="this.form.submit()">
        {% for opt in hist_overlay_options %}
          <option value="{{opt}}" {% if opt==hist_overlay %}selected{% endif %}>{{opt}}</option>
        {% endfor %}
      </select>
    {% else %}
      <label for="init" style="margin-left: 12px;"><b>Init date:</b></label>
      <input type="date" id="init" name="init"
             value="{{ fc_init_value }}"
             min="{{ fc_min_init }}"
             max="{{ fc_max_init }}"
             onchange="this.form.submit()">

      <label for="lead" style="margin-left: 12px;"><b>Lead:</b></label>
      <select id="lead" name="lead" onchange="this.form.submit()">
        {% for ld in fc_leads_to_show %}
          <option value="{{ld}}" {% if ld==fc_lead %}selected{% endif %}>{{ld}}</option>
        {% endfor %}
      </select>

      <label for="layer" style="margin-left: 12px;"><b>Layer:</b></label>
      <select id="layer" name="layer" onchange="this.form.submit()">
        {% for lab in fc_layer_labels %}
          <option value="{{lab}}" {% if lab==fc_layer_label %}selected{% endif %}>{{lab}}</option>
        {% endfor %}
      </select>
    {% endif %}
  </form>

  <h2>{{ headline }}</h2>

  <img src="{{ plot_url }}"
       alt="map"
       style="max-width:{{ max_width_px }}px; width:100%; border: 1px solid #ddd;">

  <p style="margin-top: 10px;">
    <a href="{{ dl_nc_url }}">Download NetCDF</a> |
    <a href="{{ dl_csv_url }}">Download CSV</a> |
    <a href="{{ dl_json_url }}">Download JSON</a>
  </p>

  <p style="color:#666; font-size: 0.80em; max-width: {{ max_width_px }}px;">
    {{ footer_html|safe }}
  </p>
</body>
</html>
"""

# =========================
# Safety helpers
# =========================
def _safe_mode(mode: str) -> str:
    return mode if mode in MODES else MODES[0]

def _nearest_hist_date_str(date_str: str) -> str:
    try:
        sel = pd.to_datetime(date_str)
    except Exception:
        sel = pd.to_datetime(HIST_MIN_DATE)
    if sel in hist_time:
        nearest = sel
    else:
        pos = hist_time.get_indexer([sel], method="nearest")[0]
        nearest = hist_time[pos]
    return nearest.strftime("%Y-%m-%d")

def _safe_hist_layer_label(label: str) -> str:
    return label if label in HIST_LABEL_TO_VAR else "MHW Flag"

def _safe_hist_overlay(overlay: str) -> str:
    return overlay if overlay in HIST_OVERLAY_OPTIONS else "None"

def _safe_fc_layer_label(label: str) -> str:
    return label if label in FC_LAYER_LABEL_TO_KEY else "MHW Flag"

def _safe_fc_key(key: str) -> str:
    return key if key in FC_KEY_TO_LAYER_LABEL else "mhw_flag"

def _safe_fc_lead(lead) -> int:
    try:
        lead = int(lead)
    except Exception:
        lead = LEADS_TO_SHOW[0]
    return lead if lead in LEADS_TO_SHOW else LEADS_TO_SHOW[0]

def _mode_to_fc_id(mode_label: str):
    for fc_id, cfg in FORECAST_CONFIGS.items():
        if cfg["label"] == mode_label:
            return fc_id
    return None

def _nearest_fc_init_str(fc_id: str, init_str: str) -> str:
    init_times = fc_data[fc_id]["init_times"]
    min_init = fc_data[fc_id]["min_init"]
    try:
        d = pd.to_datetime(init_str)
    except Exception:
        d = pd.to_datetime(min_init)
    if d in init_times:
        return d.strftime("%Y-%m-%d")
    pos = init_times.get_indexer([d], method="nearest")[0]
    return init_times[pos].strftime("%Y-%m-%d")

# =========================
# Main index
# =========================
@app.route("/")
def index():
    mode = _safe_mode(request.args.get("mode", MODES[0]))

    # ---------- Historical ----------
    if mode == "Historical (1982-2024)":
        date_value = _nearest_hist_date_str(request.args.get("date", HIST_MIN_DATE))
        layer_label = _safe_hist_layer_label(request.args.get("layer", "MHW Flag"))
        overlay = _safe_hist_overlay(request.args.get("overlay", "None"))
        varname = HIST_LABEL_TO_VAR[layer_label]

        dt = pd.to_datetime(date_value)
        headline = f"{layer_label} - {dt.strftime('%d/%m/%Y')}"

        plot_url = f"/hist/plot/{varname}/{date_value}.png?overlay={overlay}"
        dl_nc_url = f"/hist/download/netcdf?varname={varname}&date={date_value}"
        dl_csv_url = f"/hist/download/csv?varname={varname}&date={date_value}"
        dl_json_url = f"/hist/download/json?varname={varname}&date={date_value}"

        footer_html = (
            "<b>Definitions:</b> "
            "<b>Anomaly</b> = SST − climatology (°C); "
            "<b>Intensity</b> = max(0, SST − threshold) (°C); "
            "<b>MHW Flag</b> = 1 indicates Marine Heatwave (MHW), 0 indicates no MHW. "
            "<b>HAB/Toxicity sites</b> (yellow triangles) are shown only when the selected date "
            "falls within the recorded HAB period for that site."
        )

        return render_template_string(
            TEMPLATE,
            page_title="Historical MHW detection (1982–2024)",
            modes=MODES,
            mode=mode,
            is_hist=True,
            # hist fields
            hist_date_value=date_value,
            hist_min_date=HIST_MIN_DATE,
            hist_max_date=HIST_MAX_DATE,
            hist_layer_label=layer_label,
            hist_layer_labels=list(HIST_LABEL_TO_VAR.keys()),
            hist_overlay=overlay,
            hist_overlay_options=HIST_OVERLAY_OPTIONS,
            # fc placeholders
            fc_init_value="",
            fc_min_init="",
            fc_max_init="",
            fc_lead=LEADS_TO_SHOW[0],
            fc_leads_to_show=LEADS_TO_SHOW,
            fc_layer_label="MHW Flag",
            fc_layer_labels=list(FC_LAYER_LABEL_TO_KEY.keys()),
            # shared
            headline=headline,
            plot_url=plot_url,
            dl_nc_url=dl_nc_url,
            dl_csv_url=dl_csv_url,
            dl_json_url=dl_json_url,
            footer_html=footer_html,
            max_width_px=MAX_WIDTH_PX
        )

    # ---------- Forecast (2024 or 2025) ----------
    fc_id = _mode_to_fc_id(mode)
    if fc_id is None:
        fc_id = list(FORECAST_CONFIGS.keys())[0]

    cfg = fc_data[fc_id]["cfg"]
    min_init = fc_data[fc_id]["min_init"]
    max_init = fc_data[fc_id]["max_init"]

    init_value = _nearest_fc_init_str(fc_id, request.args.get("init", min_init))
    lead = _safe_fc_lead(request.args.get("lead", LEADS_TO_SHOW[0]))
    layer_label = _safe_fc_layer_label(request.args.get("layer", "MHW Flag"))
    key = FC_LAYER_LABEL_TO_KEY[layer_label]

    init_dt = pd.to_datetime(init_value)
    target_dt = init_dt + pd.to_timedelta(int(lead), unit="D")

    headline = f"{layer_label} | Init={init_dt.strftime('%d/%m/%Y')} | Lead={lead} | Target={target_dt.strftime('%d/%m/%Y')}"

    plot_url = f"/fc/{fc_id}/plot/{key}/{init_value}/{lead}.png"
    dl_nc_url = f"/fc/{fc_id}/download/netcdf?key={key}&init={init_value}&lead={lead}"
    dl_csv_url = f"/fc/{fc_id}/download/csv?key={key}&init={init_value}&lead={lead}"
    dl_json_url = f"/fc/{fc_id}/download/json?key={key}&init={init_value}&lead={lead}"

    footer_html = (
        "<b>What the layers mean:</b>"
        "<br>• <b>SST Forecast</b>: CNN2D-LSTM predicted SST for the target date."
        f"<br>• <b>Anomaly</b>: predicted SST − daily climatology (baseline {cfg['baseline_text']})."
        "<br>• <b>Intensity</b>: max(0, predicted SST − 99th percentile threshold)."
        "<br>• <b>MHW Flag</b>: 1 if the day belongs to a detected MHW event (min 5 days), 0 otherwise."
        "<br>Detection is run on a 7-day forecast window (lead 0..6) + recent observed history; "
        "the website shows only lead day 0–2."
    )

    return render_template_string(
        TEMPLATE,
        page_title=cfg["page_title"],
        modes=MODES,
        mode=mode,
        is_hist=False,
        # hist placeholders
        hist_date_value=HIST_MIN_DATE,
        hist_min_date=HIST_MIN_DATE,
        hist_max_date=HIST_MAX_DATE,
        hist_layer_label="MHW Flag",
        hist_layer_labels=list(HIST_LABEL_TO_VAR.keys()),
        hist_overlay="None",
        hist_overlay_options=HIST_OVERLAY_OPTIONS,
        # forecast fields
        fc_init_value=init_value,
        fc_min_init=min_init,
        fc_max_init=max_init,
        fc_lead=int(lead),
        fc_leads_to_show=LEADS_TO_SHOW,
        fc_layer_label=layer_label,
        fc_layer_labels=list(FC_LAYER_LABEL_TO_KEY.keys()),
        # shared
        headline=headline,
        plot_url=plot_url,
        dl_nc_url=dl_nc_url,
        dl_csv_url=dl_csv_url,
        dl_json_url=dl_json_url,
        footer_html=footer_html,
        max_width_px=MAX_WIDTH_PX
    )

# =========================
# HISTORICAL ROUTES
# =========================
@app.route("/hist/plot/<varname>/<date>.png")
def hist_plot_png(varname, date):
    if varname not in HIST_VAR_NAMES:
        varname = "mhw_flag"

    date = _nearest_hist_date_str(date)
    overlay = _safe_hist_overlay(request.args.get("overlay", "None"))
    dt = pd.to_datetime(date)

    da = ds_hist[varname].sel(time=dt, method="nearest")

    fig = plt.figure(figsize=FIGSIZE_HIST)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    field = da.values

    if varname == "mhw_flag":
        cmap = ListedColormap([NO_MHW_COLOR, MHW_COLOR])
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
        pcm = ax.pcolormesh(da["lon"].values, da["lat"].values, field,
                            transform=ccrs.PlateCarree(),
                            shading="auto", cmap=cmap, norm=norm)
    else:
        pcm = ax.pcolormesh(da["lon"].values, da["lat"].values, field,
                            transform=ccrs.PlateCarree(),
                            shading="auto", cmap="viridis")

    lon0, lon1 = float(da["lon"].min()), float(da["lon"].max())
    lat0, lat1 = float(da["lat"].min()), float(da["lat"].max())
    _set_geo_style(ax, lon0, lon1, lat0, lat1)

    # ---- No title on image (only HTML headline) ----

    # HAB overlay (date-filtered)
    show_legend = False
    if overlay == "HAB/Toxicity sites" and dt.year in HAB_SITES:
        hab_active = _hab_active_on_date(HAB_SITES[dt.year], dt)
        if not hab_active.empty:
            ax.scatter(hab_active["lon"], hab_active["lat"],
                       transform=ccrs.PlateCarree(),
                       zorder=8, s=70, marker="^",
                       facecolor="gold", edgecolor="black",
                       linewidth=0.5, label="HAB/Toxicity sites")
            show_legend = True
    if show_legend:
        ax.legend(loc="upper left", frameon=True)

    _add_colorbar(fig, ax, pcm, is_flag=(varname == "mhw_flag"), label=HIST_VAR_TO_LABEL[varname])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

@app.route("/hist/download/netcdf")
def hist_dl_nc():
    varname = request.args.get("varname", "mhw_flag")
    if varname not in HIST_VAR_NAMES:
        varname = "mhw_flag"
    date = _nearest_hist_date_str(request.args.get("date", HIST_MIN_DATE))
    dt = pd.to_datetime(date)

    da = ds_hist[varname].sel(time=dt, method="nearest").rename(varname)
    out = da.to_dataset(name=varname)

    nc_bytes = out.to_netcdf()
    buf = io.BytesIO(nc_bytes)
    buf.seek(0)
    tag = pd.to_datetime(date).strftime("%Y%m%d")
    return send_file(buf, as_attachment=True,
                     download_name=f"hist_{varname}_{tag}.nc",
                     mimetype="application/x-netcdf")

@app.route("/hist/download/csv")
def hist_dl_csv():
    varname = request.args.get("varname", "mhw_flag")
    if varname not in HIST_VAR_NAMES:
        varname = "mhw_flag"
    date = _nearest_hist_date_str(request.args.get("date", HIST_MIN_DATE))
    dt = pd.to_datetime(date)

    da = ds_hist[varname].sel(time=dt, method="nearest").rename(varname)
    df = da.to_dataframe().reset_index()

    s = io.StringIO()
    df.to_csv(s, index=False)
    s.seek(0)
    tag = pd.to_datetime(date).strftime("%Y%m%d")
    return send_file(io.BytesIO(s.getvalue().encode()),
                     as_attachment=True,
                     download_name=f"hist_{varname}_{tag}.csv",
                     mimetype="text/csv")

@app.route("/hist/download/json")
def hist_dl_json():
    varname = request.args.get("varname", "mhw_flag")
    if varname not in HIST_VAR_NAMES:
        varname = "mhw_flag"
    date = _nearest_hist_date_str(request.args.get("date", HIST_MIN_DATE))
    dt = pd.to_datetime(date)

    da = ds_hist[varname].sel(time=dt, method="nearest").rename(varname)
    df = da.to_dataframe().reset_index()

    js = df.to_json(orient="records")
    tag = pd.to_datetime(date).strftime("%Y%m%d")
    return send_file(io.BytesIO(js.encode()),
                     as_attachment=True,
                     download_name=f"hist_{varname}_{tag}.json",
                     mimetype="application/json")

# =========================
# FORECAST ROUTES (2024 + 2025)
# =========================
def _fc_get_field(fc_id: str, key: str, init_str: str, lead: int):
    cfg = fc_data[fc_id]["cfg"]
    ds_pred = fc_data[fc_id]["ds_pred"]
    ds_mhw  = fc_data[fc_id]["ds_mhw"]

    init_str = _nearest_fc_init_str(fc_id, init_str)
    lead = _safe_fc_lead(lead)
    key = _safe_fc_key(key)

    init_dt = pd.to_datetime(init_str)
    target_dt = init_dt + pd.to_timedelta(int(lead), unit="D")

    if key == "sst":
        da = ds_pred[cfg["pred_var"]].sel(init_time=init_dt, lead=int(lead))
    else:
        da = ds_mhw[key].sel(init_time=init_dt, lead=int(lead))

    return da, init_dt, target_dt

@app.route("/fc/<fc_id>/plot/<key>/<init>/<int:lead>.png")
def fc_plot_png(fc_id, key, init, lead):
    if fc_id not in fc_data:
        fc_id = list(fc_data.keys())[0]

    da, init_dt, target_dt = _fc_get_field(fc_id, key, init, lead)

    fig = plt.figure(figsize=FIGSIZE_FCST)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    field = da.values

    if key == "mhw_flag":
        cmap = ListedColormap([NO_MHW_COLOR, MHW_COLOR])
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
        pcm = ax.pcolormesh(da["lon"].values, da["lat"].values, field,
                            transform=ccrs.PlateCarree(),
                            shading="auto", cmap=cmap, norm=norm)
    else:
        pcm = ax.pcolormesh(da["lon"].values, da["lat"].values, field,
                            transform=ccrs.PlateCarree(),
                            shading="auto", cmap="viridis")

    lon0, lon1 = float(da["lon"].min()), float(da["lon"].max())
    lat0, lat1 = float(da["lat"].min()), float(da["lat"].max())
    _set_geo_style(ax, lon0, lon1, lat0, lat1)

    # ---- No title on image (only HTML headline) ----

    _add_colorbar(fig, ax, pcm, is_flag=(key == "mhw_flag"), label=FC_KEY_TO_LAYER_LABEL.get(key, ""))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

@app.route("/fc/<fc_id>/download/netcdf")
def fc_dl_nc(fc_id):
    if fc_id not in fc_data:
        fc_id = list(fc_data.keys())[0]

    key = _safe_fc_key(request.args.get("key", "mhw_flag"))
    init = request.args.get("init", fc_data[fc_id]["min_init"])
    init = _nearest_fc_init_str(fc_id, init)
    lead = _safe_fc_lead(request.args.get("lead", LEADS_TO_SHOW[0]))

    da, init_dt, target_dt = _fc_get_field(fc_id, key, init, lead)
    da = da.rename(key)
    ds_out = da.to_dataset(name=key)

    nc_bytes = ds_out.to_netcdf()
    buf = io.BytesIO(nc_bytes)
    buf.seek(0)

    tag = f"init{init_dt.strftime('%Y%m%d')}_lead{lead}_t{target_dt.strftime('%Y%m%d')}"
    return send_file(buf, as_attachment=True,
                     download_name=f"{fc_id}_{key}_{tag}.nc",
                     mimetype="application/x-netcdf")

@app.route("/fc/<fc_id>/download/csv")
def fc_dl_csv(fc_id):
    if fc_id not in fc_data:
        fc_id = list(fc_data.keys())[0]

    key = _safe_fc_key(request.args.get("key", "mhw_flag"))
    init = request.args.get("init", fc_data[fc_id]["min_init"])
    init = _nearest_fc_init_str(fc_id, init)
    lead = _safe_fc_lead(request.args.get("lead", LEADS_TO_SHOW[0]))

    da, init_dt, target_dt = _fc_get_field(fc_id, key, init, lead)
    da = da.rename(key)
    df = da.to_dataframe().reset_index()

    s = io.StringIO()
    df.to_csv(s, index=False)
    s.seek(0)

    tag = f"init{init_dt.strftime('%Y%m%d')}_lead{lead}_t{target_dt.strftime('%Y%m%d')}"
    return send_file(io.BytesIO(s.getvalue().encode()),
                     as_attachment=True,
                     download_name=f"{fc_id}_{key}_{tag}.csv",
                     mimetype="text/csv")

@app.route("/fc/<fc_id>/download/json")
def fc_dl_json(fc_id):
    if fc_id not in fc_data:
        fc_id = list(fc_data.keys())[0]

    key = _safe_fc_key(request.args.get("key", "mhw_flag"))
    init = request.args.get("init", fc_data[fc_id]["min_init"])
    init = _nearest_fc_init_str(fc_id, init)
    lead = _safe_fc_lead(request.args.get("lead", LEADS_TO_SHOW[0]))

    da, init_dt, target_dt = _fc_get_field(fc_id, key, init, lead)
    da = da.rename(key)
    df = da.to_dataframe().reset_index()

    js = df.to_json(orient="records")
    tag = f"init{init_dt.strftime('%Y%m%d')}_lead{lead}_t{target_dt.strftime('%Y%m%d')}"
    return send_file(io.BytesIO(js.encode()),
                     as_attachment=True,
                     download_name=f"{fc_id}_{key}_{tag}.json",
                     mimetype="application/json")

# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
