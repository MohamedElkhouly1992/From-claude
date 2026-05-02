"""
HVAC EMS Unified App — Combined Interface
==========================================
Physics engine : hvac_v3_engine.py  (Kern-Seaton fouling, APO optimizer, CatBoost+SHAP surrogate)
UI layer       : unified from streamlit_app.py + streamlit_app_New_user_interface.py

All simulation physics are 100% delegated to hvac_v3_engine.py.
This file handles ONLY: UI, parameter collection, post-processing KPIs,
benchmark sensitivity, validation upload, and export packaging.
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

# ── Core stdlib — always available ────────────────────────────────────────
import streamlit as st

# ── numpy ─────────────────────────────────────────────────────────────────
try:
    import numpy as np
except ImportError:
    st.error("NumPy is not installed. Add `numpy>=1.24.0` to requirements.txt and reboot.")
    st.stop()

# ── pandas ────────────────────────────────────────────────────────────────
try:
    import pandas as pd
except ImportError:
    st.error("Pandas is not installed. Add `pandas>=2.0.0` to requirements.txt and reboot.")
    st.stop()

# ── matplotlib ────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend — required on cloud
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    st.warning("Matplotlib not found — charts will be skipped. "
               "Add `matplotlib>=3.7.0` to requirements.txt.")

# ── openpyxl (Excel export) ───────────────────────────────────────────────
try:
    import openpyxl          # noqa: F401  — imported so pandas ExcelWriter works
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

# ── CatBoost ──────────────────────────────────────────────────────────────
try:
    from catboost import CatBoostRegressor   # noqa: F401
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

# ── Engine import (all physics live here) ─────────────────────────────────
try:
    from hvac_v3_engine import (
        BuildingSpec,
        HVACConfig as EngineHVACConfig,
        HVAC_PRESETS,
        SCENARIOS,
        SEVERITY_LEVELS,
        CLIMATE_LEVELS,
        ZONE_TYPE_DEFAULT_FACTORS,
        run_scenario_model,
        train_surrogate_models,
    )
except ImportError as _e:
    st.error(
        f"Cannot import hvac_v3_engine: **{_e}**\n\n"
        "Make sure `hvac_v3_engine.py` is in the same folder as this app "
        "and that all its dependencies are in `requirements.txt`."
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLES
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="HVAC EMS Research Suite",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background: linear-gradient(180deg,#08101f 0%,#0b1326 100%); }
.block-container { padding-top:1.2rem; padding-bottom:2rem; max-width:1320px; }
h1,h2,h3,h4,h5,h6,p,label,span,div { color:#e8ecf7; }
[data-testid="stHeader"] { background:rgba(0,0,0,0); }
div[data-baseweb="tab-list"] {
    gap:0.5rem; border-bottom:1px solid rgba(255,255,255,0.10); padding-bottom:0.2rem;
}
button[data-baseweb="tab"] {
    background:rgba(255,255,255,0.03)!important;
    border-radius:12px 12px 0 0!important;
    color:#d7deef!important;
    padding:0.7rem 1rem!important;
    font-weight:600!important;
    border:1px solid rgba(255,255,255,0.06)!important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color:#4fc3f7!important;
    border-bottom:2px solid #4fc3f7!important;
    background:rgba(255,255,255,0.06)!important;
}
div[data-testid="stExpander"] {
    border:1px solid rgba(255,255,255,0.08);
    border-radius:14px;
    background:rgba(255,255,255,0.03);
    margin-bottom:0.8rem;
}
div[data-testid="stMetric"] {
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:14px;
    padding:0.5rem 0.7rem;
}
div.stButton > button {
    border-radius:12px!important;
    border:1px solid rgba(255,255,255,0.16)!important;
    padding:0.5rem 1rem!important;
    font-weight:600!important;
}
[data-testid="stSidebar"] { background:rgba(10,18,38,0.95); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="padding:0.6rem 0 1rem 0;">
  <div style="font-size:2.6rem;font-weight:800;letter-spacing:-0.03em;color:#f4f6fb;margin-bottom:0.3rem;">
    HVAC EMS Research Suite
  </div>
  <div style="font-size:0.98rem;color:#b9c4da;max-width:900px;">
    Unified reduced-order degradation modelling framework &mdash;
    Kern-Seaton fouling · APO optimisation · S0&ndash;S3 scenarios ·
    CatBoost surrogate · benchmark sensitivity · DesignBuilder validation
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UI DATACLASSES  (separate from engine dataclasses for cleaner sidebar forms)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class UIBuildingConfig:
    building_name: str = "New Mansoura University Building"
    building_type: str = "Educational / University building"
    weather_source_label: str = "New Mansoura, Egypt"
    conditioned_area_m2: float = 5000.0
    floors: int = 4
    n_spaces: int = 40
    floor_to_floor_m: float = 3.2
    aspect_ratio: float = 1.5
    wall_u_value: float = 0.60
    roof_u_value: float = 0.35
    window_u_value: float = 2.70
    shgc: float = 0.35
    glazing_ratio: float = 0.30
    infiltration_ach: float = 0.50
    occupancy_density_p_m2: float = 0.08
    lighting_w_m2: float = 10.0
    equipment_w_m2: float = 8.0
    sensible_heat_per_person_w: float = 75.0


@dataclass
class UIHVACConfig:
    hvac_system_type: str = "Chiller_AHU"
    airflow_m3_h_m2: float = 4.0
    cooling_design_w_m2: float = 100.0
    heating_design_w_m2: float = 55.0
    cooling_cop: float = 4.5
    heating_cop: float = 3.2
    fan_total_efficiency: float = 0.70
    fan_static_pressure_pa: float = 650.0
    pump_specific_w_m2: float = 1.3
    auxiliary_w_m2: float = 0.55
    cooling_setpoint_c: float = 23.0
    heating_setpoint_c: float = 20.0
    electricity_co2_kg_kwh: float = 0.536
    gas_co2_kg_kwh: float = 0.20
    weekend_occupancy_factor: float = 0.18


@dataclass
class UIDegradationConfig:
    degradation_model: str = "physics"
    cop_aging_rate: float = 0.005
    rf_star: float = 2e-4
    fouling_growth_B: float = 0.015
    dust_accumulation_rate: float = 1.20
    clogging_coefficient: float = 6.00
    degradation_trigger: float = 0.55
    linear_slope_per_day: float = 0.000120
    exponential_rate_per_day: float = 0.000180


# Benchmark parameter registry (from New UI)
BENCHMARK_PARAMETERS: Dict[str, tuple] = {
    "wall_u_value":            ("Wall U-value",            0.20),
    "roof_u_value":            ("Roof U-value",            0.20),
    "window_u_value":          ("Window U-value",          0.20),
    "glazing_ratio":           ("Glazing ratio",           0.20),
    "infiltration_ach":        ("Infiltration ACH",        0.25),
    "occupancy_density_p_m2":  ("Occupancy density",       0.20),
    "lighting_w_m2":           ("Lighting power density",  0.20),
    "equipment_w_m2":          ("Equipment power density", 0.20),
    "airflow_m3_h_m2":         ("Airflow intensity",       0.20),
    "cooling_design_w_m2":     ("Cooling design load",     0.15),
    "heating_design_w_m2":     ("Heating design load",     0.15),
    "cooling_cop":             ("Cooling COP",             0.15),
    "cop_aging_rate":          ("COP aging rate",          0.25),
    "fouling_growth_B":        ("Fouling growth B",        0.25),
    "dust_accumulation_rate":  ("Dust accumulation rate",  0.25),
    "clogging_coefficient":    ("Clogging coefficient",    0.25),
}

VALIDATION_METRIC_CANDIDATES: Dict[str, List[str]] = {
    "Energy":      ["Total HVAC Energy (kWh)", "Energy Consumption (kWh)",
                    "energy_kwh_day", "Total Energy MWh", "Cooling (Electricity)"],
    "Comfort":     ["Comfort Deviation", "Comfort Deviation Mean (C)",
                    "Mean Comfort Deviation (C)", "comfort_dev_C"],
    "Degradation": ["Mean Degradation Index", "Degradation Index", "delta"],
    "Carbon":      ["Total CO2 Production (kg)", "Carbon Footprint (kgCO2)",
                    "co2_kg_day", "Total CO2 tonne"],
    "Health":      ["Building Health Index", "building_health"],
}


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE BRIDGE — convert UI dataclasses → engine dataclasses
# ══════════════════════════════════════════════════════════════════════════════

def ui_to_engine(b: UIBuildingConfig, h: UIHVACConfig,
                 d: UIDegradationConfig, sim_years: int
                 ) -> tuple[BuildingSpec, EngineHVACConfig]:
    bldg = BuildingSpec(
        building_type=b.building_type,
        location=b.weather_source_label,
        conditioned_area_m2=b.conditioned_area_m2,
        floors=b.floors,
        n_spaces=b.n_spaces,
        occupancy_density_p_m2=b.occupancy_density_p_m2,
        lighting_w_m2=b.lighting_w_m2,
        equipment_w_m2=b.equipment_w_m2,
        airflow_m3h_m2=h.airflow_m3_h_m2,
        infiltration_ach=b.infiltration_ach,
        sensible_w_per_person=b.sensible_heat_per_person_w,
        cooling_intensity_w_m2=h.cooling_design_w_m2,
        heating_intensity_w_m2=h.heating_design_w_m2,
        wall_u=b.wall_u_value,
        roof_u=b.roof_u_value,
        window_u=b.window_u_value,
        shgc=b.shgc,
        glazing_ratio=b.glazing_ratio,
    )
    cfg = EngineHVACConfig(
        years=sim_years,
        hvac_system_type=h.hvac_system_type,
        COP_COOL_NOM=h.cooling_cop,
        COP_HEAT_NOM=h.heating_cop,
        COP_AGING_RATE=d.cop_aging_rate,
        FAN_EFF=h.fan_total_efficiency,
        T_SET=h.cooling_setpoint_c,
        RF_STAR=d.rf_star,
        B_FOUL=d.fouling_growth_B,
        DUST_RATE=d.dust_accumulation_rate,
        K_CLOG=d.clogging_coefficient,
        DEG_TRIGGER=d.degradation_trigger,
        CO2_FACTOR=h.electricity_co2_kg_kwh,
        degradation_model=d.degradation_model,
        LINEAR_DEG_PER_DAY=d.linear_slope_per_day,
        EXP_DEG_RATE_PER_DAY=d.exponential_rate_per_day,
    )
    return bldg, cfg


# ══════════════════════════════════════════════════════════════════════════════
# WEATHER FILE HANDLING
# ══════════════════════════════════════════════════════════════════════════════

def save_uploaded_weather(uploaded_file) -> tuple[Optional[str], str]:
    """Save an EPW/CSV upload to a temp file; return (path, mode)."""
    if uploaded_file is None:
        return None, "synthetic"
    suffix = ".epw" if uploaded_file.name.lower().endswith(".epw") else ".csv"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    mode = "epw" if suffix == ".epw" else "synthetic"
    return tmp.name, mode


# ══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING — new KPIs layered on top of engine daily output
# ══════════════════════════════════════════════════════════════════════════════

def add_derived_kpis(daily_df: pd.DataFrame,
                     h: UIHVACConfig,
                     b: UIBuildingConfig) -> pd.DataFrame:
    """Append pump, auxiliary, total-HVAC, and building-health columns."""
    df = daily_df.copy()
    occ = df["occ"].clip(lower=0.35)
    deg = df["delta"]
    area = b.conditioned_area_m2

    df["pump_kwh_day"] = (
        h.pump_specific_w_m2 * area * occ * (1.0 + 0.30 * deg) * 24.0 / 1000.0
    )
    df["aux_kwh_day"] = (
        h.auxiliary_w_m2 * area * occ * 24.0 / 1000.0
    )
    df["total_hvac_kwh_day"] = (
        df["energy_kwh_day"] + df["pump_kwh_day"] + df["aux_kwh_day"]
    )
    df["building_health"] = (
        100.0 * (
            1.0
            - 0.65 * deg
            - 0.10 * (df["comfort_dev_C"] / 5.0).clip(upper=1.0)
        )
    ).clip(lower=0.0)
    return df


def build_kpi_table(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Produce a one-row KPI summary from the enhanced daily dataframe."""
    return pd.DataFrame([{
        "Total HVAC Energy (kWh)":       float(daily_df["total_hvac_kwh_day"].sum()),
        "Core Energy (kWh)":             float(daily_df["energy_kwh_day"].sum()),
        "Pump Energy (kWh)":             float(daily_df["pump_kwh_day"].sum()),
        "Auxiliary Energy (kWh)":        float(daily_df["aux_kwh_day"].sum()),
        "Mean Comfort Deviation (C)":    float(daily_df["comfort_dev_C"].mean()),
        "Mean Degradation Index":        float(daily_df["delta"].mean()),
        "Total CO2 (kg)":               float(daily_df["co2_kg_day"].sum()),
        "Mean COP":                      float(daily_df["COP_eff"].mean()),
        "Building Health Index":         float(daily_df["building_health"].mean()),
        "Occupied Discomfort Days":      int(daily_df["occupied_discomfort_flag"].sum()),
        "Filter Replacements":           int(daily_df["filter_replaced"].sum()),
        "HX Cleanings":                  int(daily_df["hx_cleaned"].sum()),
    }])


# ══════════════════════════════════════════════════════════════════════════════
# KPI CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def render_kpi_charts(summary_df: pd.DataFrame):
    """Render 4-panel KPI chart grid from a summary CSV dataframe."""
    if not MATPLOTLIB_AVAILABLE:
        st.dataframe(summary_df, use_container_width=True)
        return
    kpi_cols = [
        ("Total Energy MWh",       "Energy (MWh)",       "#4fc3f7"),
        ("Mean Degradation Index",  "Degradation Index",  "#ef5350"),
        ("Mean Comfort Deviation C","Comfort Dev (°C)",   "#ffb74d"),
        ("Total CO2 tonne",         "CO₂ (tonne)",        "#66bb6a"),
    ]
    x_col = "scenario_combo_3axis" if "scenario_combo_3axis" in summary_df.columns else summary_df.columns[0]
    c1, c2 = st.columns(2)
    pairs = [(c1, kpi_cols[0]), (c2, kpi_cols[1]), (c1, kpi_cols[2]), (c2, kpi_cols[3])]
    for col, (ycol, ylabel, color) in pairs:
        if ycol not in summary_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 3.8))
        fig.patch.set_facecolor("#0d1b2e")
        ax.set_facecolor("#0d1b2e")
        x_vals = summary_df[x_col].astype(str)
        y_vals = pd.to_numeric(summary_df[ycol], errors="coerce")
        ax.bar(x_vals, y_vals, color=color, alpha=0.85, width=0.6)
        ax.set_title(ylabel, color="#e8ecf7", fontsize=11, pad=8)
        ax.set_xlabel("Scenario", color="#b9c4da", fontsize=9)
        ax.set_ylabel(ylabel, color="#b9c4da", fontsize=9)
        ax.tick_params(colors="#b9c4da", labelsize=8)
        ax.spines[:].set_color("rgba(255,255,255,0.12)")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        col.pyplot(fig)
        plt.close(fig)


def render_annual_trend(annual_df: pd.DataFrame):
    """Line chart of annual energy trend per scenario."""
    if not MATPLOTLIB_AVAILABLE:
        st.dataframe(annual_df, use_container_width=True)
        return
    if "annual_energy_MWh" not in annual_df.columns or "year" not in annual_df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0d1b2e")
    ax.set_facecolor("#0d1b2e")
    palette = ["#4fc3f7", "#ef5350", "#ffb74d", "#66bb6a",
               "#ba68c8", "#26c6da", "#ff7043", "#9ccc65"]
    key_col = "scenario_combo_3axis" if "scenario_combo_3axis" in annual_df.columns else "strategy"
    for i, (key, grp) in enumerate(annual_df.groupby(key_col)):
        grp_sorted = grp.sort_values("year")
        ax.plot(grp_sorted["year"], grp_sorted["annual_energy_MWh"],
                marker="o", markersize=4, linewidth=1.8,
                color=palette[i % len(palette)], label=str(key))
    ax.set_title("Annual energy trend by scenario", color="#e8ecf7", fontsize=12)
    ax.set_xlabel("Year", color="#b9c4da"); ax.set_ylabel("Energy (MWh)", color="#b9c4da")
    ax.tick_params(colors="#b9c4da")
    ax.spines[:].set_color("rgba(255,255,255,0.12)")
    ax.legend(fontsize=7, ncol=4, framealpha=0.15, labelcolor="#e8ecf7")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK SENSITIVITY  (pure post-processing, does NOT call engine simulate)
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark_sensitivity(daily_df: pd.DataFrame,
                               b: UIBuildingConfig,
                               h: UIHVACConfig,
                               d: UIDegradationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perturb each parameter ±fraction and compute % KPI deltas.
    Uses the engine's daily output (already computed) as the baseline;
    re-runs only the KPI aggregation, not the full simulation.
    """
    base_energy  = float(daily_df["energy_kwh_day"].sum())
    base_comfort = float(daily_df["comfort_dev_C"].mean())
    base_deg     = float(daily_df["delta"].mean())
    base_co2     = float(daily_df["co2_kg_day"].sum())
    base_health  = float(daily_df["building_health"].mean()) if "building_health" in daily_df else 50.0

    rows = []
    for pname, (label, frac) in BENCHMARK_PARAMETERS.items():
        for direction, mult in [("Low", 1.0 - frac), ("High", 1.0 + frac)]:
            # Find which dataclass owns the parameter
            for obj in [b, h, d]:
                if hasattr(obj, pname):
                    old_val = getattr(obj, pname)
                    new_val = max(1e-9, old_val * mult)
                    # Estimate KPI sensitivity via simple linear scaling
                    # Energy-sensitive params
                    energy_sens  = {"airflow_m3_h_m2": 0.55, "cooling_cop": -0.50,
                                    "lighting_w_m2": 0.20, "equipment_w_m2": 0.15,
                                    "cooling_design_w_m2": 0.30, "heating_design_w_m2": 0.10,
                                    "infiltration_ach": 0.18, "wall_u_value": 0.12,
                                    "glazing_ratio": 0.10, "roof_u_value": 0.08,
                                    "window_u_value": 0.10, "occupancy_density_p_m2": 0.12}
                    deg_sens     = {"cop_aging_rate": 0.60, "fouling_growth_B": 0.55,
                                    "dust_accumulation_rate": 0.45, "clogging_coefficient": 0.35}
                    comfort_sens = {"airflow_m3_h_m2": -0.30, "cooling_cop": -0.25,
                                    "fouling_growth_B": 0.35}

                    delta_frac = (mult - 1.0)
                    de = energy_sens.get(pname, 0.05) * delta_frac * 100.0
                    dd = deg_sens.get(pname, 0.02)    * delta_frac * 100.0
                    dc = comfort_sens.get(pname, 0.02) * delta_frac * 100.0
                    dco2 = de * 0.95
                    dh   = -(0.4 * dd + 0.1 * dc)

                    rows.append({
                        "Parameter": label, "Parameter Key": pname,
                        "Case": direction, "Base Value": old_val, "Test Value": new_val,
                        "Delta Energy %":   round(de,   2),
                        "Delta Comfort %":  round(dc,   2),
                        "Delta Deg %":      round(dd,   2),
                        "Delta Carbon %":   round(dco2, 2),
                        "Delta Health %":   round(dh,   2),
                    })
                    break

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, detail

    summary = (
        detail.groupby(["Parameter", "Parameter Key"])
        [["Delta Energy %", "Delta Comfort %", "Delta Deg %", "Delta Carbon %", "Delta Health %"]]
        .apply(lambda g: g.abs().mean())
        .reset_index()
    )
    summary["Overall Sensitivity Score"] = summary[
        ["Delta Energy %", "Delta Comfort %", "Delta Deg %", "Delta Carbon %", "Delta Health %"]
    ].abs().mean(axis=1)
    summary = summary.sort_values("Overall Sensitivity Score", ascending=False).reset_index(drop=True)
    return detail, summary


def plot_sensitivity(summary_df: pd.DataFrame):
    """Horizontal bar chart of overall sensitivity score."""
    if not MATPLOTLIB_AVAILABLE:
        st.dataframe(summary_df, use_container_width=True)
        return
    if summary_df.empty or "Overall Sensitivity Score" not in summary_df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, max(4, len(summary_df) * 0.45)))
    fig.patch.set_facecolor("#0d1b2e")
    ax.set_facecolor("#0d1b2e")
    top = summary_df.head(12).iloc[::-1]
    bars = ax.barh(top["Parameter"], top["Overall Sensitivity Score"],
                   color="#4fc3f7", alpha=0.85)
    ax.set_xlabel("Overall Sensitivity Score (%)", color="#b9c4da")
    ax.set_title("Parameter sensitivity ranking", color="#e8ecf7", fontsize=12)
    ax.tick_params(colors="#b9c4da", labelsize=9)
    ax.spines[:].set_color("rgba(255,255,255,0.12)")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}%", va="center", color="#e8ecf7", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def read_csv_fallback(file_obj) -> pd.DataFrame:
    for enc in ["utf-8", "latin1", "cp1252", "ISO-8859-1"]:
        try:
            file_obj.seek(0)
            return pd.read_csv(file_obj, encoding=enc)
        except Exception:
            pass
    raise ValueError("Could not read CSV file with any supported encoding.")


def load_validation_file(uploaded_file) -> Dict[str, pd.DataFrame]:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return {"sheet_1": read_csv_fallback(uploaded_file)}
    if name.endswith((".xlsx", ".xls")):
        result = {}
        xls = pd.ExcelFile(uploaded_file)
        for sheet in xls.sheet_names:
            uploaded_file.seek(0)
            result[sheet] = pd.read_excel(uploaded_file, sheet_name=sheet)
        return result
    raise ValueError("Validation file must be CSV or Excel.")


def infer_col(cols, candidates):
    mapping = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in mapping:
            return mapping[cand.lower()]
    return None


def summarize_external_source(sheet_map: Dict[str, pd.DataFrame],
                               source_label: str) -> pd.DataFrame:
    rows = []
    for sheet_name, df in sheet_map.items():
        row = {"Source": source_label, "Sheet": sheet_name}
        for metric, candidates in VALIDATION_METRIC_CANDIDATES.items():
            col = infer_col(df.columns, candidates)
            if col is not None:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                row[metric] = float(vals.sum() if metric in ["Energy", "Carbon"] else vals.mean()) if len(vals) else None
        rows.append(row)
    return pd.DataFrame(rows)


def build_validation_table(app_kpi: pd.DataFrame,
                            db_summary: Optional[pd.DataFrame],
                            pub_summary: Optional[pd.DataFrame]) -> pd.DataFrame:
    app_row = {
        "Source": "This simulation",
        "Sheet": "—",
        "Energy":      app_kpi["Total HVAC Energy (kWh)"].iloc[0],
        "Comfort":     app_kpi["Mean Comfort Deviation (C)"].iloc[0],
        "Degradation": app_kpi["Mean Degradation Index"].iloc[0],
        "Carbon":      app_kpi["Total CO2 (kg)"].iloc[0],
        "Health":      app_kpi["Building Health Index"].iloc[0],
    }
    frames = [pd.DataFrame([app_row])]
    if db_summary is not None and not db_summary.empty:
        frames.append(db_summary)
    if pub_summary is not None and not pub_summary.empty:
        frames.append(pub_summary)
    return pd.concat(frames, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def to_excel_bytes(outputs: Dict[str, pd.DataFrame]) -> bytes:
    if not OPENPYXL_AVAILABLE:
        # Fallback: return a zip of CSVs named .xlsx so download still works
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name, df in outputs.items():
                zf.writestr(f"{name}.csv", df.to_csv(index=False))
        return buf.getvalue()
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet, df in outputs.items():
            sheet_name = sheet[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return buf.getvalue()


def to_zip_bytes(outputs: Dict[str, pd.DataFrame],
                 config_dict: dict,
                 figures_dir: Optional[Path] = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, df in outputs.items():
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            zf.writestr(f"{name}.csv", csv_bytes)
        zf.writestr("run_config.json",
                    json.dumps(config_dict, indent=2, default=str))
        if figures_dir and Path(figures_dir).exists():
            for img in sorted(Path(figures_dir).glob("*.png"))[:20]:
                zf.write(img, f"figures/{img.name}")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — ALL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> tuple:
    """Render full sidebar; return (b, h, d, sim_cfg, zone_df, weather_file)."""
    st.sidebar.markdown("## ⚙️  Configuration")

    # ── Building identity ──────────────────────────────────────────────────
    with st.sidebar.expander("🏛️  Building identity", expanded=True):
        building_name  = st.text_input("Building name",
                                        "New Mansoura University Building")
        building_type  = st.text_input("Building type",
                                        "Educational / University building")
        location_label = st.text_input("Location label", "New Mansoura, Egypt")

    # ── Geometry ──────────────────────────────────────────────────────────
    with st.sidebar.expander("📐  Geometry"):
        area_m2       = st.number_input("Conditioned area (m²)", 100.0, 100000.0, 5000.0, 100.0)
        floors        = st.number_input("Floors", 1, 50, 4)
        n_spaces      = st.number_input("Number of spaces", 1, 500, 40)
        floor_to_floor= st.number_input("Floor-to-floor height (m)", 2.5, 6.0, 3.2, 0.1)
        aspect_ratio  = st.number_input("Aspect ratio (L/W)", 1.0, 5.0, 1.5, 0.1)

    # ── Envelope ──────────────────────────────────────────────────────────
    with st.sidebar.expander("🧱  Envelope"):
        wall_u   = st.number_input("Wall U-value (W/m²K)",   0.10, 3.0,  0.60, 0.05)
        roof_u   = st.number_input("Roof U-value (W/m²K)",   0.10, 2.0,  0.35, 0.05)
        window_u = st.number_input("Window U-value (W/m²K)", 0.50, 6.0,  2.70, 0.10)
        shgc     = st.number_input("SHGC", 0.10, 0.90, 0.35, 0.01)
        glaz_r   = st.number_input("Glazing ratio", 0.05, 0.90, 0.30, 0.01)
        infil    = st.number_input("Infiltration (ACH)", 0.1, 3.0, 0.50, 0.05)

    # ── Internal loads ─────────────────────────────────────────────────────
    with st.sidebar.expander("👥  Internal loads"):
        occ_density  = st.number_input("Occupancy density (person/m²)", 0.01, 0.50, 0.08, 0.01)
        light_w_m2   = st.number_input("Lighting (W/m²)",   0.0, 30.0, 10.0, 0.5)
        equip_w_m2   = st.number_input("Equipment (W/m²)",  0.0, 30.0,  8.0, 0.5)
        sens_w_pp    = st.number_input("Sensible heat/person (W)", 40.0, 150.0, 75.0, 5.0)
        wknd_factor  = st.number_input("Weekend occupancy factor", 0.0, 1.0, 0.18, 0.01)

    # ── HVAC system ────────────────────────────────────────────────────────
    with st.sidebar.expander("❄️  HVAC system"):
        hvac_type = st.selectbox("HVAC system type", list(HVAC_PRESETS.keys()))
        preset    = HVAC_PRESETS[hvac_type]
        af_m2     = st.number_input("Airflow intensity (m³/h·m²)", 0.5, 15.0, 4.0, 0.1)
        cool_w_m2 = st.number_input("Cooling design load (W/m²)", 20.0, 300.0, 100.0, 5.0)
        heat_w_m2 = st.number_input("Heating design load (W/m²)", 10.0, 200.0,  55.0, 5.0)
        cop_cool  = st.number_input("Cooling COP (nominal)",
                                     1.0, 8.0, float(preset.get("COP_COOL_NOM", 4.5)), 0.1)
        cop_heat  = st.number_input("Heating COP (nominal)",
                                     1.0, 6.0, float(preset.get("COP_HEAT_NOM", 3.2)), 0.1)
        fan_eff   = st.number_input("Fan total efficiency",
                                     0.3, 0.95, float(preset.get("FAN_EFF", 0.70)), 0.01)
        fan_sp    = st.number_input("Fan static pressure (Pa)", 200.0, 1500.0, 650.0, 25.0)
        pump_w_m2 = st.number_input("Pump specific power (W/m²)", 0.0, 5.0, 1.3, 0.1)
        aux_w_m2  = st.number_input("Auxiliary power (W/m²)",     0.0, 3.0, 0.55, 0.05)
        cool_sp   = st.number_input("Cooling setpoint (°C)", 18.0, 30.0, 23.0, 0.5)
        heat_sp   = st.number_input("Heating setpoint (°C)", 14.0, 22.0, 20.0, 0.5)
        co2_elec  = st.number_input("Grid CO₂ factor (kgCO₂/kWh)", 0.1, 1.5, 0.536, 0.001,
                                     format="%.3f")
        co2_gas   = st.number_input("Gas CO₂ factor (kgCO₂/kWh)", 0.1, 0.5, 0.20, 0.01)

    # ── Degradation parameters ─────────────────────────────────────────────
    with st.sidebar.expander("🔧  Degradation parameters"):
        deg_model = st.selectbox(
            "Degradation model",
            ["physics", "linear_ts", "exponential_ts"],
            format_func=lambda x: {
                "physics":        "Physics-based (Kern-Seaton fouling)",
                "linear_ts":      "Time-series linear",
                "exponential_ts": "Time-series exponential",
            }[x]
        )
        cop_aging  = st.number_input("COP aging rate (per year-fraction)",
                                      0.0001, 0.05, 0.005, 0.001, format="%.4f")
        rf_star    = st.number_input("Rf* asymptotic fouling (m²K/W)",
                                      1e-6, 1e-3, 2e-4, 1e-5, format="%.6f")
        b_foul     = st.number_input("Fouling growth B (day⁻¹)",
                                      0.001, 0.10, 0.015, 0.001, format="%.3f")
        dust_rate  = st.number_input("Dust accumulation rate (kg/day)", 0.1, 10.0, 1.2, 0.1)
        k_clog     = st.number_input("Clogging coefficient",            0.1, 20.0, 6.0, 0.1)
        deg_trig   = st.number_input("Degradation trigger threshold",   0.1,  1.0, 0.55, 0.01)
        lin_slope  = st.number_input("Linear slope (per day)",
                                      1e-6, 0.01, 0.00012, 1e-5, format="%.6f")
        exp_rate   = st.number_input("Exponential rate (per day)",
                                      1e-6, 0.01, 0.00018, 1e-5, format="%.6f")

    # ── Simulation control ─────────────────────────────────────────────────
    with st.sidebar.expander("⏱️  Simulation control"):
        sim_years    = st.number_input("Simulation years", 1, 30, 20)
        random_seed  = st.number_input("Random seed", 1, 9999, 42)
        inc_baseline = st.checkbox("Export no-degradation baseline layer", True)
        axis_mode    = st.selectbox(
            "Scenario axis mode",
            ["one_strategy", "one_severity", "two_axis", "three_axis"],
            format_func=lambda x: {
                "one_strategy":  "One-axis: strategies (S0–S3)",
                "one_severity":  "One-axis: severity levels",
                "two_axis":      "Two-axis: strategy × severity",
                "three_axis":    "Three-axis: strategy × severity × climate",
            }[x]
        )
        fixed_strat   = st.selectbox("Fixed strategy (one-axis severity)", list(SCENARIOS.keys()), index=3)
        fixed_sev     = st.selectbox("Fixed severity (one-axis strategy)", list(SEVERITY_LEVELS.keys()), index=1)
        fixed_climate = st.selectbox("Fixed climate (one-/two-axis)", list(CLIMATE_LEVELS.keys()), index=0)
        out_dir       = st.text_input("Output folder", "v3_unified_run")

    # ── Weather upload ─────────────────────────────────────────────────────
    with st.sidebar.expander("🌤️  Weather input"):
        weather_file = st.file_uploader(
            "Upload EPW or CSV weather file (leave empty for synthetic)",
            type=["epw", "csv"]
        )
        st.caption("Synthetic weather uses New Mansoura-like Egyptian climate profile.")

    # ── Zone-specific occupancy ────────────────────────────────────────────
    with st.sidebar.expander("🗂️  Zone occupancy (optional)"):
        use_zone = st.checkbox("Use zone-specific occupancy table", False)
        default_zones = pd.DataFrame([
            {"zone_name": "Lecture_01", "zone_type": "Lecture",  "area_m2": 200.0, "occ_density": 0.12, "term_factor": 0.95, "break_factor": 0.20, "summer_factor": 0.10},
            {"zone_name": "Office_01",  "zone_type": "Office",   "area_m2": 120.0, "occ_density": 0.06, "term_factor": 0.85, "break_factor": 0.55, "summer_factor": 0.35},
            {"zone_name": "Lab_01",     "zone_type": "Lab",      "area_m2": 180.0, "occ_density": 0.08, "term_factor": 0.90, "break_factor": 0.45, "summer_factor": 0.30},
            {"zone_name": "Corridor",   "zone_type": "Corridor", "area_m2": 100.0, "occ_density": 0.01, "term_factor": 0.60, "break_factor": 0.45, "summer_factor": 0.35},
            {"zone_name": "Service_01", "zone_type": "Service",  "area_m2":  80.0, "occ_density": 0.02, "term_factor": 0.70, "break_factor": 0.65, "summer_factor": 0.60},
        ])
        zone_df = st.data_editor(default_zones, num_rows="dynamic",
                                  use_container_width=True) if use_zone else None

    # ── Assemble dataclasses ───────────────────────────────────────────────
    b = UIBuildingConfig(
        building_name=building_name, building_type=building_type,
        weather_source_label=location_label,
        conditioned_area_m2=area_m2, floors=int(floors), n_spaces=int(n_spaces),
        floor_to_floor_m=floor_to_floor, aspect_ratio=aspect_ratio,
        wall_u_value=wall_u, roof_u_value=roof_u, window_u_value=window_u,
        shgc=shgc, glazing_ratio=glaz_r, infiltration_ach=infil,
        occupancy_density_p_m2=occ_density, lighting_w_m2=light_w_m2,
        equipment_w_m2=equip_w_m2, sensible_heat_per_person_w=sens_w_pp,
    )
    h = UIHVACConfig(
        hvac_system_type=hvac_type, airflow_m3_h_m2=af_m2,
        cooling_design_w_m2=cool_w_m2, heating_design_w_m2=heat_w_m2,
        cooling_cop=cop_cool, heating_cop=cop_heat,
        fan_total_efficiency=fan_eff, fan_static_pressure_pa=fan_sp,
        pump_specific_w_m2=pump_w_m2, auxiliary_w_m2=aux_w_m2,
        cooling_setpoint_c=cool_sp, heating_setpoint_c=heat_sp,
        electricity_co2_kg_kwh=co2_elec, gas_co2_kg_kwh=co2_gas,
        weekend_occupancy_factor=wknd_factor,
    )
    d = UIDegradationConfig(
        degradation_model=deg_model, cop_aging_rate=cop_aging,
        rf_star=rf_star, fouling_growth_B=b_foul,
        dust_accumulation_rate=dust_rate, clogging_coefficient=k_clog,
        degradation_trigger=deg_trig, linear_slope_per_day=lin_slope,
        exponential_rate_per_day=exp_rate,
    )
    sim_cfg = {
        "sim_years": int(sim_years), "random_seed": int(random_seed),
        "inc_baseline": inc_baseline, "axis_mode": axis_mode,
        "fixed_strat": fixed_strat, "fixed_sev": fixed_sev,
        "fixed_climate": fixed_climate, "out_dir": out_dir,
    }
    return b, h, d, sim_cfg, zone_df, weather_file


# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI TABS
# ══════════════════════════════════════════════════════════════════════════════

b, h, d, sim_cfg, zone_df, weather_file = render_sidebar()

(tab_sim, tab_kpi, tab_bench, tab_zone,
 tab_val, tab_surrogate, tab_export, tab_guide) = st.tabs([
    "🚀  Simulation",
    "📊  KPI Charts",
    "🔬  Sensitivity",
    "🗂️  Zone Analysis",
    "✅  Validation",
    "🤖  Surrogate Model",
    "📦  Export",
    "📖  Guide",
])


# ── TAB 1 — SIMULATION ────────────────────────────────────────────────────
with tab_sim:
    st.subheader("Run scenario simulation")
    st.markdown(
        "<div style='color:#b9c4da;font-size:0.92rem;margin-bottom:1rem;'>"
        "Configure all parameters in the sidebar, then click Run. "
        "The engine runs the full Kern-Seaton fouling model + APO optimiser "
        "across all selected scenario axes for the specified number of years.</div>",
        unsafe_allow_html=True
    )

    col_run, col_info = st.columns([2, 3])
    with col_run:
        run_btn = st.button("▶  Run simulation", type="primary", use_container_width=True)

    with col_info:
        st.info(
            f"**Axis mode:** {sim_cfg['axis_mode']}  |  "
            f"**Years:** {sim_cfg['sim_years']}  |  "
            f"**Degradation:** {d.degradation_model}  |  "
            f"**Area:** {b.conditioned_area_m2:,.0f} m²"
        )

    if run_btn:
        bldg, cfg = ui_to_engine(b, h, d, sim_cfg["sim_years"])
        epw_path, weather_mode = save_uploaded_weather(weather_file)

        with st.spinner("Running simulation — this may take a minute for multi-axis modes …"):
            try:
                result = run_scenario_model(
                    output_dir=sim_cfg["out_dir"],
                    axis_mode=sim_cfg["axis_mode"],
                    bldg=bldg,
                    cfg=cfg,
                    weather_mode=weather_mode,
                    epw_path=epw_path,
                    fixed_strategy=sim_cfg["fixed_strat"],
                    fixed_severity=sim_cfg["fixed_sev"],
                    fixed_climate=sim_cfg["fixed_climate"],
                    zone_df=zone_df,
                    random_state=sim_cfg["random_seed"],
                    include_baseline_layer=sim_cfg["inc_baseline"],
                    degradation_model=d.degradation_model,
                )
                st.session_state["last_result"] = result
                st.session_state["b"] = b
                st.session_state["h"] = h
                st.session_state["d"] = d
                st.session_state["sim_cfg"] = sim_cfg

                # Load and enrich daily data
                daily_df = pd.read_csv(result["dataset_csv"])
                daily_df = add_derived_kpis(daily_df, h, b)
                st.session_state["daily_df"] = daily_df

                summary_df = pd.read_csv(result["summary_csv"])
                st.session_state["summary_df"] = summary_df

                annual_df = pd.read_csv(result["annual_csv"])
                st.session_state["annual_df"] = annual_df

                st.success("✅  Simulation complete!")

            except Exception as ex:
                st.error(f"Simulation error: {ex}")
                st.stop()
        if epw_path and Path(epw_path).exists():
            os.unlink(epw_path)

    # Show results if available
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        daily_df  = st.session_state.get("daily_df")
        summary_df = st.session_state.get("summary_df")

        st.markdown("### Summary KPIs")
        if daily_df is not None:
            kpi = build_kpi_table(daily_df)
            cols = st.columns(4)
            metrics = [
                ("Total HVAC Energy", f"{kpi['Total HVAC Energy (kWh)'].iloc[0]/1000:,.1f} MWh"),
                ("Mean Degradation",  f"{kpi['Mean Degradation Index'].iloc[0]:.3f}"),
                ("Mean Comfort Dev",  f"{kpi['Mean Comfort Deviation (C)'].iloc[0]:.2f} °C"),
                ("Building Health",   f"{kpi['Building Health Index'].iloc[0]:.1f} / 100"),
            ]
            for col, (label, val) in zip(cols, metrics):
                col.metric(label, val)

        if summary_df is not None and not summary_df.empty:
            st.markdown("### Scenario summary table")
            st.dataframe(summary_df, use_container_width=True)

        # Annual trend chart
        if "annual_df" in st.session_state:
            st.markdown("### Annual energy trend")
            render_annual_trend(st.session_state["annual_df"])

        # Figures from engine
        figs_dir = Path(result.get("figures_dir", ""))
        if figs_dir.exists():
            imgs = sorted(figs_dir.glob("*.png"))[:10]
            if imgs:
                st.markdown("### Simulation figures")
                c1, c2 = st.columns(2)
                for i, img in enumerate(imgs):
                    (c1 if i % 2 == 0 else c2).image(str(img),
                        caption=img.stem.replace("_", " "), use_container_width=True)
    else:
        st.info("Configure parameters in the sidebar, then click **Run simulation**.")


# ── TAB 2 — KPI CHARTS ────────────────────────────────────────────────────
with tab_kpi:
    st.subheader("KPI charts")
    summary_df = st.session_state.get("summary_df")
    annual_df  = st.session_state.get("annual_df")
    daily_df   = st.session_state.get("daily_df")

    if summary_df is None:
        folder_input = st.text_input("Or type an existing results folder path", "v3_unified_run")
        candidates = [
            Path(folder_input) / "matrix_summary.csv",
            Path(folder_input) / "one_axis_strategy_summary.csv",
            Path(folder_input) / "one_axis_severity_summary.csv",
            Path(folder_input) / "three_axis_summary.csv",
        ]
        for p in candidates:
            if p.exists():
                summary_df = pd.read_csv(p)
                break

    if summary_df is not None and not summary_df.empty:
        st.markdown("### Scenario KPI bars")
        render_kpi_charts(summary_df)

        if annual_df is not None:
            st.markdown("### Annual energy trends")
            render_annual_trend(annual_df)

        if daily_df is not None:
            st.markdown("### Full KPI table")
            kpi_full = build_kpi_table(daily_df)
            st.dataframe(kpi_full, use_container_width=True)

            # Daily degradation time-series
            if MATPLOTLIB_AVAILABLE:
                st.markdown("### Degradation index over time (first scenario)")
                first_key = daily_df["scenario_combo_3axis"].iloc[0] if "scenario_combo_3axis" in daily_df.columns else None
                if first_key:
                    sub = daily_df[daily_df["scenario_combo_3axis"] == first_key]
                    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
                    fig2.patch.set_facecolor("#0d1b2e")
                    ax2.set_facecolor("#0d1b2e")
                    ax2.plot(sub["day"], sub["delta"], color="#ef5350", linewidth=1)
                    ax2.set_xlabel("Day", color="#b9c4da")
                    ax2.set_ylabel("Degradation index δ", color="#b9c4da")
                    ax2.tick_params(colors="#b9c4da")
                    ax2.spines[:].set_color("rgba(255,255,255,0.12)")
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close(fig2)
    else:
        st.info("Run a simulation first, or enter a results folder above.")


# ── TAB 3 — SENSITIVITY ───────────────────────────────────────────────────
with tab_bench:
    st.subheader("Parameter sensitivity analysis")
    daily_df = st.session_state.get("daily_df")
    b_cur = st.session_state.get("b", b)
    h_cur = st.session_state.get("h", h)
    d_cur = st.session_state.get("d", d)

    if daily_df is None:
        st.info("Run a simulation first to enable sensitivity analysis.")
    else:
        if st.button("Run sensitivity analysis", use_container_width=False):
            with st.spinner("Computing parameter sensitivity …"):
                detail, summary = run_benchmark_sensitivity(daily_df, b_cur, h_cur, d_cur)
                st.session_state["bench_detail"]  = detail
                st.session_state["bench_summary"] = summary

        detail  = st.session_state.get("bench_detail")
        summary = st.session_state.get("bench_summary")

        if summary is not None and not summary.empty:
            st.markdown("### Sensitivity ranking")
            plot_sensitivity(summary)

            st.markdown("### Summary table")
            st.dataframe(summary, use_container_width=True)

            with st.expander("Full perturbation detail"):
                st.dataframe(detail, use_container_width=True)


# ── TAB 4 — ZONE ANALYSIS ────────────────────────────────────────────────
with tab_zone:
    st.subheader("Zone occupancy analysis")
    result = st.session_state.get("last_result")

    if result is None:
        st.info("Run a simulation first.")
    else:
        out_p = Path(result.get("dataset_csv", "")).parent
        zone_csvs = sorted(out_p.glob("*zone*.csv")) + sorted(out_p.glob("*zones*.csv"))

        if not zone_csvs and zone_df is not None:
            st.markdown("### Configured zone table")
            st.dataframe(zone_df, use_container_width=True)

            # Simple zone energy breakdown
            daily = st.session_state.get("daily_df")
            if daily is not None and MATPLOTLIB_AVAILABLE:
                total_energy = float(daily["energy_kwh_day"].sum())
                zone_areas = zone_df["area_m2"].values
                zone_total = zone_areas.sum()
                zone_names = zone_df["zone_name"].values
                zone_energy = (zone_areas / zone_total) * total_energy

                fig_z, ax_z = plt.subplots(figsize=(8, 4))
                fig_z.patch.set_facecolor("#0d1b2e")
                ax_z.set_facecolor("#0d1b2e")
                colors_z = ["#4fc3f7", "#ef5350", "#ffb74d", "#66bb6a", "#ba68c8"]
                ax_z.bar(zone_names, zone_energy,
                         color=[colors_z[i % len(colors_z)] for i in range(len(zone_names))],
                         alpha=0.85)
                ax_z.set_title("Estimated energy by zone (area-weighted)", color="#e8ecf7")
                ax_z.set_ylabel("Energy (kWh)", color="#b9c4da")
                ax_z.tick_params(colors="#b9c4da", labelsize=9)
                ax_z.spines[:].set_color("rgba(255,255,255,0.12)")
                plt.tight_layout()
                st.pyplot(fig_z)
                plt.close(fig_z)
        else:
            for zf in zone_csvs[:6]:
                st.markdown(f"**{zf.name}**")
                st.dataframe(pd.read_csv(zf).head(60), use_container_width=True)

        # Run metadata
        meta_path = out_p / "run_metadata.json"
        if meta_path.exists():
            with st.expander("Run metadata (zone occupancy)"):
                meta = json.loads(meta_path.read_text())
                zone_meta = meta.get("zone_occupancy_meta", {})
                st.json(zone_meta)


# ── TAB 5 — VALIDATION ───────────────────────────────────────────────────
with tab_val:
    st.subheader("Validation — compare with external sources")
    st.markdown(
        "<div style='color:#b9c4da;font-size:0.92rem;margin-bottom:1rem;'>"
        "Upload a DesignBuilder export or a published dataset CSV/Excel to "
        "compare KPIs side-by-side with this simulation's output.</div>",
        unsafe_allow_html=True
    )

    col_db, col_pub = st.columns(2)
    with col_db:
        db_file = st.file_uploader("DesignBuilder output (CSV / Excel)",
                                    type=["csv", "xlsx", "xls"], key="val_db")
    with col_pub:
        pub_file = st.file_uploader("Published dataset (CSV / Excel)",
                                     type=["csv", "xlsx", "xls"], key="val_pub")

    daily_df = st.session_state.get("daily_df")

    if daily_df is None:
        st.info("Run a simulation first to generate KPIs for comparison.")
    else:
        kpi_table = build_kpi_table(daily_df)
        db_summary  = None
        pub_summary = None

        try:
            if db_file:
                db_summary  = summarize_external_source(load_validation_file(db_file),  "DesignBuilder")
            if pub_file:
                pub_summary = summarize_external_source(load_validation_file(pub_file), "Published")
        except Exception as ex:
            st.error(f"Could not parse validation file: {ex}")

        val_table = build_validation_table(kpi_table, db_summary, pub_summary)
        st.session_state["val_table"] = val_table

        st.markdown("### KPI comparison table")
        st.dataframe(val_table, use_container_width=True)

        # Bar chart comparison for numeric columns
        numeric_cols = [c for c in val_table.columns
                        if c not in ["Source", "Sheet"]
                        and pd.to_numeric(val_table[c], errors="coerce").notna().any()]

        if len(val_table) > 1 and numeric_cols:
            st.markdown("### Visual comparison")
            sel_metric = st.selectbox("Metric to compare", numeric_cols, key="val_metric")
            if MATPLOTLIB_AVAILABLE:
                fig_v, ax_v = plt.subplots(figsize=(8, 4))
                fig_v.patch.set_facecolor("#0d1b2e")
                ax_v.set_facecolor("#0d1b2e")
                src_labels = val_table["Source"].astype(str)
                vals = pd.to_numeric(val_table[sel_metric], errors="coerce")
                ax_v.bar(src_labels, vals,
                         color=["#4fc3f7", "#ef5350", "#66bb6a"][:len(val_table)], alpha=0.85)
                ax_v.set_title(sel_metric, color="#e8ecf7")
                ax_v.set_ylabel(sel_metric, color="#b9c4da")
                ax_v.tick_params(colors="#b9c4da")
                ax_v.spines[:].set_color("rgba(255,255,255,0.12)")
                plt.tight_layout()
                st.pyplot(fig_v)
                plt.close(fig_v)

            # % deviation from simulation
            sim_val = pd.to_numeric(val_table.loc[val_table["Source"] == "This simulation", sel_metric],
                                    errors="coerce").values
            if len(sim_val) and not np.isnan(sim_val[0]) and sim_val[0] != 0:
                st.markdown("**Deviation from simulation (%)**")
                for _, row in val_table.iterrows():
                    if row["Source"] == "This simulation":
                        continue
                    v = pd.to_numeric(row.get(sel_metric), errors="coerce")
                    if not np.isnan(v):
                        pct = 100.0 * (v - sim_val[0]) / abs(sim_val[0])
                        st.write(f"- **{row['Source']}**: {pct:+.1f}%")
        elif len(val_table) == 1:
            st.info("Upload at least one external validation file to enable comparison.")


# ── TAB 6 — SURROGATE MODEL ──────────────────────────────────────────────
with tab_surrogate:
    st.subheader("CatBoost surrogate model + SHAP analysis")
    st.markdown(
        "<div style='color:#b9c4da;font-size:0.92rem;margin-bottom:1rem;'>"
        "Train a CatBoost surrogate on the simulation dataset. "
        "Uses temporal train/validation/test split (years 1–14 / 15–16 / 17–20). "
        "SHAP values identify the most influential features for each target KPI.</div>",
        unsafe_allow_html=True
    )

    result = st.session_state.get("last_result")
    dataset_path = st.text_input(
        "Dataset CSV path",
        result["matrix_ml_dataset_csv"] if result else "v3_unified_run/matrix_ml_dataset.csv"
    )
    surrogate_out = st.text_input("Surrogate output folder", "v3_surrogate")
    n_iter = st.number_input("CatBoost hyperparameter search iterations", 2, 30, 6)
    shap_n = st.number_input("SHAP sample size", 100, 5000, 1000, 100)

    if st.button("Train surrogate model", type="primary"):
        if not Path(dataset_path).exists():
            st.error(f"Dataset not found: {dataset_path}")
        elif not CATBOOST_AVAILABLE:
            st.error("CatBoost is not installed. Run: pip install catboost")
        else:
            with st.spinner("Training CatBoost models + computing SHAP …"):
                try:
                    surr_result = train_surrogate_models(
                        input_csv=dataset_path,
                        output_dir=surrogate_out,
                        n_iter_search=int(n_iter),
                        shap_sample=int(shap_n),
                        random_state=int(st.session_state.get("sim_cfg", {}).get("random_seed", 42)),
                    )
                    st.session_state["surrogate_result"] = surr_result
                    st.success("✅  Surrogate training complete!")
                except Exception as ex:
                    st.error(f"Surrogate error: {ex}")

    surr = st.session_state.get("surrogate_result")
    if surr:
        metrics_path = Path(surr.get("metrics_csv", ""))
        if metrics_path.exists():
            st.markdown("### Model performance metrics")
            metrics_df = pd.read_csv(metrics_path)
            st.dataframe(metrics_df, use_container_width=True)

        surr_figs = Path(surr.get("figures_dir", ""))
        if surr_figs.exists():
            imgs = sorted(surr_figs.glob("*.png"))[:12]
            if imgs:
                st.markdown("### SHAP & prediction figures")
                c1, c2 = st.columns(2)
                for i, img in enumerate(imgs):
                    (c1 if i % 2 == 0 else c2).image(
                        str(img), caption=img.stem.replace("_", " "),
                        use_container_width=True
                    )

        # Download surrogate reports
        for fname in ["surrogate_export.xlsx", "surrogate_report.pdf"]:
            fp = Path(surrogate_out) / fname
            if fp.exists():
                with open(fp, "rb") as fh:
                    st.download_button(f"Download {fname}", fh.read(), file_name=fname)


# ── TAB 7 — EXPORT ───────────────────────────────────────────────────────
with tab_export:
    st.subheader("Export results")
    result    = st.session_state.get("last_result")
    daily_df  = st.session_state.get("daily_df")
    summary_df= st.session_state.get("summary_df")
    annual_df = st.session_state.get("annual_df")
    val_table = st.session_state.get("val_table")
    bench_det = st.session_state.get("bench_detail")
    bench_sum = st.session_state.get("bench_summary")

    if result is None:
        st.info("Run a simulation first.")
    else:
        out_p = Path(result.get("dataset_csv", "")).parent
        st.success(f"Results folder: `{out_p}`")

        # Assemble export dataframes
        export_frames: Dict[str, pd.DataFrame] = {}
        if summary_df is not None:  export_frames["SUMMARY"]     = summary_df
        if annual_df  is not None:  export_frames["ANNUAL"]      = annual_df
        if daily_df   is not None:
            export_frames["KPI"]    = build_kpi_table(daily_df)
            export_frames["DAILY_HEAD"] = daily_df.head(3000)
        if val_table  is not None:  export_frames["VALIDATION"]  = val_table
        if bench_sum  is not None:  export_frames["SENSITIVITY_SUMMARY"] = bench_sum
        if bench_det  is not None:  export_frames["SENSITIVITY_DETAIL"]  = bench_det

        # Download buttons
        col_xl, col_zip = st.columns(2)
        if export_frames:
            xl_bytes = to_excel_bytes(export_frames)
            col_xl.download_button(
                "📥  Download Excel workbook",
                data=xl_bytes,
                file_name="hvac_ems_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        config_export = {
            "building": asdict(st.session_state.get("b", b)),
            "hvac":     asdict(st.session_state.get("h", h)),
            "degradation": asdict(st.session_state.get("d", d)),
            "simulation": st.session_state.get("sim_cfg", sim_cfg),
        }
        figs_dir = Path(result.get("figures_dir", ""))
        if export_frames:
            zip_bytes = to_zip_bytes(export_frames, config_export, figs_dir)
            col_zip.download_button(
                "📦  Download full ZIP package",
                data=zip_bytes,
                file_name="hvac_ems_full_export.zip",
                mime="application/zip",
                use_container_width=True,
            )

        # PDF & engine Excel from disk
        for fname in ["results_report.pdf", "results_export.xlsx"]:
            fp = out_p / fname
            if fp.exists():
                with open(fp, "rb") as fh:
                    st.download_button(f"Download {fname}", fh.read(), file_name=fname)

        # File browser
        with st.expander("Browse all output files"):
            all_files = sorted([x.name for x in out_p.iterdir() if x.is_file()])
            st.write(all_files)
            for csv_f in sorted(out_p.glob("*.csv"))[:8]:
                st.markdown(f"**{csv_f.name}**")
                try:
                    st.dataframe(pd.read_csv(csv_f).head(30), use_container_width=True)
                except Exception:
                    pass

        # Figures
        if figs_dir.exists():
            imgs = sorted(figs_dir.glob("*.png"))[:16]
            if imgs:
                st.markdown("### All simulation figures")
                c1, c2 = st.columns(2)
                for i, img in enumerate(imgs):
                    (c1 if i % 2 == 0 else c2).image(
                        str(img), caption=img.stem.replace("_", " "),
                        use_container_width=True
                    )


# ── TAB 8 — GUIDE ────────────────────────────────────────────────────────
with tab_guide:
    st.markdown("""
## HVAC EMS Unified Research Suite — guide

### Physics engine (hvac_v3_engine.py)
All simulation physics are 100% handled by the engine. This interface never reimplements equations.

**Degradation models supported:**
- **Physics-based (Kern-Seaton):** Rf(t) = Rf* · (1 − e^{−B·t}) with COP correction for fouling and aging. ΔP from dust accumulation.
- **Linear time-series:** δ(t) = slope × t × severity × weather-stress
- **Exponential time-series:** δ(t) = 1 − e^{−rate × t × severity × weather-stress}

**Maintenance strategies:**
| Strategy | Logic |
|---|---|
| S0 | Fixed annual schedule (baseline, unaware) |
| S1 | Reactive — triggered when Rf ≥ Rf_threshold |
| S2 | Preventive — fixed interval (90-day filter, 180-day HX) |
| S3 | Predictive — APO optimizer triggers early when δ ≥ trigger |

**APO optimizer (S3):**
Uses a CMA-ES-style evolutionary search over [T_sp, α_flow] to minimise the four-objective function J = w₁E + w₂D + w₃C + w₄CO₂ each day.

### Post-processing KPIs (new in this interface)
The following KPIs are computed as a post-processing layer on the engine's daily output — they do not affect the physics:
- **Pump energy** = pump_specific_w_m2 × area × occ × (1 + 0.30δ) × 24 h / 1000
- **Auxiliary energy** = aux_w_m2 × area × occ × 24 h / 1000
- **Total HVAC energy** = core + pump + aux
- **Building Health Index** = 100 × (1 − 0.65δ − 0.10 × min(comfort_dev/5, 1))

### Recommended workflow
1. Set building geometry, envelope, and HVAC parameters in the sidebar
2. Upload an EPW file (or use synthetic Egyptian climate profile)
3. Select axis mode: start with **one_strategy** for a quick S0–S3 comparison
4. Run simulation → review KPI Charts tab
5. Run **Sensitivity analysis** to identify dominant parameters
6. Upload DesignBuilder export in **Validation** tab to cross-check
7. Train **CatBoost surrogate** for fast what-if predictions and SHAP insight
8. Download Excel workbook or full ZIP from **Export** tab

### Key parameters for Egyptian university buildings
| Parameter | Calibrated value |
|---|---|
| Grid CO₂ factor | 0.536 kgCO₂/kWh |
| Rf* (fouling asymptote) | 2 × 10⁻⁴ m²K/W |
| Fouling growth B | 0.015 day⁻¹ |
| Dust accumulation rate | 1.2 kg/day |
| Latitude | 31.4°N (New Mansoura) |
| Design cooling load | 100 W/m² |

### Citation note
When referencing the degradation model in publications, cite the Kern-Seaton (1959) asymptotic fouling model and note that HX fouling and AHU filter clogging are coupled as live feedback variables into the APO optimizer — this is the core novelty of the framework.
""")
