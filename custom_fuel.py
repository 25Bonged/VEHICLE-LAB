#!/usr/bin/env python3
"""
custom_fuel.py — Comprehensive Fuel Consumption Analysis Module
MDF-native fuel consumption analyzer for dashboards (compatible with app.py).

This module provides professional-grade fuel consumption analysis including:
- BSFC (Brake Specific Fuel Consumption) calculations
- Fuel flow rate analysis (volume and mass)
- Distance-based fuel consumption metrics
- Statistical summaries and distributions
- Operating point analysis (RPM vs Torque vs Fuel)
- Time-series analysis
- Efficiency metrics and calibration insights

Produces structure:
{
  "tables": {
    "Fuel Summary": [statistics],
    "Operating Point Analysis": [binned data],
    "Fuel Channels Found": [channel mapping]
  },
  "plots": {
    "BSFC vs Operating Points": {...},
    "Fuel Flow Time Series": {...},
    "BSFC Distribution": {...},
    "Speed vs Fuel Consumption": {...},
    "Fuel Efficiency Map": {...}
  },
  "meta": {}
}
"""

from __future__ import annotations
import re
import math
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from asammdf import MDF
except ImportError:
    MDF = None

# Import centralized signal mapping system
try:
    from signal_mapping import (
        SIGNAL_MAP, find_signal_advanced, find_signal_by_role,
        RPM_CANDIDATES, TORQUE_CANDIDATES, LAMBDA_CANDIDATES,
        COOLANT_TEMP_CANDIDATES, INTAKE_TEMP_CANDIDATES,
        MAP_CANDIDATES, THROTTLE_CANDIDATES
    )
    # Map our internal signal names to signal_mapping roles
    FUEL_SIGNAL_ROLES = {
        "fuel_vol_consumption": "fuel_rate",
        "fuel_mass_flow": "fuel_rate",  # May also be in fuel_rate
        "rpm": "rpm",
        "torque": "torque",
        "speed": "vehicle_speed",
        "distance": "distance",  # Will check if exists in SIGNAL_MAP
        "air_mass_flow": "air_mass_flow",
        "lambda": "lambda",
        "throttle": "throttle",
        "coolant_temp": "coolant_temp"
    }
    USE_CENTRALIZED_MAPPING = True
except ImportError:
    # Fallback to local patterns if signal_mapping not available
    USE_CENTRALIZED_MAPPING = False
    FUEL_SIGNAL_PATTERNS = {
        "fuel_vol_consumption": [
            "FuelRate", "fuel_flow", "FuelCons", "FuCns_volFuCnsTot", 
            "FuelRate_Lh", "FuelConsumption", "fuel_rate", "FuelFlowRate",
            "FR", "Fuel", "fuelFlow", "fuelCons"
        ],
        "fuel_mass_flow": [
            "fuel_mass_flow", "FuelMassFlow", "mfFuel", "mFuel",
            "fuel_mass", "FuelMass"
        ],
        "rpm": [
            "Epm_nEng", "Epm_nEng_RTE", "Ext_nEng_RTE", "EngSpeed", "rpm",
            "engine_rpm", "n_engine", "inRpm", "outRpmSpeed", "nEng",
            "EngineSpeed", "n_Eng", "Engine_RPM", "EngineSpeed_RPM"
        ],
        "torque": [
            "TqSys_tqCkEngReal", "TqSys_tqCkEngReal_RTE", "EngineTorque",
            "torque", "Torque", "inTorque", "tqEng", "trqEng", "Trq",
            "Tq", "Engine_Torque", "ActualTorque"
        ],
        "speed": [
            "VehSpd", "VehicleSpeed", "vVehicle", "Speed", "speed",
            "VehSpeed_kmh", "VS", "Vehicle_Velocity", "vVehicle_kmh"
        ],
        "distance": [
            "Distance", "Odometer", "TotalDistance", "TripDistance",
            "dist", "Dist", "VehicleDistance", "Odo"
        ],
        "air_mass_flow": [
            "air_mass_flow", "MAF_gps", "InM_mfAirCanPurgEstim", "mAir",
            "AirFlow", "MAF", "MassAirFlow", "AirMassFlow"
        ],
        "lambda": [
            "afr", "lambda", "AirFuelRatio", "Lambda", "AFR", "airFuelRatio",
            "ExL_rlamUsOxCMes_RTE", "lambda_raw"
        ],
        "throttle": [
            "ThrottlePos", "ThrottlePosition", "APP", "AccPedalPos",
            "Throttle", "AccelPedal", "PedalPos"
        ],
        "coolant_temp": [
            "ECT_C", "CoolantTemp", "temp_coolant", "ECT", "EngineTemp",
            "Coolant_Temp", "TEMP_EAU_MOT", "TEMP_COOLANT"
        ]
    }

# Constants
FUEL_DENSITY_KG_PER_L = 0.745  # Typical gasoline density
MIN_POWER_KW = 0.1  # Minimum power threshold for valid BSFC calculation
EDGE_ZOOM = 0.67
BASE_FONT = 11
LEGEND_FONT = 10
TICK_FONT = 10


def find_signal(channels: List[str], role: str, mdf=None) -> Optional[str]:
    """Find a signal name in the channel list based on role patterns."""
    if USE_CENTRALIZED_MAPPING:
        # Use centralized signal mapping
        signal_mapping_role = FUEL_SIGNAL_ROLES.get(role)
        if signal_mapping_role:
            # Check if role exists in SIGNAL_MAP
            if signal_mapping_role in SIGNAL_MAP or signal_mapping_role == "distance":
                if mdf is not None:
                    # Use find_signal_by_role if MDF object available
                    return find_signal_by_role(mdf, signal_mapping_role)
                else:
                    # Use find_signal_advanced with channel list
                    result = find_signal_advanced(channels, signal_mapping_role, fuzzy_match=True, substring_match=True)
                    if result:
                        return result
        
        # Fallback: try direct lookup for distance (may not be in SIGNAL_MAP)
        if role == "distance":
            distance_patterns = ["Distance", "Odometer", "TotalDistance", "TripDistance", "dist", "Dist", "VehicleDistance", "Odo"]
            for ch in channels:
                ch_lower = ch.lower()
                for pattern in distance_patterns:
                    if pattern.lower() in ch_lower:
                        return ch
    
    # Fallback to local patterns
    if not USE_CENTRALIZED_MAPPING:
        patterns = FUEL_SIGNAL_PATTERNS.get(role, [])
        for ch in channels:
            ch_lower = ch.lower()
            for pattern in patterns:
                if pattern.lower() in ch_lower:
                    return ch
    
    return None


def safe_read_signal(mdf: MDF, name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Safely read a signal from MDF, returning (timestamps, samples) or None."""
    try:
        sig = mdf.get(name)
        if hasattr(sig, "samples") and hasattr(sig, "timestamps"):
            samples = np.array(sig.samples, dtype=np.float64)
            timestamps = np.array(sig.timestamps, dtype=np.float64)
            # Remove NaN and invalid values
            valid_mask = np.isfinite(samples) & np.isfinite(timestamps)
            if not valid_mask.any():
                return None
            return timestamps[valid_mask], samples[valid_mask]
    except Exception:
        pass
    return None


def extract_fuel_data(files: List[Path]) -> Tuple[pd.DataFrame, Dict[str, str], List[Dict[str, Any]]]:
    """
    Extract fuel consumption data from MDF files.
    Returns: (dataframe with aligned data, channel mapping, evidence log)
    """
    if MDF is None:
        raise RuntimeError("asammdf not installed")
    
    all_data = []
    channel_mapping: Dict[str, str] = {}
    evidence: List[Dict[str, Any]] = []
    
    for file_path in files:
        if not file_path.exists():
            continue
        
        try:
            mdf = MDF(str(file_path))
            channels = list(mdf.channels_db.keys())
            
            # Find all required signals
            found_signals = {}
            file_evidence = {
                "file": file_path.name,
                "channels_found": {},
                "missing_signals": []
            }
            
            # Define roles we need (using centralized mapping if available)
            if USE_CENTRALIZED_MAPPING:
                roles_to_find = list(FUEL_SIGNAL_ROLES.keys())
            else:
                roles_to_find = list(FUEL_SIGNAL_PATTERNS.keys())
            
            for role in roles_to_find:
                ch_name = find_signal(channels, role, mdf=mdf)
                if ch_name:
                    found_signals[role] = ch_name
                    channel_mapping[role] = ch_name
                    file_evidence["channels_found"][role] = ch_name
                else:
                    file_evidence["missing_signals"].append(role)
            
            evidence.append(file_evidence)
            
            # Read all found signals
            signal_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for role, ch_name in found_signals.items():
                result = safe_read_signal(mdf, ch_name)
                if result:
                    signal_data[role] = result
            
            if not signal_data:
                mdf.close()
                continue
            
            # Align signals by timestamps (use shortest time range)
            all_times = [t for t, _ in signal_data.values()]
            if not all_times:
                mdf.close()
                continue
            
            # Find common time range
            min_time = max([t[0] for t in all_times])
            max_time = min([t[-1] for t in all_times])
            
            if min_time >= max_time:
                mdf.close()
                continue
            
            # Create uniform time grid (1 Hz sampling)
            time_grid = np.arange(min_time, max_time, 1.0)
            if len(time_grid) == 0:
                mdf.close()
                continue
            
            # Interpolate all signals to common time grid
            aligned_data = {"time": time_grid, "file": file_path.name}
            
            for role, (times, values) in signal_data.items():
                # Interpolate to common grid
                aligned_values = np.interp(time_grid, times, values, left=np.nan, right=np.nan)
                aligned_data[role] = aligned_values
            
            # Convert to DataFrame
            df_chunk = pd.DataFrame(aligned_data)
            all_data.append(df_chunk)
            
            mdf.close()
            
        except Exception as e:
            evidence.append({
                "file": file_path.name,
                "error": str(e)
            })
            continue
    
    if not all_data:
        return pd.DataFrame(), channel_mapping, evidence
    
    # Concatenate all files
    df = pd.concat(all_data, ignore_index=True)
    
    # Compute derived quantities
    df["omega_rad_s"] = np.nan
    if "rpm" in df.columns:
        df["omega_rad_s"] = df["rpm"] * (2.0 * math.pi / 60.0)
    
    # Mechanical power (kW) = torque (Nm) * angular_velocity (rad/s) / 1000
    df["mech_power_kw"] = np.nan
    if "torque" in df.columns and "omega_rad_s" in df.columns:
        df["mech_power_kw"] = df["torque"] * df["omega_rad_s"] / 1000.0
    
    # Convert fuel volume consumption to mass flow
    df["fuel_mass_flow_kgps"] = np.nan
    if "fuel_vol_consumption" in df.columns:
        series = df["fuel_vol_consumption"].abs()
        valid = series.dropna()
        if len(valid) > 0:
            median_val = valid.median()
            # Auto-detect units based on typical values
            if 0 < median_val < 200:  # Assume L/h
                df["fuel_mass_flow_kgps"] = df["fuel_vol_consumption"] * FUEL_DENSITY_KG_PER_L / 3600.0
            elif median_val >= 200:  # Assume mL/h or needs conversion
                # If values are very high (>1000), might be in different unit
                if median_val > 1000:
                    # Possibly in L/h but recorded incorrectly, or different unit
                    # Try treating as L/h anyway for plotting
                    logging.info(f"Fuel consumption median value {median_val:.2f} is high - treating as L/h")
                    df["fuel_mass_flow_kgps"] = df["fuel_vol_consumption"] * FUEL_DENSITY_KG_PER_L / 3600.0
                else:
                    # mL/h
                    df["fuel_mass_flow_kgps"] = df["fuel_vol_consumption"] * FUEL_DENSITY_KG_PER_L / 3_600_000.0
    
    # Use direct mass flow if available
    if "fuel_mass_flow" in df.columns:
        valid_mass = df["fuel_mass_flow"].dropna()
        if len(valid_mass) > 0:
            # Check units (likely kg/s or g/s)
            median_mass = valid_mass.abs().median()
            if median_mass < 0.1:  # Likely kg/s
                df.loc[valid_mass.index, "fuel_mass_flow_kgps"] = valid_mass
            elif median_mass < 100:  # Likely g/s -> convert to kg/s
                df.loc[valid_mass.index, "fuel_mass_flow_kgps"] = valid_mass / 1000.0
    
    # BSFC = Brake Specific Fuel Consumption (g/kWh)
    # BSFC = (fuel_mass_flow_kgps * 3600000) / mech_power_kw
    df["bsfc_gpkwh"] = np.nan
    MIN_POWER_KW = 0.1  # Minimum power threshold for BSFC calculation
    valid_mask = (
        df["mech_power_kw"].notna()
        & (df["mech_power_kw"] > MIN_POWER_KW)
        & df["fuel_mass_flow_kgps"].notna()
        & (df["fuel_mass_flow_kgps"] > 0)
    )
    if valid_mask.any():
        df.loc[valid_mask, "bsfc_gpkwh"] = (
            df.loc[valid_mask, "fuel_mass_flow_kgps"] * 3_600_000.0
            / df.loc[valid_mask, "mech_power_kw"]
        )
        logging.info(f"[FUEL] Calculated BSFC for {valid_mask.sum()} data points")
    else:
        logging.warning(f"[FUEL] ⚠️  Could not calculate BSFC - missing required data. mech_power_kw valid: {df['mech_power_kw'].notna().sum()}, fuel_mass_flow valid: {df['fuel_mass_flow_kgps'].notna().sum()}")
    
    # Distance-based fuel consumption (if distance and speed available)
    if "distance" in df.columns and "speed" in df.columns:
        df["distance_delta"] = df["distance"].diff().fillna(0)
        df["fuel_per_100km"] = np.nan
        valid_dist = (df["distance_delta"] > 0) & df["fuel_vol_consumption"].notna()
        if valid_dist.any():
            # L/100km = (fuel_consumed_L / distance_km) * 100
            df.loc[valid_dist, "fuel_per_100km"] = (
                df.loc[valid_dist, "fuel_vol_consumption"] * (df.loc[valid_dist, "distance_delta"] / 100000.0)
            ) * 100.0
    
    return df, channel_mapping, evidence


def compute_fuel_statistics(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Compute statistical summary of fuel consumption metrics."""
    stats = []
    
    metrics = {
        "BSFC (g/kWh)": "bsfc_gpkwh",
        "Fuel Flow Rate (L/h)": "fuel_vol_consumption",
        "Fuel Mass Flow (kg/s)": "fuel_mass_flow_kgps",
        "Mechanical Power (kW)": "mech_power_kw",
        "Engine RPM": "rpm",
        "Engine Torque (Nm)": "torque",
    }
    
    for label, col in metrics.items():
        if col not in df.columns:
            continue
        
        series = df[col].dropna()
        if len(series) == 0:
            continue
        
        stats.append({
            "metric": label,
            "count": len(series),
            "mean": round(float(series.mean()), 3),
            "median": round(float(series.median()), 3),
            "std": round(float(series.std()), 3) if len(series) > 1 else 0.0,
            "min": round(float(series.min()), 3),
            "max": round(float(series.max()), 3),
            "p25": round(float(series.quantile(0.25)), 3),
            "p75": round(float(series.quantile(0.75)), 3),
            "p95": round(float(series.quantile(0.95)), 3),
        })
    
    # Distance-based consumption
    if "fuel_per_100km" in df.columns:
        fuel_100km = df["fuel_per_100km"].dropna()
        if len(fuel_100km) > 0:
            stats.append({
                "metric": "Fuel Consumption (L/100km)",
                "count": len(fuel_100km),
                "mean": round(float(fuel_100km.mean()), 2),
                "median": round(float(fuel_100km.median()), 2),
                "std": round(float(fuel_100km.std()), 2) if len(fuel_100km) > 1 else 0.0,
                "min": round(float(fuel_100km.min()), 2),
                "max": round(float(fuel_100km.max()), 2),
                "p25": round(float(fuel_100km.quantile(0.25)), 2),
                "p75": round(float(fuel_100km.quantile(0.75)), 2),
                "p95": round(float(fuel_100km.quantile(0.95)), 2),
            })
    
    return stats


def compute_operating_point_analysis(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Analyze fuel consumption by operating points (RPM-Torque bins)."""
    if "rpm" not in df.columns or "torque" not in df.columns or "bsfc_gpkwh" not in df.columns:
        return []
    
    valid = df[["rpm", "torque", "bsfc_gpkwh", "fuel_vol_consumption"]].dropna()
    if len(valid) == 0:
        return []
    
    # Create bins
    rpm_bins = np.arange(0, 7000, 500)
    torque_bins = np.arange(-50, 600, 50)
    
    valid["rpm_bin"] = pd.cut(valid["rpm"], bins=rpm_bins, labels=rpm_bins[:-1], include_lowest=True)
    valid["torque_bin"] = pd.cut(valid["torque"], bins=torque_bins, labels=torque_bins[:-1], include_lowest=True)
    
    grouped = valid.groupby(["rpm_bin", "torque_bin"], observed=True)
    
    results = []
    for (rpm_bin, torque_bin), group in grouped:
        if len(group) < 3:  # Skip bins with too few samples
            continue
        
        results.append({
            "rpm_bin_center": float(rpm_bin),
            "torque_bin_center": float(torque_bin),
            "sample_count": len(group),
            "bsfc_mean": round(float(group["bsfc_gpkwh"].mean()), 2),
            "bsfc_median": round(float(group["bsfc_gpkwh"].median()), 2),
            "bsfc_min": round(float(group["bsfc_gpkwh"].min()), 2),
            "bsfc_max": round(float(group["bsfc_gpkwh"].max()), 2),
            "fuel_flow_mean_Lh": round(float(group["fuel_vol_consumption"].mean()), 2) if "fuel_vol_consumption" in group.columns else None,
        })
    
    return sorted(results, key=lambda x: (x["rpm_bin_center"], x["torque_bin_center"]))


def _apply_common_layout(fig: go.Figure, *, height: int, title: str) -> None:
    """Apply common layout styling to plots."""
    fig.update_layout(
        title=title,
        template="plotly_dark",
        autosize=True,
        uirevision="keep",
        margin=dict(l=60, r=40, t=54, b=120),
        height=height,
        font=dict(size=BASE_FONT, color='#dce1e6'),  # Light text for dark mode
        legend=dict(orientation="h", y=-0.22, font=dict(size=LEGEND_FONT)),
        paper_bgcolor='black',  # Deep black background
        plot_bgcolor='black'  # Deep black background
    )
    fig.update_xaxes(automargin=True, tickfont=dict(size=TICK_FONT))
    fig.update_yaxes(automargin=True, tickfont=dict(size=TICK_FONT))


def plot_bsfc_operating_points(df: pd.DataFrame, high_quality: bool = False) -> Optional[str]:
    """Create BSFC heatmap vs RPM-Torque operating points."""
    if "rpm" not in df.columns or "torque" not in df.columns:
        logging.warning("[FUEL] BSFC Operating Points: Missing rpm or torque columns")
        return None
    
    if "bsfc_gpkwh" not in df.columns:
        logging.warning("[FUEL] BSFC Operating Points: bsfc_gpkwh column not found")
        return None
    
    # Filter out inf and nan values explicitly
    valid = df[["rpm", "torque", "bsfc_gpkwh"]].copy()
    valid = valid.replace([np.inf, -np.inf], np.nan)
    valid = valid.dropna()
    
    # Also filter out zero or negative BSFC (invalid)
    valid = valid[(valid["bsfc_gpkwh"] > 0)]
    
    # Diagnostic logging before filtering
    if len(valid) > 0:
        bsfc_stats = valid["bsfc_gpkwh"]
        logging.info(f"[FUEL] BSFC Operating Points: Before filtering - {len(valid)} points, BSFC range: [{bsfc_stats.min():.2f}, {bsfc_stats.max():.2f}], mean: {bsfc_stats.mean():.2f}")
    
    # Filter out unrealistic BSFC values - use adaptive range based on data
    # Typical range is 150-450 g/kWh, but some engines/situations can have different ranges
    if len(valid) > 0:
        try:
            bsfc_min_percentile = max(0, valid["bsfc_gpkwh"].quantile(0.01))  # 1st percentile
            bsfc_max_percentile = valid["bsfc_gpkwh"].quantile(0.99)  # 99th percentile
            
            # If data has very high values, use a more reasonable cap
            # Check if the 99th percentile suggests the data might be in wrong units
            if bsfc_max_percentile > 100000:
                # Data seems to have very high values - use a wide but reasonable range
                logging.info(f"[FUEL] BSFC Operating Points: Very high BSFC values detected (99th percentile: {bsfc_max_percentile:.2f}), using wide filter")
                bsfc_min = 0
                bsfc_max = min(bsfc_max_percentile, 50000)  # Cap at 50000 for extremely high values
            elif bsfc_max_percentile > 5000:
                # Moderately high values
                bsfc_min = 0
                bsfc_max = min(bsfc_max_percentile, 10000)
            else:
                # Normal range - use percentile-based filtering with reasonable caps
                bsfc_min = max(0, bsfc_min_percentile)
                bsfc_max = min(bsfc_max_percentile, 2000)
                # Ensure min < max
                if bsfc_min >= bsfc_max:
                    bsfc_min = 0
                    bsfc_max = min(bsfc_max_percentile, 2000)
            
            # Ensure we have a valid range
            if bsfc_min >= bsfc_max:
                logging.warning(f"[FUEL] BSFC Operating Points: Invalid range [{bsfc_min:.2f}, {bsfc_max:.2f}], using fallback")
                bsfc_min = 0
                bsfc_max = 5000
            
            logging.info(f"[FUEL] BSFC Operating Points: Using adaptive filter range: [{bsfc_min:.2f}, {bsfc_max:.2f}]")
            valid = valid[(valid["bsfc_gpkwh"] >= bsfc_min) & (valid["bsfc_gpkwh"] <= bsfc_max)]
        except Exception as e:
            logging.warning(f"[FUEL] BSFC Operating Points: Error in quantile calculation: {e}, using standard range")
            # Fallback: use standard range if quantile fails
            valid = valid[(valid["bsfc_gpkwh"] > 0) & (valid["bsfc_gpkwh"] < 5000)]
    else:
        # Fallback: use standard range
        valid = valid[(valid["bsfc_gpkwh"] > 0) & (valid["bsfc_gpkwh"] < 5000)]
    
    min_samples = 3  # Minimum samples for any plot
    if len(valid) < min_samples:
        logging.info(f"[FUEL] BSFC Operating Points: Only {len(valid)} points passed initial filter, trying relaxed fallback...")
        logging.warning(f"[FUEL] BSFC Operating Points: Insufficient valid data after filtering ({len(valid)} < {min_samples} required)")
        # Try with even more relaxed filter as fallback
        valid_unfiltered = df[["rpm", "torque", "bsfc_gpkwh"]].copy()
        valid_unfiltered = valid_unfiltered.replace([np.inf, -np.inf], np.nan)
        valid_unfiltered = valid_unfiltered.dropna()
        # Use a very wide range for fallback - handle very high BSFC values
        if len(valid_unfiltered) > 0:
            max_bsfc = valid_unfiltered["bsfc_gpkwh"].max()
            if max_bsfc > 100000:
                fallback_max = min(max_bsfc * 1.1, 200000)  # Allow 10% above max, cap at 200k
            elif max_bsfc > 10000:
                fallback_max = min(max_bsfc * 1.1, 100000)  # Cap at 100k
            else:
                fallback_max = 10000  # Standard high cap
            valid_unfiltered = valid_unfiltered[(valid_unfiltered["bsfc_gpkwh"] > 0) & (valid_unfiltered["bsfc_gpkwh"] < fallback_max)]
        else:
            valid_unfiltered = valid_unfiltered[(valid_unfiltered["bsfc_gpkwh"] > 0) & (valid_unfiltered["bsfc_gpkwh"] < 50000)]
        if len(valid_unfiltered) >= min_samples:
            logging.info(f"[FUEL] BSFC Operating Points: Using relaxed filter - {len(valid_unfiltered)} points")
            valid = valid_unfiltered
        else:
            return None
    
    logging.info(f"[FUEL] BSFC Operating Points: Creating plot with {len(valid)} valid points")
    
    # Create 2D histogram
    fig = go.Figure()
    
    # Use scatter plot with color intensity for BSFC
    # Calculate reasonable color scale bounds
    bsfc_values = valid["bsfc_gpkwh"]
    cmin = max(bsfc_values.quantile(0.01), 0)  # Use 1st percentile to exclude extreme outliers
    cmax = min(bsfc_values.quantile(0.99), 1000)  # Cap at 1000 g/kWh
    
    fig.add_trace(go.Scatter(
        x=valid["rpm"].tolist(),
        y=valid["torque"].tolist(),
        mode="markers",
        marker=dict(
            size=4 if high_quality else 3,
            color=valid["bsfc_gpkwh"].tolist(),
            colorscale="Viridis",
            colorbar=dict(title="BSFC (g/kWh)", len=0.8),
            showscale=True,
            cmin=float(cmin),
            cmax=float(cmax),
            line=dict(width=0) if not high_quality else None,
        ),
        hovertemplate="RPM: %{x:.0f}<br>Torque: %{y:.1f} Nm<br>BSFC: %{marker.color:.1f} g/kWh<extra></extra>",
        name="Operating Points"
    ))
    
    fig.update_layout(
        xaxis_title="Engine RPM",
        yaxis_title="Engine Torque (Nm)",
        title="BSFC vs Operating Points",
    )
    _apply_common_layout(fig, height=int(500 / EDGE_ZOOM), title="BSFC vs Operating Points")
    
    return fig.to_json()


def plot_fuel_flow_timeseries(df: pd.DataFrame) -> Optional[str]:
    """Plot fuel flow rate over time."""
    if "time" not in df.columns:
        return None
    
    fig = go.Figure()
    
    # Fuel volume consumption
    if "fuel_vol_consumption" in df.columns:
        valid = df[["time", "fuel_vol_consumption"]].dropna()
        if len(valid) > 0:
            fig.add_trace(go.Scatter(
                x=valid["time"].tolist(),
                y=valid["fuel_vol_consumption"].tolist(),
                mode="lines",
                name="Fuel Flow (L/h)",
                line=dict(color="#4e79a7", width=1.5)
            ))
    
    # Fuel mass flow
    if "fuel_mass_flow_kgps" in df.columns:
        valid = df[["time", "fuel_mass_flow_kgps"]].dropna()
        if len(valid) > 0:
            mass_flow_gps = (valid["fuel_mass_flow_kgps"] * 1000).tolist()  # Convert to g/s for better scale
            fig.add_trace(go.Scatter(
                x=valid["time"].tolist(),
                y=mass_flow_gps,
                mode="lines",
                name="Fuel Mass Flow (g/s)",
                line=dict(color="#f28e2b", width=1.5),
                yaxis="y2"
            ))
    
    if len(fig.data) == 0:
        return None
    
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Fuel Flow (L/h)",
        yaxis2=dict(title="Fuel Mass Flow (g/s)", overlaying="y", side="right"),
    )
    _apply_common_layout(fig, height=int(400 / EDGE_ZOOM), title="Fuel Flow Time Series")
    
    return fig.to_json()


def plot_bsfc_distribution(df: pd.DataFrame, high_quality: bool = False) -> Optional[str]:
    """Plot BSFC distribution histogram."""
    if "bsfc_gpkwh" not in df.columns:
        logging.warning("[FUEL] BSFC Distribution: bsfc_gpkwh column not found")
        return None
    
    # Filter out inf and nan values explicitly
    valid = df["bsfc_gpkwh"].copy()
    valid = valid.replace([np.inf, -np.inf], np.nan)
    valid = valid.dropna()
    
    # Also filter out zero or negative BSFC (invalid)
    valid = valid[(valid > 0)]
    
    # Diagnostic logging before filtering
    if len(valid) > 0:
        logging.info(f"[FUEL] BSFC Distribution: Before filtering - {len(valid)} points, range: [{valid.min():.2f}, {valid.max():.2f}], mean: {valid.mean():.2f}")
    
    # Filter unrealistic BSFC values - use adaptive range
    if len(valid) > 0:
        try:
            bsfc_min_percentile = max(0, valid.quantile(0.01))
            bsfc_max_percentile = valid.quantile(0.99)
            
            # If data has very high values, use a more reasonable cap
            if bsfc_max_percentile > 100000:
                logging.info(f"[FUEL] BSFC Distribution: Very high BSFC values detected (99th percentile: {bsfc_max_percentile:.2f}), using wide filter")
                bsfc_min = 0
                bsfc_max = min(bsfc_max_percentile, 50000)
            elif bsfc_max_percentile > 5000:
                bsfc_min = 0
                bsfc_max = min(bsfc_max_percentile, 10000)
            else:
                bsfc_min = max(0, bsfc_min_percentile)
                bsfc_max = min(bsfc_max_percentile, 2000)
                if bsfc_min >= bsfc_max:
                    bsfc_min = 0
                    bsfc_max = min(bsfc_max_percentile, 2000)
            
            # Ensure we have a valid range
            if bsfc_min >= bsfc_max:
                logging.warning(f"[FUEL] BSFC Distribution: Invalid range [{bsfc_min:.2f}, {bsfc_max:.2f}], using fallback")
                bsfc_min = 0
                bsfc_max = 5000
            
            logging.info(f"[FUEL] BSFC Distribution: Using adaptive filter range: [{bsfc_min:.2f}, {bsfc_max:.2f}]")
            valid = valid[(valid >= bsfc_min) & (valid <= bsfc_max)]
        except Exception as e:
            logging.warning(f"[FUEL] BSFC Distribution: Error in quantile calculation: {e}, using standard range")
            valid = valid[(valid > 0) & (valid < 5000)]
    else:
        valid = valid[(valid > 0) & (valid < 5000)]
    
    min_samples = 5  # Minimum samples for histogram
    if len(valid) < min_samples:
        logging.info(f"[FUEL] BSFC Distribution: Only {len(valid)} points passed initial filter, trying relaxed fallback...")
        logging.warning(f"[FUEL] BSFC Distribution: Insufficient valid data after filtering ({len(valid)} < {min_samples} required)")
        # Try with relaxed filter as fallback
        valid_unfiltered = df["bsfc_gpkwh"].copy()
        valid_unfiltered = valid_unfiltered.replace([np.inf, -np.inf], np.nan)
        valid_unfiltered = valid_unfiltered.dropna()
        # Use a very wide range for fallback - handle very high BSFC values
        if len(valid_unfiltered) > 0:
            max_bsfc = valid_unfiltered.max()
            if max_bsfc > 100000:
                fallback_max = min(max_bsfc * 1.1, 200000)
            elif max_bsfc > 10000:
                fallback_max = min(max_bsfc * 1.1, 100000)
            else:
                fallback_max = 10000
            valid_unfiltered = valid_unfiltered[(valid_unfiltered > 0) & (valid_unfiltered < fallback_max)]
        else:
            valid_unfiltered = valid_unfiltered[(valid_unfiltered > 0) & (valid_unfiltered < 50000)]
        if len(valid_unfiltered) >= min_samples:
            logging.info(f"[FUEL] BSFC Distribution: Using relaxed filter - {len(valid_unfiltered)} points")
            valid = valid_unfiltered
        else:
            return None
    
    logging.info(f"[FUEL] BSFC Distribution: Creating plot with {len(valid)} valid points")
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=valid.tolist(),
        nbinsx=50,
        marker_color="#59a14f",
        name="BSFC Distribution",
        hovertemplate="BSFC: %{x:.1f} g/kWh<br>Count: %{y}<extra></extra>"
    ))
    
    # Add vertical lines for percentiles (only if we have enough data)
    if len(valid) >= 10:
        median_val = float(valid.median())
        p25_val = float(valid.quantile(0.25))
        p75_val = float(valid.quantile(0.75))
        
        fig.add_vline(x=median_val, line_dash="dash", line_color="red", 
                     annotation_text=f"Median: {median_val:.1f}", 
                     annotation_position="top")
        if p25_val != p75_val:
            fig.add_vline(x=p25_val, line_dash="dot", line_color="orange", 
                         annotation_text=f"P25: {p25_val:.1f}", 
                         annotation_position="top")
            fig.add_vline(x=p75_val, line_dash="dot", line_color="orange", 
                         annotation_text=f"P75: {p75_val:.1f}", 
                         annotation_position="top")
    
    fig.update_layout(
        xaxis_title="BSFC (g/kWh)",
        yaxis_title="Frequency",
    )
    _apply_common_layout(fig, height=int(400 / EDGE_ZOOM), title="BSFC Distribution")
    
    return fig.to_json()


def plot_speed_vs_fuel(df: pd.DataFrame, high_quality: bool = False) -> Optional[str]:
    """Plot speed vs fuel consumption correlation."""
    
    # Check for speed column variations (prioritize role names from extract_fuel_data)
    speed_col = None
    speed_candidates = [
        "speed",  # Role name from extract_fuel_data
        "vehicle_speed", "veh_speed", "vehspd", "spd", 
        "Veh_spdVeh", "Ext_spdVeh", "VITESSE_VEHICULE_ROUES",
        "VehSpd", "VehicleSpeed", "vVehicle", "Speed",
        "VehSpeed_kmh", "VS", "Vehicle_Velocity", "vVehicle_kmh"
    ]
    
    # Check exact matches first
    for col in speed_candidates:
        if col in df.columns:
            speed_col = col
            break
    
    # If not found, try case-insensitive match
    if speed_col is None:
        df_cols_lower = {col.lower(): col for col in df.columns}
        for candidate in speed_candidates:
            if candidate.lower() in df_cols_lower:
                speed_col = df_cols_lower[candidate.lower()]
                break
    
    # If still not found, try partial match
    if speed_col is None:
        for col in df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ["speed", "spd", "veh", "vitesse"]):
                speed_col = col
                logging.debug(f"Speed vs Fuel: Found speed column by partial match: {col}")
                break
    
    if speed_col is None:
        logging.debug(f"Speed vs Fuel: No speed column found. Available columns: {list(df.columns)}")
        return None
    
    # Check for fuel column variations (prioritize role names from extract_fuel_data)
    fuel_col = None
    fuel_candidates = [
        "fuel_vol_consumption",  # Primary role name from extract_fuel_data
        "fuel_mass_flow",  # Alternative role name
        "fuel_per_100km", "fuel_consumption", 
        "fuel_rate", "fuel_flow", "FuelMassFlow", "FuelRate",
        "FuelRate", "fuel_flow", "FuelCons", "FuCns_volFuCnsTot",
        "FuelRate_Lh", "FuelConsumption", "fuel_rate", "FuelFlowRate",
        "FR", "Fuel", "fuelFlow", "fuelCons"
    ]
    
    # Check exact matches first
    for col in fuel_candidates:
        if col in df.columns:
            fuel_col = col
            break
    
    # If not found, try case-insensitive match
    if fuel_col is None:
        df_cols_lower = {col.lower(): col for col in df.columns}
        for candidate in fuel_candidates:
            if candidate.lower() in df_cols_lower:
                fuel_col = df_cols_lower[candidate.lower()]
                break
    
    # If still not found, try partial match
    if fuel_col is None:
        for col in df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ["fuel", "consumption", "flow"]):
                fuel_col = col
                logging.debug(f"Speed vs Fuel: Found fuel column by partial match: {col}")
                break
    
    if fuel_col is None:
        logging.debug(f"Speed vs Fuel: No fuel column found. Available columns: {list(df.columns)}")
        return None
    
    # Determine fuel label based on column
    if "100km" in fuel_col.lower() or "per_100km" in fuel_col.lower():
        fuel_label = "Fuel Consumption (L/100km)"
    elif "flow" in fuel_col.lower() or "rate" in fuel_col.lower():
        fuel_label = "Fuel Flow (L/h)"
    else:
        fuel_label = "Fuel Consumption"
    
    # Log successful column detection
    logging.info(f"Speed vs Fuel: Using speed column '{speed_col}' and fuel column '{fuel_col}'")
    
    # Debug: Print available columns and data info
    logging.debug(f"Speed vs Fuel: Available columns: {list(df.columns)}")
    logging.debug(f"Speed vs Fuel: DataFrame shape: {df.shape}")
    
    # Check if columns have any valid data
    speed_has_data = not df[speed_col].isna().all()
    fuel_has_data = not df[fuel_col].isna().all()
    
    if not speed_has_data or not fuel_has_data:
        logging.warning(f"Speed vs Fuel: Columns have no valid data. Speed NaN: {df[speed_col].isna().sum()}/{len(df)}, Fuel NaN: {df[fuel_col].isna().sum()}/{len(df)}")
        logging.warning(f"  Speed column '{speed_col}': min={df[speed_col].min() if speed_has_data else 'N/A'}, max={df[speed_col].max() if speed_has_data else 'N/A'}")
        logging.warning(f"  Fuel column '{fuel_col}': min={df[fuel_col].min() if fuel_has_data else 'N/A'}, max={df[fuel_col].max() if fuel_has_data else 'N/A'}")
        return None
    
    # Get non-null data
    valid = df[[speed_col, fuel_col]].dropna()
    initial_count = len(valid)
    logging.debug(f"Speed vs Fuel: Initial non-null pairs: {initial_count}")
    
    if initial_count == 0:
        logging.warning(f"Speed vs Fuel: No overlapping non-null data between '{speed_col}' and '{fuel_col}'")
        return None
    
    # Filter out infinite values
    valid = valid[np.isfinite(valid[speed_col]) & np.isfinite(valid[fuel_col])]
    if len(valid) == 0:
        logging.warning(f"Speed vs Fuel: All values are infinite after filtering")
        return None
    
    # Filter out negative speeds and extreme values (more lenient)
    valid = valid[(valid[speed_col] >= 0) & (valid[speed_col] <= 400)]  # Increased max speed to 400 km/h
    after_speed_filter = len(valid)
    
    if after_speed_filter == 0:
        speed_range = f"{df[speed_col].min():.1f} to {df[speed_col].max():.1f}"
        logging.warning(f"Speed vs Fuel: All speeds filtered out. Speed range: {speed_range}")
        return None
    
    # Fuel filtering - be more lenient
    if "100km" in fuel_col.lower() or "per_100km" in fuel_col.lower():
        # Consumption in L/100km
        valid = valid[(valid[fuel_col] > 0) & (valid[fuel_col] < 100)]  # Increased from 50 to 100
    else:
        # For flow/rate, allow very wide range
        valid = valid[(valid[fuel_col] >= 0) & (valid[fuel_col] < 1000)]  # Increased from 500 to 1000
    
    if len(valid) == 0:
        fuel_range = f"{df[fuel_col].min():.2f} to {df[fuel_col].max():.2f}"
        logging.warning(f"Speed vs Fuel: All fuel values filtered out. Fuel range: {fuel_range}")
        return None
    
    min_samples = 2  # Further reduced to just 2 points minimum
    if len(valid) < min_samples:
        # Enhanced debug logging for troubleshooting
        logging.warning(f"Speed vs Fuel plot: Insufficient data. Found {len(valid)} valid points (minimum {min_samples} required)")
        logging.warning(f"  Initial non-null count: {initial_count}")
        logging.warning(f"  After speed filter: {after_speed_filter}")
        logging.warning(f"  Final valid count: {len(valid)}")
        if len(valid) > 0:
            logging.warning(f"  Speed column '{speed_col}': range in valid data: {valid[speed_col].min():.1f} to {valid[speed_col].max():.1f}")
            logging.warning(f"  Fuel column '{fuel_col}': range in valid data: {valid[fuel_col].min():.2f} to {valid[fuel_col].max():.2f}")
        else:
            logging.warning(f"  Speed column '{speed_col}': full range: {df[speed_col].min():.1f} to {df[speed_col].max():.1f}")
            logging.warning(f"  Fuel column '{fuel_col}': full range: {df[fuel_col].min():.2f} to {df[fuel_col].max():.2f}")
        logging.warning(f"  DataFrame shape: {df.shape}")
        return None
    
    logging.info(f"Speed vs Fuel: Creating plot with {len(valid)} valid data points (speed: {valid[speed_col].min():.1f}-{valid[speed_col].max():.1f} km/h, fuel: {valid[fuel_col].min():.2f}-{valid[fuel_col].max():.2f})")
    
    try:
        fig = go.Figure()
        
        # Ensure we have numeric data
        x_data = pd.to_numeric(valid[speed_col], errors='coerce').dropna().tolist()
        y_data = pd.to_numeric(valid[fuel_col], errors='coerce').dropna().tolist()
        
        # Align data lengths (in case of any mismatches)
        min_len = min(len(x_data), len(y_data))
        if min_len == 0:
            logging.warning("Speed vs Fuel: No valid numeric data after conversion")
            return None
        
        x_data = x_data[:min_len]
        y_data = y_data[:min_len]
        
        # Vibrant colorful markers - FEV brand red with better visibility
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode="markers",
            marker=dict(size=4, color="#E91E63", opacity=0.7, line=dict(color="#FFFFFF", width=0.5)),
            name="Data Points",
            hovertemplate="Speed: %{x:.1f} km/h<br>" + fuel_label + ": %{y:.2f}<extra></extra>"
        ))
        
        # Add trend line with vibrant color (if enough points)
        if len(x_data) >= 3:
            try:
                # Use numpy arrays for polyfit
                x_array = np.array(x_data)
                y_array = np.array(y_data)
                z = np.polyfit(x_array, y_array, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(x_array), max(x_array), 100).tolist()
                y_trend = p(x_trend).tolist()
                fig.add_trace(go.Scatter(
                    x=x_trend,
                    y=y_trend,
                    mode="lines",
                    name="Trend Line",
                    line=dict(color="#FFD700", width=3, dash="dash"),
                    hovertemplate="Trend Line<extra></extra>"
                ))
            except Exception as e:
                logging.debug(f"Speed vs Fuel: Could not add trend line: {e}")
                pass  # Skip trend line if fitting fails
        
        fig.update_layout(
            xaxis_title="Vehicle Speed (km/h)",
            yaxis_title=fuel_label,
        )
        _apply_common_layout(fig, height=int(400 / EDGE_ZOOM), title="Speed vs Fuel Consumption")
        
        plot_json = fig.to_json()
        logging.info(f"Speed vs Fuel: Plot created successfully ({len(plot_json)} chars)")
        return plot_json
        
    except Exception as e:
        logging.error(f"Speed vs Fuel: Error creating plot: {e}", exc_info=True)
        return None


def plot_fuel_efficiency_map(df: pd.DataFrame, high_quality: bool = False) -> Optional[str]:
    """Create 2D efficiency map (BSFC heatmap on RPM-Torque grid)."""
    if "rpm" not in df.columns or "torque" not in df.columns:
        logging.warning("[FUEL] Fuel Efficiency Map: Missing rpm or torque columns")
        return None
    
    if "bsfc_gpkwh" not in df.columns:
        logging.warning("[FUEL] Fuel Efficiency Map: bsfc_gpkwh column not found")
        return None
    
    # Filter out inf and nan values explicitly
    valid = df[["rpm", "torque", "bsfc_gpkwh"]].copy()
    valid = valid.replace([np.inf, -np.inf], np.nan)
    valid = valid.dropna()
    
    # Also filter out zero or negative BSFC (invalid)
    valid = valid[(valid["bsfc_gpkwh"] > 0)]
    
    # Diagnostic logging before filtering
    if len(valid) > 0:
        bsfc_stats = valid["bsfc_gpkwh"]
        logging.info(f"[FUEL] Fuel Efficiency Map: Before filtering - {len(valid)} points, BSFC range: [{bsfc_stats.min():.2f}, {bsfc_stats.max():.2f}]")
    
    # Filter unrealistic BSFC values - use adaptive range
    if len(valid) > 0:
        try:
            bsfc_min_percentile = max(0, valid["bsfc_gpkwh"].quantile(0.01))
            bsfc_max_percentile = valid["bsfc_gpkwh"].quantile(0.99)
            
            # If data has very high values, use a more reasonable cap
            if bsfc_max_percentile > 100000:
                logging.info(f"[FUEL] Fuel Efficiency Map: Very high BSFC values detected (99th percentile: {bsfc_max_percentile:.2f}), using wide filter")
                bsfc_min = 0
                bsfc_max = min(bsfc_max_percentile, 50000)
            elif bsfc_max_percentile > 5000:
                bsfc_min = 0
                bsfc_max = min(bsfc_max_percentile, 10000)
            else:
                bsfc_min = max(0, bsfc_min_percentile)
                bsfc_max = min(bsfc_max_percentile, 2000)
                if bsfc_min >= bsfc_max:
                    bsfc_min = 0
                    bsfc_max = min(bsfc_max_percentile, 2000)
            
            # Ensure we have a valid range
            if bsfc_min >= bsfc_max:
                logging.warning(f"[FUEL] Fuel Efficiency Map: Invalid range [{bsfc_min:.2f}, {bsfc_max:.2f}], using fallback")
                bsfc_min = 0
                bsfc_max = 5000
            
            logging.info(f"[FUEL] Fuel Efficiency Map: Using adaptive filter range: [{bsfc_min:.2f}, {bsfc_max:.2f}]")
            valid = valid[(valid["bsfc_gpkwh"] >= bsfc_min) & (valid["bsfc_gpkwh"] <= bsfc_max)]
        except Exception as e:
            logging.warning(f"[FUEL] Fuel Efficiency Map: Error in quantile calculation: {e}, using standard range")
            valid = valid[(valid["bsfc_gpkwh"] > 0) & (valid["bsfc_gpkwh"] < 5000)]
    else:
        valid = valid[(valid["bsfc_gpkwh"] > 0) & (valid["bsfc_gpkwh"] < 5000)]
    
    min_samples = 10  # Reduced minimum samples for heatmap (was 20)
    if len(valid) < min_samples:
        logging.info(f"[FUEL] Fuel Efficiency Map: Only {len(valid)} points passed initial filter, trying relaxed fallback...")
        logging.warning(f"[FUEL] Fuel Efficiency Map: Insufficient valid data after filtering ({len(valid)} < {min_samples} required)")
        # Try with relaxed filter as fallback
        valid_unfiltered = df[["rpm", "torque", "bsfc_gpkwh"]].copy()
        valid_unfiltered = valid_unfiltered.replace([np.inf, -np.inf], np.nan)
        valid_unfiltered = valid_unfiltered.dropna()
        # Use a very wide range for fallback - handle very high BSFC values
        if len(valid_unfiltered) > 0:
            max_bsfc = valid_unfiltered["bsfc_gpkwh"].max()
            if max_bsfc > 100000:
                fallback_max = min(max_bsfc * 1.1, 200000)
            elif max_bsfc > 10000:
                fallback_max = min(max_bsfc * 1.1, 100000)
            else:
                fallback_max = 10000
            valid_unfiltered = valid_unfiltered[(valid_unfiltered["bsfc_gpkwh"] > 0) & (valid_unfiltered["bsfc_gpkwh"] < fallback_max)]
        else:
            valid_unfiltered = valid_unfiltered[(valid_unfiltered["bsfc_gpkwh"] > 0) & (valid_unfiltered["bsfc_gpkwh"] < 50000)]
        if len(valid_unfiltered) >= min_samples:
            logging.info(f"[FUEL] Fuel Efficiency Map: Using relaxed filter - {len(valid_unfiltered)} points")
            valid = valid_unfiltered
        else:
            return None
    
    logging.info(f"[FUEL] Fuel Efficiency Map: Creating plot with {len(valid)} valid points")
    
    # Create bins - adaptive based on data range and quality mode
    rpm_range = valid["rpm"].max() - valid["rpm"].min()
    torque_range = valid["torque"].max() - valid["torque"].min()
    
    if rpm_range < 100 or torque_range < 10:
        # Too narrow range, use scatter instead
        return None
    
    # Determine number of bins based on data and quality mode
    n_bins = 25 if high_quality else 15
    rpm_bins = np.linspace(valid["rpm"].min(), valid["rpm"].max(), n_bins + 1)
    torque_bins = np.linspace(valid["torque"].min(), valid["torque"].max(), n_bins + 1)
    
    # Calculate mean BSFC for each bin
    valid["rpm_bin"] = pd.cut(valid["rpm"], bins=rpm_bins, labels=rpm_bins[:-1], include_lowest=True)
    valid["torque_bin"] = pd.cut(valid["torque"], bins=torque_bins, labels=torque_bins[:-1], include_lowest=True)
    
    grouped = valid.groupby(["rpm_bin", "torque_bin"], observed=True)
    
    # Build 2D array for heatmap
    heatmap_data = np.full((len(rpm_bins)-1, len(torque_bins)-1), np.nan)
    rpm_centers = []
    torque_centers = []
    
    for i, rpm_center in enumerate(rpm_bins[:-1]):
        rpm_centers.append(rpm_center)
        for j, torque_center in enumerate(torque_bins[:-1]):
            if i == 0:
                torque_centers.append(torque_center)
            
            try:
                if (rpm_center, torque_center) in grouped.groups:
                    group = grouped.get_group((rpm_center, torque_center))
                    if len(group) >= 2:  # Minimum samples per bin
                        # Use median for more robust estimate
                        heatmap_data[i, j] = float(group["bsfc_gpkwh"].median())
            except (KeyError, ValueError):
                pass
    
    # Check if we have any valid data in the heatmap
    if np.isnan(heatmap_data).all():
        return None
    
    # Convert heatmap data and centers to lists for JSON serialization
    # Replace NaN with None for proper JSON serialization
    heatmap_list = []
    for row in heatmap_data:
        heatmap_list.append([float(x) if not np.isnan(x) else None for x in row])
    
    rpm_list = [float(x) for x in rpm_centers]
    torque_list = [float(x) for x in torque_centers]
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_list,
        x=torque_list,
        y=rpm_list,
        colorscale="Viridis",
        colorbar=dict(title="BSFC (g/kWh)", len=0.8),
        hovertemplate="RPM: %{y:.0f}<br>Torque: %{x:.1f} Nm<br>BSFC: %{z:.1f} g/kWh<extra></extra>",
        zsmooth='best'  # Smooth interpolation
    ))
    
    fig.update_layout(
        xaxis_title="Torque (Nm)",
        yaxis_title="RPM",
    )
    _apply_common_layout(fig, height=int(500 / EDGE_ZOOM), title="Fuel Efficiency Map (BSFC Heatmap)")
    
    return fig.to_json()


# --------------------------- Public API ---------------------------------------
def compute_fuel(files: List[Path], include_plots: bool = True, high_quality: bool = False, **_) -> Dict[str, Any]:
    """
    Main API function to compute fuel consumption analysis.
    
    Args:
        files: List of MDF file paths
        include_plots: Whether to generate plots
        high_quality: If True, use stricter thresholds for more accurate plots (may take longer)
    
    Returns:
        Dictionary with tables, plots, and metadata
    """
    try:
        df, channel_mapping, evidence = extract_fuel_data(files)
    except Exception as e:
        return {
            "tables": {"Fuel Summary": []},
            "plots": {},
            "meta": {"error": str(e)}
        }
    
    if df.empty:
        return {
            "tables": {"Fuel Summary": []},
            "plots": {},
            "meta": {"error": "No fuel data found in files"},
            "channels": evidence
        }
    
    # Compute statistics
    statistics = compute_fuel_statistics(df)
    
    # Operating point analysis
    operating_points = compute_operating_point_analysis(df)
    
    # Channel mapping table
    channel_table = [
        {"signal_role": role, "channel_name": ch_name, "status": "Found"}
        for role, ch_name in channel_mapping.items()
    ]
    
    tables = {
        "Fuel Summary": statistics,
        "Operating Point Analysis": operating_points[:100],  # Limit to top 100
        "Fuel Channels Found": channel_table
    }
    
    plots: Dict[str, Dict[str, str]] = {}
    
    if include_plots:
        logging.info(f"[FUEL] Starting plot generation - {len(df)} data points available")
        
        # Generate all plots with quality setting
        try:
            bsfc_op = plot_bsfc_operating_points(df, high_quality=high_quality)
            if bsfc_op:
                plots["BSFC vs Operating Points"] = {"type": "plotly", "plotly_json": bsfc_op}
                logging.info("[FUEL] ✅ Created: BSFC vs Operating Points")
            else:
                logging.warning("[FUEL] ⚠️  BSFC vs Operating Points returned None")
        except Exception as e:
            logging.error(f"[FUEL] ❌ Failed to create BSFC vs Operating Points: {e}", exc_info=True)
        
        try:
            fuel_ts = plot_fuel_flow_timeseries(df)
            if fuel_ts:
                plots["Fuel Flow Time Series"] = {"type": "plotly", "plotly_json": fuel_ts}
                logging.info("[FUEL] ✅ Created: Fuel Flow Time Series")
            else:
                logging.warning("[FUEL] ⚠️  Fuel Flow Time Series returned None")
        except Exception as e:
            logging.error(f"[FUEL] ❌ Failed to create Fuel Flow Time Series: {e}", exc_info=True)
        
        try:
            bsfc_dist = plot_bsfc_distribution(df, high_quality=high_quality)
            if bsfc_dist:
                plots["BSFC Distribution"] = {"type": "plotly", "plotly_json": bsfc_dist}
                logging.info("[FUEL] ✅ Created: BSFC Distribution")
            else:
                logging.warning("[FUEL] ⚠️  BSFC Distribution returned None")
        except Exception as e:
            logging.error(f"[FUEL] ❌ Failed to create BSFC Distribution: {e}", exc_info=True)
        
        try:
            speed_fuel = plot_speed_vs_fuel(df, high_quality=high_quality)
            if speed_fuel:
                plots["Speed vs Fuel Consumption"] = {"type": "plotly", "plotly_json": speed_fuel}
                logging.info("[FUEL] ✅ Created: Speed vs Fuel Consumption")
            else:
                logging.warning("[FUEL] ⚠️  Speed vs Fuel Consumption returned None - check logs for details")
        except Exception as e:
            logging.error(f"[FUEL] ❌ Failed to create Speed vs Fuel Consumption: {e}", exc_info=True)
        
        try:
            efficiency_map = plot_fuel_efficiency_map(df, high_quality=high_quality)
            if efficiency_map:
                plots["Fuel Efficiency Map"] = {"type": "plotly", "plotly_json": efficiency_map}
                logging.info("[FUEL] ✅ Created: Fuel Efficiency Map")
            else:
                logging.warning("[FUEL] ⚠️  Fuel Efficiency Map returned None")
        except Exception as e:
            logging.error(f"[FUEL] ❌ Failed to create Fuel Efficiency Map: {e}", exc_info=True)
        
        logging.info(f"[FUEL] Plot generation complete - Created {len(plots)} plots: {list(plots.keys())}")
    
    files_processed = 0
    if "file" in df.columns:
        files_processed = len(df["file"].dropna().unique())
    
    # Add diagnostic info about speed vs fuel plot
    diagnostic_info = {}
    if "speed" in df.columns and "fuel_vol_consumption" in df.columns:
        speed_non_null = df["speed"].notna().sum()
        fuel_non_null = df["fuel_vol_consumption"].notna().sum()
        diagnostic_info["speed_vs_fuel_columns_found"] = True
        diagnostic_info["speed_non_null_count"] = int(speed_non_null)
        diagnostic_info["fuel_non_null_count"] = int(fuel_non_null)
    else:
        diagnostic_info["speed_vs_fuel_columns_found"] = False
        diagnostic_info["available_columns"] = list(df.columns)
    
    meta = {
        "total_samples": len(df),
        "files_processed": files_processed,
        "channels_found": len(channel_mapping),
        "evidence": evidence,
        "diagnostics": diagnostic_info
    }
    
    return {
        "tables": tables,
        "plots": plots,
        "meta": meta
    }


def compute_fuel_plotly(files: List[str], high_quality: bool = False) -> Dict[str, Any]:
    """Convenience function for app.py integration."""
    return compute_fuel([Path(f) for f in files], include_plots=True, high_quality=high_quality)


# --------------------------- CLI ----------------------------------------------
if __name__ == "__main__":
    import argparse
    import json
    
    ap = argparse.ArgumentParser(description="Fuel Consumption Analysis")
    ap.add_argument("--files", required=True, nargs="+", help="MDF file paths")
    ap.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = ap.parse_args()
    
    result = compute_fuel([Path(f) for f in args.files], include_plots=not args.no_plots)
    print(json.dumps(result, indent=2)[:50000])

