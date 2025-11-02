#!/usr/bin/env python3
"""
custom_cc_sl.py — Advanced Cruise Control & Speed Limiter Analyzer

Enhanced with:
- Comprehensive signal detection using signal_mapping
- Advanced control metrics (response time, settling time, steady-state error)
- PID control analysis and regulation accuracy
- Correlation with throttle, brake, gear signals
- Statistical analysis and performance assessment
- Multi-file analysis support

Outputs JSON payload with:
  - Enhanced plots with statistical overlays
  - Performance metrics tables
  - Control system analysis
  - Correlation analysis
"""

import json
import sys
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

# Import signal mapping for comprehensive signal detection
try:
    from signal_mapping import (
        SIGNAL_MAP, find_signal_advanced, find_multiple_signals,
        VEHICLE_SPEED_CANDIDATES, THROTTLE_CANDIDATES, GEAR_CANDIDATES
    )
except ImportError:
    # Fallback if signal_mapping not available
    SIGNAL_MAP = {}
    def find_signal_advanced(channels, role, **kwargs):
        return None
    def find_multiple_signals(channels, roles, **kwargs):
        return {}
    VEHICLE_SPEED_CANDIDATES = ['VehicleSpeed', 'VehSpd', 'speed']
    THROTTLE_CANDIDATES = ['Throttle', 'throttle', 'APP']
    GEAR_CANDIDATES = ['Gear', 'gear']


# ==================== Advanced Signal Detection ====================

def find_channel_comprehensive(mdf: MDF, role: str, fallback_candidates: List[str]) -> Optional[pd.Series]:
    """
    Comprehensive channel finder using signal_mapping with fallback.
    
    Args:
        mdf: MDF file object
        role: Signal role (e.g., 'vehicle_speed', 'throttle')
        fallback_candidates: Fallback candidate names
        
    Returns:
        pandas.Series with signal data or None
    """
    try:
        channels = list(mdf.channels_db.keys())
        
        # Try signal_mapping first
        if SIGNAL_MAP:
            channel_name = find_signal_advanced(channels, role)
            if channel_name:
                try:
                    sig = mdf.get(channel_name)
                    return pd.Series(sig.samples, index=pd.to_datetime(sig.timestamps, unit="s"))
                except Exception:
                    pass
        
        # Fallback to direct candidates
        for ch in fallback_candidates:
            try:
                sig = mdf.get(ch)
                return pd.Series(sig.samples, index=pd.to_datetime(sig.timestamps, unit="s"))
            except Exception:
                continue
        
        # Fuzzy search in channel names
        ch_lower = {c.lower(): c for c in channels}
        role_parts = role.lower().split('_')
        for part in role_parts:
            for ch_low, ch_orig in ch_lower.items():
                if part in ch_low and len(part) > 2:
                    try:
                        sig = mdf.get(ch_orig)
                        return pd.Series(sig.samples, index=pd.to_datetime(sig.timestamps, unit="s"))
                    except Exception:
                        continue
        
    except Exception:
        pass
    
    return None


# ==================== Control System Analysis ====================

def calculate_response_metrics(actual: pd.Series, target: pd.Series, 
                              tolerance: float = 0.02) -> Dict[str, Any]:
    """
    Calculate advanced control system metrics:
    - Rise time (10% to 90% of target)
    - Settling time (within tolerance band)
    - Overshoot percentage
    - Steady-state error
    - Peak error
    - Control quality index
    
    Args:
        actual: Actual speed series
        target: Target/set speed series
        tolerance: Tolerance band for settling (fraction of target)
        
    Returns:
        Dictionary with metrics
    """
    if len(actual) < 10 or len(target) < 10:
        return {}
    
    # Align series (handle different lengths)
    common_idx = actual.index.intersection(target.index)
    if len(common_idx) < 10:
        return {}
    
    actual_aligned = actual.loc[common_idx]
    target_aligned = target.loc[common_idx]
    
    # Calculate error
    error = actual_aligned - target_aligned
    error_pct = (error / (target_aligned + 1e-6)) * 100
    
    metrics = {
        "mean_error": float(error.mean()),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "max_error": float(error.abs().max()),
        "max_error_pct": float(error_pct.abs().max()),
        "mean_absolute_error": float(error.abs().mean()),
        "mean_error_pct": float(error_pct.mean()),
    }
    
    # Steady-state error (last 20% of data)
    if len(error) > 10:
        steady_start = int(len(error) * 0.8)
        steady_error = error.iloc[steady_start:]
        metrics["steady_state_error"] = float(steady_error.mean())
        metrics["steady_state_error_pct"] = float((steady_error / (target_aligned.iloc[steady_start:] + 1e-6)).mean() * 100)
    
    # Peak overshoot
    positive_errors = error[error > 0]
    if len(positive_errors) > 0:
        max_overshoot = positive_errors.max()
        metrics["max_overshoot"] = float(max_overshoot)
        metrics["overshoot_pct"] = float((max_overshoot / (target_aligned.loc[positive_errors.idxmax()] + 1e-6)) * 100)
    
    # Find step response (when target changes significantly)
    target_diff = target_aligned.diff().abs()
    significant_changes = target_diff[target_diff > 2.0]  # > 2 km/h change
    
    if len(significant_changes) > 0:
        # Analyze first significant change
        step_time = significant_changes.index[0]
        step_idx = common_idx.get_loc(step_time)
        
        if step_idx < len(actual_aligned) - 20:
            # Rise time (10% to 90%)
            target_value = target_aligned.iloc[step_idx + 1]
            if target_value > 0:
                step_response = actual_aligned.iloc[step_idx:]
                step_target = target_aligned.iloc[step_idx:]
                
                initial_value = actual_aligned.iloc[step_idx]
                final_value = target_value
                delta_target = final_value - initial_value
                
                if delta_target > 1.0:  # Only for significant steps
                    # Find 10% and 90% points
                    threshold_10 = initial_value + 0.1 * delta_target
                    threshold_90 = initial_value + 0.9 * delta_target
                    
                    idx_10 = None
                    idx_90 = None
                    
                    for i, val in enumerate(step_response):
                        if idx_10 is None and val >= threshold_10:
                            idx_10 = i
                        if val >= threshold_90:
                            idx_90 = i
                            break
                    
                    if idx_10 is not None and idx_90 is not None:
                        time_10 = step_response.index[idx_10]
                        time_90 = step_response.index[idx_90]
                        time_diff = (time_90 - time_10)
                        if hasattr(time_diff, 'total_seconds'):
                            metrics["rise_time"] = float(time_diff.total_seconds())
                        else:
                            metrics["rise_time"] = float(time_diff)
                        
                    # Settling time (within tolerance band)
                    tolerance_band = tolerance * target_value
                    settled = False
                    for i in range(max(10, idx_90) if idx_90 else 10, len(step_response)):
                        val = step_response.iloc[i]
                        if abs(val - target_value) <= tolerance_band:
                            # Check if stays within band for next N samples
                            window = min(20, len(step_response) - i)
                            if all(abs(step_response.iloc[i+j] - target_value) <= tolerance_band 
                                   for j in range(min(10, window))):
                                settled = True
                                settle_idx = i
                                break
                    
                    if settled:
                        settle_time = step_response.index[settle_idx]
                        settle_diff = settle_time - step_response.index[0]
                        if hasattr(settle_diff, 'total_seconds'):
                            metrics["settling_time"] = float(settle_diff.total_seconds())
                        else:
                            metrics["settling_time"] = float(settle_diff)
    
    # Control quality index (0-100, higher is better)
    rmse_pct = metrics.get("rmse", 0) / (target_aligned.mean() + 1e-6) * 100
    overshoot_penalty = metrics.get("overshoot_pct", 0) * 0.5
    quality_score = max(0, 100 - rmse_pct - overshoot_penalty - abs(metrics.get("steady_state_error_pct", 0)))
    metrics["control_quality_index"] = float(quality_score)
    
    return metrics


def analyze_pid_characteristics(error: pd.Series, derivative: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Analyze PID control characteristics from error signal.
    
    Args:
        error: Control error series
        derivative: Optional derivative signal
        
    Returns:
        PID analysis results
    """
    if len(error) < 10:
        return {}
    
    analysis = {
        "error_mean": float(error.mean()),
        "error_std": float(error.std()),
        "error_variance": float(error.var()),
        "integral_error": float(error.sum()),
        "abs_integral_error": float(error.abs().sum()),
    }
    
    # Oscillation detection
    if len(error) > 20:
        # Detect oscillations (sign changes)
        sign_changes = (error.diff().fillna(0) != 0).sum()
        analysis["oscillations"] = int(sign_changes)
        analysis["oscillation_rate"] = float(sign_changes / len(error))
        
        # Dominant frequency (if oscillating)
        if sign_changes > len(error) * 0.1:  # Significant oscillations
            fft = np.fft.fft(error.values)
            freqs = np.fft.fftfreq(len(error))
            dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
            analysis["dominant_frequency"] = float(freqs[dominant_freq_idx])
    
    # Derivative analysis if provided
    if derivative is not None and len(derivative) > 0:
        analysis["derivative_mean"] = float(derivative.mean())
        analysis["derivative_std"] = float(derivative.std())
        analysis["max_derivative"] = float(derivative.abs().max())
    
    return analysis


# ==================== Correlation Analysis ====================

def analyze_correlation(actual: pd.Series, target: pd.Series, 
                       throttle: Optional[pd.Series] = None,
                       brake: Optional[pd.Series] = None,
                       gear: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Analyze correlations between speed control and other signals.
    
    Args:
        actual: Actual vehicle speed
        target: Target/set speed
        throttle: Throttle position signal
        brake: Brake signal
        gear: Gear position signal
        
    Returns:
        Correlation analysis results
    """
    correlations = {}
    
    # Find common timebase
    all_signals = {"actual": actual, "target": target}
    if throttle is not None:
        all_signals["throttle"] = throttle
    if brake is not None:
        all_signals["brake"] = brake
    if gear is not None:
        all_signals["gear"] = gear
    
    # Align all signals
    common_idx = None
    for sig in all_signals.values():
        if sig is not None:
            if common_idx is None:
                common_idx = sig.index
            else:
                common_idx = common_idx.intersection(sig.index)
    
    if common_idx is None or len(common_idx) < 10:
        return correlations
    
    # Create aligned dataframe
    df_aligned = pd.DataFrame(index=common_idx)
    for name, sig in all_signals.items():
        if sig is not None:
            df_aligned[name] = sig.loc[common_idx]
    
    # Calculate error
    if "actual" in df_aligned and "target" in df_aligned:
        df_aligned["error"] = df_aligned["actual"] - df_aligned["target"]
    
    # Correlations
    if "error" in df_aligned:
        if "throttle" in df_aligned:
            corr = df_aligned["error"].corr(df_aligned["throttle"])
            if not np.isnan(corr):
                correlations["error_throttle_correlation"] = float(corr)
        
        if "brake" in df_aligned:
            corr = df_aligned["error"].corr(df_aligned["brake"])
            if not np.isnan(corr):
                correlations["error_brake_correlation"] = float(corr)
        
        if "gear" in df_aligned:
            # Correlation with gear
            corr = df_aligned["error"].corr(df_aligned["gear"])
            if not np.isnan(corr):
                correlations["error_gear_correlation"] = float(corr)
            
            # Analyze error by gear
            gear_stats = df_aligned.groupby("gear")["error"].agg(["mean", "std", "count"])
            correlations["error_by_gear"] = gear_stats.to_dict("index")
    
    # Throttle usage during control
    if "throttle" in df_aligned:
        correlations["throttle_mean"] = float(df_aligned["throttle"].mean())
        correlations["throttle_std"] = float(df_aligned["throttle"].std())
        correlations["throttle_max"] = float(df_aligned["throttle"].max())
    
    return correlations


# ==================== Enhanced Overshoot Detection ====================

def detect_overshoot_enhanced(df: pd.DataFrame, actual_col: str, target_col: str,
                             flag_col: Optional[str] = None, 
                             threshold: float = 2.5,
                             analyze_context: bool = True) -> Tuple[List[Dict[str, Any]], pd.DatetimeIndex]:
    """
    Enhanced overshoot detection with context analysis.
    
    Args:
        df: Dataframe with signals
        actual_col: Actual speed column name
        target_col: Target speed column name
        flag_col: Optional flag column name
        threshold: Overshoot threshold in km/h
        analyze_context: Whether to analyze context around overshoots
        
    Returns:
        Tuple of (events list, overshoot indices)
    """
    events = []
    overshoot_idx = pd.DatetimeIndex([])
    
    if actual_col not in df.columns or target_col not in df.columns:
        return events, overshoot_idx
    
    # Calculate overshoots
    if flag_col and flag_col in df.columns:
        # Only check when flag indicates active
        df[flag_col] = df[flag_col].fillna(0).astype(int)
        active_mask = df[flag_col] == 5  # Assuming 5 means active
        active_df = df[active_mask]
    else:
        active_df = df
    
    if active_df.empty:
        return events, overshoot_idx
    
    # Detect overshoots
    overshoot_mask = (active_df[actual_col] > active_df[target_col] + threshold)
    overshoot_idx = active_df.index[overshoot_mask]
    
    if len(overshoot_idx) == 0:
        return events, overshoot_idx
    
    # Analyze each overshoot event
    for ts in overshoot_idx:
        event = {
            "timestamp": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
            "actual": float(active_df.loc[ts, actual_col]),
            "target": float(active_df.loc[ts, target_col]),
            "overshoot": float(active_df.loc[ts, actual_col] - active_df.loc[ts, target_col]),
            "overshoot_pct": float(((active_df.loc[ts, actual_col] - active_df.loc[ts, target_col]) / 
                                    (active_df.loc[ts, target_col] + 1e-6)) * 100)
        }
        
        if analyze_context:
            # Analyze context around overshoot (before/after)
            window_idx = df.index.get_indexer([ts], method='nearest')[0]
            window_start = max(0, window_idx - 50)
            window_end = min(len(df), window_idx + 50)
            window_df = df.iloc[window_start:window_end]
            
            if len(window_df) > 10:
                # Find when overshoot started
                window_mask = (window_df[actual_col] > window_df[target_col] + threshold)
                overshoot_window = window_df.index[window_mask]
                if len(overshoot_window) > 0:
                    event["overshoot_start"] = overshoot_window[0].isoformat() if hasattr(overshoot_window[0], 'isoformat') else str(overshoot_window[0])
                    event["overshoot_duration_samples"] = len(overshoot_window)
                
                # Throttle/Brake context if available
                for ctx_col in ["throttle", "Throttle", "brake", "Brake"]:
                    if ctx_col in window_df.columns:
                        ctx_val = window_df.loc[ts, ctx_col] if ts in window_df.index else None
                        if ctx_val is not None:
                            event[f"context_{ctx_col.lower()}"] = float(ctx_val)
        
        events.append(event)
    
    return events, overshoot_idx


def add_active_regions(fig, df, flag_col, color="rgba(0,150,200,0.2)"):
    """Shade regions where flag == 5 (active control)."""
    if flag_col not in df.columns:
        return
    
    active = df[flag_col] == 5
    if not active.any():
        return

    # Find start and end points of true regions
    active_shifted = active.shift(1, fill_value=False)
    starts = df.index[active & ~active_shifted]
    ends = df.index[~active & active_shifted]

    # Add shapes for each region
    for i in range(len(starts)):
        start_ts = starts[i]
        end_ts = ends[i] if i < len(ends) else df.index[-1]
        fig.add_vrect(x0=start_ts, x1=end_ts, fillcolor=color, opacity=0.3,
                      layer="below", line_width=0)


# ==================== Enhanced Visualization ====================

def create_enhanced_plot(df: pd.DataFrame, actual_col: str, target_col: str,
                         flag_col: Optional[str], events: List[Dict],
                         title: str, mode: str = "cruise") -> go.Figure:
    """
    Create enhanced plot with statistical overlays.
    
    Args:
        df: Dataframe with signals
        actual_col: Actual speed column
        target_col: Target speed column
        flag_col: Flag column (optional)
        events: Overshoot events
        title: Plot title
        mode: "cruise" or "limiter"
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Color scheme
    if mode == "cruise":
        actual_color = "#1f77b4"
        target_color = "#ff7f0e"
        threshold_color = "rgba(255,150,0,0.3)"
    else:
        actual_color = "#1f77b4"
        target_color = "#2ca02c"
        threshold_color = "rgba(255,150,0,0.3)"
    
    # Actual speed
    if actual_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df[actual_col], 
            name="Actual Speed",
            line=dict(color=actual_color, width=2),
            hovertemplate="Time: %{x}<br>Speed: %{y:.2f} km/h<extra></extra>"
        ))
    
    # Target/set speed
    if target_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df[target_col], 
            name="Set Speed",
            line=dict(color=target_color, dash="dash", width=2),
            hovertemplate="Time: %{x}<br>Set: %{y:.2f} km/h<extra></extra>"
        ))
        
        # Tolerance band
        threshold = 2.5
        upper_band = df[target_col] + threshold
        lower_band = df[target_col] - threshold
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=upper_band,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=lower_band,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=threshold_color,
            name=f"Tolerance Band (±{threshold} km/h)",
            hoverinfo='skip'
        ))
    
    # Overshoot markers
    if events:
        overshoot_times = [pd.to_datetime(e["timestamp"]) for e in events]
        overshoot_values = [e["actual"] for e in events]
        
        fig.add_trace(go.Scatter(
            x=overshoot_times,
            y=overshoot_values,
            mode="markers",
            name="Overshoot Events",
            marker=dict(color="red", size=10, symbol="x", line=dict(width=2)),
            hovertemplate="Time: %{x}<br>Speed: %{y:.2f} km/h<br>Overshoot: %{customdata:.2f} km/h<extra></extra>",
            customdata=[e["overshoot"] for e in events]
        ))
    
    # Active regions
    if flag_col and flag_col in df.columns:
        add_active_regions(fig, df, flag_col, 
                         color="rgba(0,200,255,0.15)" if mode == "cruise" else "rgba(255,150,0,0.15)")
    
    # Statistical overlay (mean, std bands)
    if actual_col in df.columns and target_col in df.columns:
        # Calculate rolling statistics
        window_size = min(100, len(df) // 10)
        if window_size > 5:
            actual_rolling_mean = df[actual_col].rolling(window=window_size, center=True).mean()
            actual_rolling_std = df[actual_col].rolling(window=window_size, center=True).std()
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=actual_rolling_mean + actual_rolling_std,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=actual_rolling_mean - actual_rolling_std,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor="rgba(128,128,128,0.1)",
                name="Speed Std Dev",
                hovertemplate="Mean ± Std: %{y:.2f} km/h<extra></extra>"
            ))
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=500,
        xaxis_title="Time",
        yaxis_title="Speed (km/h)",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(t=60, l=70, r=30, b=100),
        hovermode='closest'
    )
    
    return fig


# ==================== Main Compute Function ====================

def compute_ccsl(files: List[Path], overshoot_threshold: float = 2.5,
                 enable_advanced_metrics: bool = True) -> Dict[str, Any]:
    """
    Enhanced CC/SL analysis with advanced metrics and correlations.
    
    Args:
        files: List of MDF file paths
        overshoot_threshold: Overshoot threshold in km/h
        enable_advanced_metrics: Enable advanced control metrics
        
    Returns:
        Comprehensive analysis results
    """
    results = {"plots": [], "tables": {}, "meta": {}, "metrics": {}, "correlations": {}}

    if MDF is None:
        results["meta"]["error"] = "asammdf library not installed on the server."
        return results

    if not files:
        results["meta"]["error"] = "No measurement files were provided for analysis."
        return results
        
    # Process all files (enhanced to support multi-file)
    all_cruise_events = []
    all_limiter_events = []
    
    for file_idx, f in enumerate(files):
        if not f.exists():
            continue
        
        try:
            mdf = MDF(str(f))
        except Exception as e:
            results["meta"][f"file_{file_idx}_error"] = str(e)
            continue
        
        try:
            channels = list(mdf.channels_db.keys())
            
            # Enhanced signal detection
            df = pd.DataFrame()
            
            # Vehicle speed (comprehensive search)
            df["VehSpd"] = find_channel_comprehensive(mdf, "vehicle_speed", 
                ["VITESSE_VEHICULE_ROUES", "VehSpd", "Vehicle_Speed", "Ext_spdVeh", "Veh_spdVeh"])
            
            # Cruise control signals
            df["CruiseReq"] = find_channel_comprehensive(mdf, "cruise_set",
                ["Ext_spdVehVSRegReq", "Cruise_Set_Speed", "VSCtl_spdVehVSRegReq", "vsreg"])
            df["CruiseFlag"] = find_channel_comprehensive(mdf, "cruise_flag",
                ["VSCtl_stVSregExtd", "Cruise_Status", "vsreg_extd", "Cruise_Active"])
            
            # Speed limiter signals
            df["LimiterReq"] = find_channel_comprehensive(mdf, "limiter_set",
                ["Ext_spdVehVSLimReq", "Limiter_Set_Speed", "VSCtl_spdVehVSLimReq", "vslim"])
            df["LimiterFlag"] = find_channel_comprehensive(mdf, "limiter_flag",
                ["VSCtl_stVSLimExtd", "Limiter_Status", "vslim_extd", "Limiter_Active"])
            
            # Correlation signals
            df["Throttle"] = find_channel_comprehensive(mdf, "throttle", THROTTLE_CANDIDATES)
            df["Brake"] = find_channel_comprehensive(mdf, "brake", 
                ["Brake", "BrakePedal", "Brake_Switch", "BRK"])
            df["Gear"] = find_channel_comprehensive(mdf, "gear", GEAR_CANDIDATES)
            
            # Collect found signals
            found_signals = {key: val is not None for key, val in df.items()}
            
            # Drop columns with all NaN
            df = df.dropna(how="all")
            
            if df.empty:
                results["meta"][f"file_{file_idx}_error"] = "No signals found"
                continue
            
            # Interpolate and align
            df = df.interpolate(method='time').ffill().bfill()
            
            # Cruise Control Analysis
            if "VehSpd" in df.columns and "CruiseReq" in df.columns:
                cruise_events, cruise_idx = detect_overshoot_enhanced(
                    df, "VehSpd", "CruiseReq", "CruiseFlag", overshoot_threshold)
                all_cruise_events.extend([{**e, "file": f.name} for e in cruise_events])
                
                # Advanced metrics
                if enable_advanced_metrics and len(df) > 50:
                    cruise_metrics = calculate_response_metrics(
                        df["VehSpd"], df["CruiseReq"])
                    results["metrics"][f"cruise_file_{file_idx}"] = cruise_metrics
                    
                    # Correlation analysis
                    cruise_corr = analyze_correlation(
                        df["VehSpd"], df["CruiseReq"],
                        df.get("Throttle"), df.get("Brake"), df.get("Gear"))
                    if cruise_corr:
                        results["correlations"][f"cruise_file_{file_idx}"] = cruise_corr
                
                # Enhanced plot
                if file_idx == 0 or len(files) == 1:  # Plot for first file or single file
                    fig_cruise = create_enhanced_plot(
                        df, "VehSpd", "CruiseReq", "CruiseFlag",
                        cruise_events, "Cruise Control Analysis (Enhanced)", "cruise")
                    results["plots"].append({
                        "name": "Cruise Analysis Enhanced",
                        "plotly_json": fig_cruise.to_json(),
                        "type": "plotly"
                    })
            
            # Speed Limiter Analysis
            if "VehSpd" in df.columns and "LimiterReq" in df.columns:
                limiter_events, limiter_idx = detect_overshoot_enhanced(
                    df, "VehSpd", "LimiterReq", "LimiterFlag", overshoot_threshold)
                all_limiter_events.extend([{**e, "file": f.name} for e in limiter_events])
                
                # Advanced metrics
                if enable_advanced_metrics and len(df) > 50:
                    limiter_metrics = calculate_response_metrics(
                        df["VehSpd"], df["LimiterReq"])
                    results["metrics"][f"limiter_file_{file_idx}"] = limiter_metrics
                    
                    # Correlation analysis
                    limiter_corr = analyze_correlation(
                        df["VehSpd"], df["LimiterReq"],
                        df.get("Throttle"), df.get("Brake"), df.get("Gear"))
                    if limiter_corr:
                        results["correlations"][f"limiter_file_{file_idx}"] = limiter_corr
                
                # Enhanced plot
                if file_idx == 0 or len(files) == 1:
                    fig_limiter = create_enhanced_plot(
                        df, "VehSpd", "LimiterReq", "LimiterFlag",
                        limiter_events, "Speed Limiter Analysis (Enhanced)", "limiter")
                    results["plots"].append({
                        "name": "Limiter Analysis Enhanced",
                        "plotly_json": fig_limiter.to_json(),
                        "type": "plotly"
                    })
            
            # Update metadata
            if "found_signals" not in results["meta"]:
                results["meta"]["found_signals"] = found_signals
            else:
                # Merge found signals
                for k, v in found_signals.items():
                    results["meta"]["found_signals"][k] = results["meta"]["found_signals"].get(k, False) or v
            
        except Exception as e:
            results["meta"][f"file_{file_idx}_error"] = str(e)
        finally:
            try:
                mdf.close()
            except Exception:
                pass
    
    # Summary tables
    results["tables"]["Cruise Overshoot Events"] = (
        all_cruise_events if all_cruise_events 
        else [{"status": "No overshoot events detected or required signals missing."}])
    
    results["tables"]["Limiter Overshoot Events"] = (
        all_limiter_events if all_limiter_events 
        else [{"status": "No overshoot events detected or required signals missing."}])
    
    # Aggregate metrics summary
    if enable_advanced_metrics and results["metrics"]:
        summary_metrics = {}
        for key, metrics in results["metrics"].items():
            if isinstance(metrics, dict):
                for metric_name, metric_value in metrics.items():
                    if metric_name not in summary_metrics:
                        summary_metrics[metric_name] = []
                    if isinstance(metric_value, (int, float)):
                        summary_metrics[metric_name].append(metric_value)
        
        # Calculate aggregates
        aggregated = {}
        for metric_name, values in summary_metrics.items():
            if values:
                aggregated[f"{metric_name}_mean"] = float(np.mean(values))
                aggregated[f"{metric_name}_max"] = float(np.max(values))
                aggregated[f"{metric_name}_min"] = float(np.min(values))
        
        if aggregated:
            results["tables"]["Control Metrics Summary"] = [aggregated]

    return results


# ==================== CLI ====================

def main():
    import argparse
    p = argparse.ArgumentParser(description="Advanced Cruise & Limiter Analyzer")
    p.add_argument("files", nargs='+', help="Path to MDF file(s)")
    p.add_argument("--threshold", type=float, default=2.5, help="Overshoot threshold in km/h")
    p.add_argument("--no-advanced", action="store_true", help="Disable advanced metrics")
    args = p.parse_args()

    files = [Path(f) for f in args.files]
    payload = compute_ccsl(files, overshoot_threshold=args.threshold,
                          enable_advanced_metrics=not args.no_advanced)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    sys.exit(main() if 'main' in dir() else 0)
