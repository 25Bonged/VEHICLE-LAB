#!/usr/bin/env python3
"""
Advanced Gear Hunting Detection System
Detects gear hunting behavior using multiple signals and pattern recognition algorithms.
Supports multiple OEM data formats with comprehensive signal mapping dictionary.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Import misfire detection for advanced gear plot
try:
    from custom_misfire import compute_misfire
    MISFIRE_AVAILABLE = True
except ImportError:
    MISFIRE_AVAILABLE = False
    logger.warning("Misfire detection not available for gear plot integration")

try:
    from asammdf import MDF
except ImportError:
    MDF = None
    logger.warning("asammdf not available. Gear hunting analysis will be limited.")

# Comprehensive signal mapping dictionary for multiple OEM data sources
# Similar to REQUIRED_SIGNALS in custom_map.py
GEAR_SIGNALS: Dict[str, List[str]] = {
    "gear": [
        # Common gear signal names across OEMs
        "Gear", "TrnsGr", "VSCtl_noGear", "Gearbox_Gear", "iGear", "GearSelected",
        "GearCurrent", "GearAct", "ActualGear", "GearPos", "nGear", "Gear_Act",
        "TransmissionGear", "TCU_Gear", "Gear_Actual", "GearValue", "CurrentGear",
        "GearPosition", "GearPos_Act", "TransGear", "Gearbox_Gear_Act",
        "ECM_Gear", "TCM_Gear", "TCM_GearActual", "Gear_Ratio", "GearRatio",
        "GearSel", "Gear_Cmd", "Gear_Des", "TargetGear", "GearRequested",
        # VW/Audi variants
        "Gang_Position", "Gang", "Gear_Gang",
        # BMW variants
        "GearboxGear", "GetriebeGang",
        # Ford variants
        "Gear_Trans", "Transmission_Gear",
        # GM variants
        "Gear_Actual", "TransGearPosition",
        # Japanese OEM variants
        "GEAR", "GearValue", "TransmissionGearPosition"
    ],
    "speed": [
        # Vehicle speed signals
        "Veh_spdVeh", "Ext_spdVeh", "Vehicle_Speed", "VehSpd", "v", "VehicleSpeed",
        "Speed", "VehSpeed", "SpdVeh", "vVeh", "VSS", "VehicleSpd", "Vehicle_Speed_Act",
        "VehSpd_Act", "VehicleSpeedActual", "SpdVeh_Act", "VehSpdV", "VSpeed",
        # VW/Audi variants
        "v_Fahrzeug", "Fahrzeuggeschwindigkeit", "v_Speed",
        # BMW variants
        "Fahrzeuggeschwindigkeit", "vFahrzeug",
        # Ford variants
        "Vehicle_Speed_Actual", "Veh_Speed",
        # GM variants
        "VehicleSpeedVSS", "SpeedVSS",
        # Japanese OEM variants
        "SPEED", "VehicleSpeedValue", "SpeedValue"
    ],
    "throttle": [
        # Throttle/accelerator pedal position signals
        "PedalPos", "Throttle", "ThrottlePos", "ThrottlePosition", "AccPedalPos",
        "AccPedal", "Pedal_Pos", "ThrottleVal", "EngineThrottle", "ThrottleAngle",
        "APP", "TPS", "ThrottleAct", "AccPedalPos_Act", "AcceleratorPedal",
        "AccPedalPos_Des", "PedalPosition", "ThrottlePos_Act", "ThrottlePos_Des",
        # VW/Audi variants
        "Pedalwert", "Gaspedalstellung", "AcceleratorPedalPosition",
        # BMW variants
        "Gaspedal", "AcceleratorPosition",
        # Ford variants
        "Accelerator_Pedal", "APP_Sensor",
        # GM variants
        "AcceleratorPedalPos", "ThrottlePosition_Actual",
        # Japanese OEM variants
        "THROTTLE", "AccPedalValue", "ThrottleOpening"
    ],
    "rpm": [
        # Engine RPM/speed signals
        "nEng", "EngineSpeed", "Engine_RPM", "rpm", "EngineSpd", "n_Eng",
        "RPM", "EngineSpeedActual", "EngSpeed", "nEng_Actual", "Epm_nEng_RTE",
        "EngineRPM", "Epm_nEng", "Ext_nEng_RTE", "inRpm", "nEng_RTE",
        "Engine_Speed", "EngineSpeed_RTE", "nEng_Raw", "EngineSpeed_Act",
        # VW/Audi variants
        "Motordrehzahl", "Drehzahl", "n_Motor",
        # BMW variants
        "Motordrehzahl", "nMotor",
        # Ford variants
        "Engine_Speed_Actual", "RPM_Engine",
        # GM variants
        "EngineSpeedValue", "RPMValue",
        # Japanese OEM variants
        "ENGINE_SPEED", "RPM_Value", "EngineSpeedValue"
    ],
    "torque": [
        # Engine torque signals
        "Tq", "Torque", "EngineTorque", "Trq", "tqEng", "EngineTq", "TorqueAct",
        "Engine_Torque", "Trq_Ext", "TqSys_tqCkEngReal", "EngineLoad", "Load",
        "TqSys_tqCkEngReal_RTE", "EngineTorqueActual", "Torque_Act", "TqEng",
        "Engine_Torque_Actual", "TorqueValue", "EngineTorqueValue",
        # VW/Audi variants
        "Motormoment", "Drehmoment",
        # BMW variants
        "Motormoment", "TorqueMotor",
        # Ford variants
        "Engine_Torque_Actual", "TorqueActual",
        # GM variants
        "EngineTorqueValue", "TorqueValue",
        # Japanese OEM variants
        "TORQUE", "EngineTorqueValue"
    ],
    "tc_lockup": [
        # Torque converter lockup/clutch signals
        "TCLockup", "TCLock", "TorqueConvLock", "TC_Status", "LockupStatus",
        "ConverterLock", "TCLockupStatus", "TCCLockup", "TCC_Lockup",
        "TorqueConverterLockup", "TC_LockupStatus", "LockupClutchStatus",
        "TCC_Status", "TC_Lockup", "TorqueConverterStatus",
        # VW/Audi variants
        "Wandlerkupplung", "Kupplung_Status",
        # BMW variants
        "Wandlerkupplung",
        # Ford/GM variants
        "TCC_Lockup_Status", "TorqueConverterLockupStatus",
        # Japanese OEM variants
        "TCC_LOCKUP", "TorqueConverterLockupValue"
    ]
}

# Critical signals required for gear hunting analysis
CRITICAL_GEAR_SIGNALS = {"gear"}

# Optional but recommended signals
OPTIONAL_GEAR_SIGNALS = {"speed", "throttle", "rpm", "torque", "tc_lockup"}


def find_signal(channels: List[str], signal_role: str, overrides: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Robust signal detection function similar to custom_map.py's find_signal.
    Searches for signals using multiple matching strategies to handle various OEM naming conventions.
    
    Args:
        channels: List of available channel names from MDF file
        signal_role: Signal role to find (e.g., "gear", "speed", "rpm")
        overrides: Optional dictionary of manual signal mappings
        
    Returns:
        Matched channel name or None if not found
    """
    if overrides and signal_role in overrides:
        override_ch = overrides[signal_role]
        if override_ch in channels:
            logger.info(f"Using override for {signal_role}: {override_ch}")
            return override_ch
    
    candidates = GEAR_SIGNALS.get(signal_role, [])
    if not candidates:
        logger.warning(f"No candidates defined for signal role: {signal_role}")
        return None
    
    # Create case-insensitive mapping
    channels_lower = {ch.lower(): ch for ch in channels}
    
    # Strategy 1: Exact match (case-insensitive)
    for candidate in candidates:
        cand_lower = candidate.lower()
        if cand_lower in channels_lower:
            matched = channels_lower[cand_lower]
            logger.info(f"Found {signal_role} signal via exact match: {matched}")
            return matched
    
    # Strategy 2: Substring match (candidate in channel name or vice versa)
    for candidate in candidates:
        cand_lower = candidate.lower()
        for ch_lower, ch_original in channels_lower.items():
            if cand_lower in ch_lower or ch_lower in cand_lower:
                logger.info(f"Found {signal_role} signal via substring match: {ch_original} (candidate: {candidate})")
                return ch_original
    
    # Strategy 3: Key part fuzzy matching (e.g., "gear" in "GearboxGearActual")
    key_parts = {
        "gear": ["gear", "gang", "transmission", "tcu", "tcm"],
        "speed": ["speed", "spd", "vss", "velocity", "geschwindigkeit"],
        "throttle": ["throttle", "pedal", "accel", "app", "tps", "gaspedal"],
        "rpm": ["rpm", "neng", "engspeed", "drehzahl", "speed", "n_"],
        "torque": ["torque", "tq", "trq", "moment", "drehmoment"],
        "tc_lockup": ["lockup", "tcc", "tclock", "kupplung", "converter"]
    }
    
    if signal_role in key_parts:
        for key_part in key_parts[signal_role]:
            for ch_lower, ch_original in channels_lower.items():
                if key_part in ch_lower and len(ch_lower) > 3:  # Avoid too short matches
                    logger.info(f"Found {signal_role} signal via fuzzy match: {ch_original} (key: {key_part})")
                    return ch_original
    
    logger.warning(f"Could not find {signal_role} signal. Candidates tried: {candidates[:10]}...")
    return None


def detect_oscillating_pattern(gear_series: pd.Series, window_size: int = 10, 
                              threshold: int = 3) -> pd.Series:
    """
    Detect oscillating gear patterns using rolling window analysis.
    
    Returns: Boolean series indicating oscillating periods
    """
    gear_diff = gear_series.diff().abs()
    
    # Calculate rolling variance and shift frequency
    rolling_variance = gear_diff.rolling(window=window_size, center=True).var()
    shift_frequency = gear_diff.rolling(window=window_size, center=True).sum()
    
    # Detect oscillations: high variance + high shift frequency
    oscillating = (rolling_variance > np.percentile(rolling_variance.dropna(), 75)) & \
                  (shift_frequency >= threshold)
    
    return oscillating


def detect_rapid_shifts(gear_series: pd.Series, time_series: pd.Series, 
                       max_interval: float = 2.0) -> pd.Series:
    """
    Detect rapid gear shifts within short time intervals.
    
    Returns: Boolean series indicating rapid shift events
    """
    gear_changes = gear_series.diff().abs() > 0
    time_diff = time_series.diff()
    
    # Rapid shifts: gear change within max_interval seconds
    rapid = gear_changes & (time_diff <= max_interval)
    
    return rapid


def detect_inefficient_shifts(gear_series: pd.Series, speed_series: Optional[pd.Series],
                              throttle_series: Optional[pd.Series],
                              min_speed_change: float = 2.0) -> pd.Series:
    """
    Detect gear shifts that don't correlate with significant speed/throttle changes.
    Indicates inefficient or unnecessary shifting.
    """
    gear_changes = gear_series.diff().abs() > 0
    
    if speed_series is not None:
        speed_diff = speed_series.diff().abs()
        speed_correlation = ~(gear_changes & (speed_diff < min_speed_change))
    else:
        speed_correlation = pd.Series(True, index=gear_series.index)
    
    if throttle_series is not None:
        throttle_diff = throttle_series.diff().abs()
        throttle_change = throttle_diff > 0.05  # 5% throttle change threshold
        throttle_correlation = gear_changes & throttle_change
    else:
        throttle_correlation = pd.Series(True, index=gear_series.index)
    
    # Inefficient: gear change without significant speed/throttle change
    inefficient = gear_changes & (~speed_correlation) & (~throttle_correlation)
    
    return inefficient


def calculate_shift_severity(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate severity score for each hunting event.
    Score based on:
    - Shift frequency
    - Speed range
    - Throttle variation
    - Time duration
    """
    if events_df.empty:
        return events_df
    
    events_df = events_df.copy()
    
    # Calculate time duration of hunting sequence
    events_df['duration'] = events_df['time_end'] - events_df['time_start']
    
    # Shift frequency (shifts per second)
    events_df['shift_rate'] = events_df['shift_count'] / (events_df['duration'] + 0.1)
    
    # Severity score (0-100)
    # Higher = more severe
    severity = (
        (events_df['shift_rate'] / 5.0).clip(0, 1) * 40 +  # Shift rate component (max 40)
        (events_df['shift_count'] / 10.0).clip(0, 1) * 30 +  # Shift count component (max 30)
        (1 - events_df.get('avg_throttle', pd.Series(0)) / 1.0).clip(0, 1) * 20 +  # Low throttle = inefficient (max 20)
        (events_df.get('gear_range', pd.Series(0)) / 4.0).clip(0, 1) * 10  # Gear range component (max 10)
    )
    
    events_df['severity'] = severity.clip(0, 100).round(1)
    
    # Classify severity
    events_df['severity_class'] = pd.cut(
        events_df['severity'],
        bins=[0, 30, 60, 80, 100],
        labels=['Low', 'Medium', 'High', 'Critical']
    )
    
    return events_df


def analyze_gear_hunting(files: List[Path], overrides: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Advanced gear hunting analysis with multi-signal correlation and pattern recognition.
    Uses comprehensive signal mapping dictionary to support multiple OEM data formats.
    
    Args:
        files: List of MDF file paths to analyze
        overrides: Optional manual signal name mappings (e.g., {"gear": "CustomGearSignal"})
    
    Returns comprehensive analysis including:
    - Hunting event detection
    - Severity scoring
    - Statistical summaries
    - Diagnostic flags
    - Rich visualizations
    """
    if not MDF:
        return {
            "tables": {
                "Hunting Events": [],
                "File Summary": [],
                "Statistics": []
            },
            "plots": {},
            "meta": {
                "ok": False,
                "error": "asammdf library not installed"
            }
        }
    
    all_hunting_events = []
    file_summaries = []
    all_data_frames = []
    signal_mapping_report = []
    global_stats = {
        "total_files": len(files),
        "files_processed": 0,
        "total_hunting_events": 0,
        "total_shifts": 0,
        "avg_shift_rate": 0.0,
        "max_severity": 0.0
    }
    
    for file_path in files:
        file_name = file_path.name
        try:
            with MDF(str(file_path)) as mdf:
                # Get all available channels
                all_channels = list(mdf.channels_db.keys())
                logger.info(f"Processing {file_name}: Found {len(all_channels)} channels")
                
                # Find all signals using robust detection
                signal_map = {}
                for signal_role in list(CRITICAL_GEAR_SIGNALS) + list(OPTIONAL_GEAR_SIGNALS):
                    matched_ch = find_signal(all_channels, signal_role, overrides)
                    if matched_ch:
                        signal_map[signal_role] = matched_ch
                        signal_mapping_report.append({
                            "file": file_name,
                            "signal_role": signal_role,
                            "channel_name": matched_ch,
                            "found": True
                        })
                
                # Check for critical gear signal
                if "gear" not in signal_map:
                    logger.warning(f"No gear signal found in {file_name}")
                    available_channels_preview = ", ".join(all_channels[:20])
                    if len(all_channels) > 20:
                        available_channels_preview += f"... ({len(all_channels)} total)"
                    
                    file_summaries.append({
                        "file": file_name,
                        "status": "No gear channel found",
                        "hunting_events": 0,
                        "signals_found": "None",
                        "available_channels": available_channels_preview
                    })
                    continue
                
                gear_ch = signal_map["gear"]
                logger.info(f"Using gear signal: {gear_ch}")
                
                # Extract gear signal
                try:
                    gear_sig = mdf.get(gear_ch)
                except Exception as e:
                    logger.error(f"Error extracting gear signal {gear_ch}: {e}")
                    file_summaries.append({
                        "file": file_name,
                        "status": f"Error extracting gear signal: {e}",
                        "hunting_events": 0
                    })
                    continue
                
                if gear_sig.samples.size == 0:
                    file_summaries.append({
                        "file": file_name,
                        "status": "Empty gear signal",
                        "hunting_events": 0
                    })
                    continue
                
                # Create base dataframe with gear signal
                df = pd.DataFrame({
                    'time': gear_sig.timestamps,
                    'gear': pd.to_numeric(gear_sig.samples, errors='coerce').round()
                })
                df = df.dropna(subset=['gear'])
                if df.empty:
                    file_summaries.append({
                        "file": file_name,
                        "status": "No valid gear data after processing",
                        "hunting_events": 0
                    })
                    continue
                    
                df['gear'] = df['gear'].astype(int)
                
                # Interpolate other signals to gear timestamps
                signals_found = {'gear': gear_ch}
                
                for signal_role in ["speed", "throttle", "rpm", "torque", "tc_lockup"]:
                    if signal_role in signal_map:
                        ch_name = signal_map[signal_role]
                        try:
                            sig = mdf.get(ch_name)
                            # Interpolate to gear signal timestamps
                            interp_sig = sig.interp(gear_sig.timestamps)
                            df[signal_role] = pd.to_numeric(interp_sig.samples, errors='coerce')
                            signals_found[signal_role] = ch_name
                            logger.debug(f"Successfully extracted {signal_role} signal: {ch_name}")
                        except Exception as e:
                            logger.warning(f"Failed to extract {signal_role} signal {ch_name}: {e}")
                            df[signal_role] = np.nan
                    else:
                        df[signal_role] = np.nan
                
                # Detect gear changes
                df['gear_change'] = df['gear'].diff().abs() > 0
                df['gear_prev'] = df['gear'].shift(1)
                df['gear_next'] = df['gear'].shift(-1)
                df['time_diff'] = df['time'].diff()
                
                # Apply detection algorithms
                df['oscillating'] = detect_oscillating_pattern(df['gear'])
                df['rapid_shifts'] = detect_rapid_shifts(df['gear'], df['time'])
                df['inefficient'] = detect_inefficient_shifts(
                    df['gear'],
                    df.get('speed') if 'speed' in df.columns else None,
                    df.get('throttle') if 'throttle' in df.columns else None
                )
                
                # Combined hunting detection
                df['hunting'] = df['oscillating'] | df['rapid_shifts'] | df['inefficient']
                
                # Group hunting events into sequences
                prev_hunting = df['hunting'].shift(1)
                # Use explicit bool conversion to avoid FutureWarning
                prev_hunting = prev_hunting.astype('boolean').fillna(False)
                hunting_start = df['hunting'].astype('boolean') & (~prev_hunting)
                df['hunting_group'] = hunting_start.astype(int).cumsum()
                
                # Analyze hunting sequences
                hunting_groups = df[df['hunting']].groupby('hunting_group')
                
                hunting_events_in_file = 0
                for group_id, group_df in hunting_groups:
                    if len(group_df) < 2:  # Need at least 2 points for a sequence
                        continue
                    
                    # Calculate sequence statistics
                    time_start = group_df['time'].min()
                    time_end = group_df['time'].max()
                    duration = time_end - time_start
                    shift_count = int(group_df['gear_change'].sum())
                    
                    # Gear range during hunting
                    gear_min = int(group_df['gear'].min())
                    gear_max = int(group_df['gear'].max())
                    gear_range = gear_max - gear_min
                    gear_list = sorted(group_df['gear'].unique().tolist())
                    
                    # Speed statistics
                    avg_speed = group_df['speed'].mean() if 'speed' in group_df.columns and group_df['speed'].notna().any() else None
                    speed_range = (group_df['speed'].max() - group_df['speed'].min()) if 'speed' in group_df.columns and group_df['speed'].notna().any() else None
                    
                    # Throttle statistics
                    avg_throttle = group_df['throttle'].mean() if 'throttle' in group_df.columns and group_df['throttle'].notna().any() else None
                    throttle_range = (group_df['throttle'].max() - group_df['throttle'].min()) if 'throttle' in group_df.columns and group_df['throttle'].notna().any() else None
                    
                    # RPM statistics
                    avg_rpm = group_df['rpm'].mean() if 'rpm' in group_df.columns and group_df['rpm'].notna().any() else None
                    
                    # Torque statistics
                    avg_torque = group_df['torque'].mean() if 'torque' in group_df.columns and group_df['torque'].notna().any() else None
                    
                    # Create event record
                    event = {
                        "file": file_name,
                        "time_start": round(time_start, 2),
                        "time_end": round(time_end, 2),
                        "duration": round(duration, 2),
                        "shift_count": shift_count,
                        "shift_rate": round(shift_count / (duration + 0.1), 2),
                        "gear_range": gear_range,
                        "gear_min": gear_min,
                        "gear_max": gear_max,
                        "gears_involved": str(gear_list),
                        "avg_speed": round(avg_speed, 1) if avg_speed is not None and not np.isnan(avg_speed) else None,
                        "speed_range": round(speed_range, 1) if speed_range is not None and not np.isnan(speed_range) else None,
                        "avg_throttle": round(avg_throttle, 3) if avg_throttle is not None and not np.isnan(avg_throttle) else None,
                        "throttle_range": round(throttle_range, 3) if throttle_range is not None and not np.isnan(throttle_range) else None,
                        "avg_rpm": round(avg_rpm, 0) if avg_rpm is not None and not np.isnan(avg_rpm) else None,
                        "avg_torque": round(avg_torque, 1) if avg_torque is not None and not np.isnan(avg_torque) else None,
                        "detection_type": []
                    }
                    
                    # Identify detection type
                    if group_df['oscillating'].any():
                        event["detection_type"].append("Oscillating")
                    if group_df['rapid_shifts'].any():
                        event["detection_type"].append("Rapid Shifts")
                    if group_df['inefficient'].any():
                        event["detection_type"].append("Inefficient")
                    
                    event["detection_type"] = ", ".join(event["detection_type"]) if event["detection_type"] else "Pattern"
                    
                    all_hunting_events.append(event)
                    hunting_events_in_file += 1
                
                # Calculate file statistics
                total_shifts = int(df['gear_change'].sum())
                total_time = df['time'].max() - df['time'].min() if len(df) > 0 else 1
                
                file_summaries.append({
                    "file": file_name,
                    "status": "OK",
                    "total_shifts": total_shifts,
                    "hunting_events": hunting_events_in_file,
                    "hunting_rate": round(hunting_events_in_file / total_time * 3600, 2) if total_time > 0 else 0,
                    "signals_found": ", ".join(signals_found.keys()),
                    "gear_signal": gear_ch
                })
                
                df['file'] = file_name
                all_data_frames.append(df)
                
                global_stats["files_processed"] += 1
                global_stats["total_shifts"] += total_shifts
                global_stats["total_hunting_events"] += hunting_events_in_file
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
            file_summaries.append({
                "file": file_name,
                "status": f"Error: {str(e)[:100]}",
                "hunting_events": 0
            })
    
    # Calculate severity for all events
    if all_hunting_events:
        events_df = pd.DataFrame(all_hunting_events)
        events_df = calculate_shift_severity(events_df)
        all_hunting_events = events_df.to_dict('records')
        global_stats["max_severity"] = float(events_df['severity'].max()) if 'severity' in events_df.columns else 0.0
    
    # Create visualizations
    plots = {}
    
    if all_data_frames:
        full_df = pd.concat(all_data_frames, ignore_index=True)
        
        # Ensure time and gear columns exist and have valid data
        if 'time' not in full_df.columns or 'gear' not in full_df.columns:
            logger.warning("Missing required columns (time or gear) for plotting")
        else:
            # Normalize time to start from 0 for each file (better visualization)
            for file_name in full_df['file'].unique():
                file_mask = full_df['file'] == file_name
                file_data = full_df.loc[file_mask, 'time']
                if len(file_data) > 0:
                    file_min_time = file_data.min()
                    if not np.isnan(file_min_time) and file_min_time != 0:
                        full_df.loc[file_mask, 'time'] = full_df.loc[file_mask, 'time'] - file_min_time
            
            logger.info(f"Creating plots from dataframe with {len(full_df)} rows")
            logger.info(f"Dataframe columns: {full_df.columns.tolist()}")
            if len(full_df) > 0:
                logger.info(f"Time range: {full_df['time'].min():.2f} to {full_df['time'].max():.2f}")
                logger.info(f"Gear value range: {full_df['gear'].min()} to {full_df['gear'].max()}")
            else:
                logger.warning("Dataframe is empty after concatenation")
                full_df = None  # Mark as invalid for plotting
            
            # Multi-panel gear analysis plot (3 rows: Gear, Speed+RPM, Throttle)
            if full_df is not None and len(full_df) > 0:
                logger.info(f"[GEAR PLOT] 🔵 ENTERING PLOT CREATION BLOCK - full_df has {len(full_df)} rows")
                try:
                    # Get misfire events for advanced plot (optional - won't fail if unavailable)
                    misfire_events = []
                    if MISFIRE_AVAILABLE:
                        try:
                            logger.info(f"[GEAR PLOT] Detecting misfire events for advanced visualization...")
                            misfire_result = compute_misfire(files, include_plots=False)
                            if misfire_result and 'tables' in misfire_result:
                                # Extract misfire events from result
                                misfire_events_list = misfire_result.get('tables', {}).get('Misfire Events', [])
                                if misfire_events_list:
                                    # Filter events that match current file's time range
                                    time_min = full_df['time'].min()
                                    time_max = full_df['time'].max()
                                    misfire_events = [
                                        evt for evt in misfire_events_list 
                                        if isinstance(evt, dict) and 'time' in evt 
                                        and time_min <= evt['time'] <= time_max
                                    ]
                                    logger.info(f"[GEAR PLOT] Found {len(misfire_events)} misfire events in time range [{time_min:.2f}, {time_max:.2f}]")
                        except Exception as misfire_err:
                            logger.warning(f"[GEAR PLOT] Could not load misfire events: {misfire_err}")
                    
                    # Check if we have speed or RPM data for the combined panel
                    has_speed = 'speed' in full_df.columns and full_df['speed'].notna().any()
                    has_rpm = 'rpm' in full_df.columns and full_df['rpm'].notna().any()
                    logger.info(f"[GEAR PLOT] Data availability: has_speed={has_speed}, has_rpm={has_rpm}, misfire_events={len(misfire_events)}")
                    
                    # Determine subplot titles
                    subplot_titles_list = ['Gear Position Over Time']
                    if has_speed and has_rpm:
                        subplot_titles_list.append('Speed & RPM')
                    elif has_speed:
                        subplot_titles_list.append('Vehicle Speed')
                    elif has_rpm:
                        subplot_titles_list.append('Engine RPM')
                    else:
                        subplot_titles_list.append('Speed & RPM')  # Default title
                    
                    subplot_titles_list.append('Throttle Position')
                    
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=subplot_titles_list,
                        vertical_spacing=0.08,
                        shared_xaxes=True,
                        specs=[[{}], [{}], [{}]]  # All regular subplots
                    )
                    
                    # Secondary y-axis for RPM will be configured after adding traces
                    
                    # Gear plot - always plot gear data
                    for file_name in full_df['file'].unique():
                        file_df = full_df[full_df['file'] == file_name].copy()
                        # Filter out NaN values for gear plot
                        gear_valid = file_df[['time', 'gear']].dropna()
                        logger.debug(f"File {file_name}: {len(gear_valid)} valid gear data points out of {len(file_df)} total")
                        if len(gear_valid) > 0:
                            # Convert to numpy arrays then to lists for proper serialization
                            time_vals = np.asarray(gear_valid['time'].values, dtype=np.float64).tolist()
                            gear_vals = np.asarray(gear_valid['gear'].values, dtype=np.float64).tolist()
                            logger.debug(f"Plotting gear: {len(time_vals)} points, time range [{min(time_vals):.2f}, {max(time_vals):.2f}], gear range [{min(gear_vals):.0f}, {max(gear_vals):.0f}]")
                            if len(time_vals) > 0 and len(gear_vals) > 0:
                                fig.add_trace(
                                    go.Scatter(
                                        x=time_vals,
                                        y=gear_vals,
                                        mode='lines+markers',
                                        name='Gear',
                                        line=dict(width=2),
                                        marker=dict(size=4),
                                        showlegend=True
                                    ),
                                    row=1, col=1
                                )
                        
                        # Highlight hunting events
                        hunting_df = file_df[file_df['hunting']].copy()
                        if not hunting_df.empty:
                            hunting_valid = hunting_df[['time', 'gear']].dropna()
                            if len(hunting_valid) > 0:
                                time_hunt = np.asarray(hunting_valid['time'].values, dtype=np.float64).tolist()
                                gear_hunt = np.asarray(hunting_valid['gear'].values, dtype=np.float64).tolist()
                                logger.debug(f"Plotting {len(time_hunt)} hunting event markers")
                                if len(time_hunt) > 0 and len(gear_hunt) > 0:
                                    fig.add_trace(
                                        go.Scatter(
                                            x=time_hunt,
                                            y=gear_hunt,
                                            mode='markers',
                                            name='Hunting',
                                            marker=dict(color='red', size=8, symbol='x', line=dict(width=2)),
                                            showlegend=(file_name == full_df['file'].unique()[0])
                                        ),
                                        row=1, col=1
                                    )
                    
                    # Collect speed and RPM data ranges before plotting to set proper axis ranges
                    speed_ranges = []
                    rpm_ranges = []
                    speed_trace_added = False
                    rpm_trace_added = False
                    
                    # Speed plot (row 2, left y-axis - automatically y2 for row 2)
                    if 'speed' in full_df.columns and full_df['speed'].notna().any():
                        for file_name in full_df['file'].unique():
                            file_df = full_df[full_df['file'] == file_name].copy()
                            # Filter out NaN values
                            speed_valid = file_df[['time', 'speed']].dropna()
                            if len(speed_valid) > 0:
                                time_speed = np.asarray(speed_valid['time'].values, dtype=np.float64).tolist()
                                speed_vals = np.asarray(speed_valid['speed'].values, dtype=np.float64).tolist()
                                if len(time_speed) > 0 and len(speed_vals) > 0:
                                    # Collect range for axis configuration
                                    speed_min_val = min(speed_vals)
                                    speed_max_val = max(speed_vals)
                                    speed_ranges.append((speed_min_val, speed_max_val))
                                    logger.info(f"[GEAR PLOT] Speed data: min={speed_min_val:.2f}, max={speed_max_val:.2f}, points={len(speed_vals)}, sample values: {speed_vals[:5] if len(speed_vals) >= 5 else speed_vals}")
                                    fig.add_trace(
                                        go.Scatter(
                                            x=time_speed,
                                            y=speed_vals,
                                            mode='lines',
                                            name='Speed',
                                            line=dict(width=3, color='#4CAF50'),  # Green for speed, thicker for visibility
                                            showlegend=True,
                                            hovertemplate='<b>Speed</b><br>Time: %{x:.2f} s<br>Speed: %{y:.2f} km/h<extra></extra>'
                                        ),
                                        row=2, col=1
                                    )
                                    speed_trace_added = True
                    
                    # RPM plot (row 2, right y-axis when both exist)
                    if 'rpm' in full_df.columns and full_df['rpm'].notna().any():
                        for file_name in full_df['file'].unique():
                            file_df = full_df[full_df['file'] == file_name].copy()
                            # Filter out NaN values
                            rpm_valid = file_df[['time', 'rpm']].dropna()
                            if len(rpm_valid) > 0:
                                time_rpm = np.asarray(rpm_valid['time'].values, dtype=np.float64).tolist()
                                rpm_vals = np.asarray(rpm_valid['rpm'].values, dtype=np.float64).tolist()
                                if len(time_rpm) > 0 and len(rpm_vals) > 0:
                                    # Collect range for axis configuration
                                    rpm_min_val = min(rpm_vals)
                                    rpm_max_val = max(rpm_vals)
                                    rpm_ranges.append((rpm_min_val, rpm_max_val))
                                    logger.info(f"[GEAR PLOT] RPM data: min={rpm_min_val:.2f}, max={rpm_max_val:.2f}, points={len(rpm_vals)}, sample values: {rpm_vals[:5] if len(rpm_vals) >= 5 else rpm_vals}")
                                    # If speed exists, use secondary y-axis (y4 overlaying y2), otherwise use y2
                                    if has_speed:
                                        fig.add_trace(
                                            go.Scatter(
                                                x=time_rpm,
                                                y=rpm_vals,
                                                mode='lines',
                                                name='RPM',
                                                line=dict(width=3, color='#FF9800'),  # Orange for RPM, thicker for visibility
                                                showlegend=True,
                                                yaxis='y4',  # Secondary axis overlaying y2
                                                hovertemplate='<b>RPM</b><br>Time: %{x:.2f} s<br>RPM: %{y:.0f}<extra></extra>'
                                            ),
                                            row=2, col=1
                                        )
                                    else:
                                        fig.add_trace(
                                            go.Scatter(
                                                x=time_rpm,
                                                y=rpm_vals,
                                                mode='lines',
                                                name='RPM',
                                                line=dict(width=2, color='#FF9800'),  # Orange for RPM, increased width
                                                showlegend=True
                                            ),
                                            row=2, col=1
                                        )
                                    rpm_trace_added = True
                    
                    # Configure yaxis4 AFTER adding traces if both speed and rpm exist
                    # This ensures the axis is properly linked to the subplot
                    if has_speed and has_rpm and speed_ranges and rpm_ranges:
                        # Calculate combined ranges with padding
                        speed_min = min(r[0] for r in speed_ranges)
                        speed_max = max(r[1] for r in speed_ranges)
                        # Ensure minimum padding, especially for near-zero values
                        speed_range = speed_max - speed_min
                        speed_padding = max(speed_range * 0.1, abs(speed_max) * 0.05) if speed_range > 0 else max(abs(speed_max), 5) * 0.1
                        if speed_padding == 0 or speed_padding < 0.5:
                            speed_padding = max(1, abs(speed_max) * 0.05)  # Minimum padding, ensure non-negative
                        
                        # Ensure speed_min doesn't go below 0 if data starts at 0
                        speed_range_min = max(0, speed_min - speed_padding) if speed_min >= 0 else speed_min - speed_padding
                        speed_range_max = speed_max + speed_padding
                        
                        rpm_min = min(r[0] for r in rpm_ranges)
                        rpm_max = max(r[1] for r in rpm_ranges)
                        rpm_range = rpm_max - rpm_min
                        rpm_padding = max(rpm_range * 0.1, abs(rpm_max) * 0.05) if rpm_range > 0 else max(abs(rpm_max), 50) * 0.1
                        if rpm_padding == 0 or rpm_padding < 5:
                            rpm_padding = max(10, abs(rpm_max) * 0.05)  # Minimum padding
                        
                        # Ensure rpm_min doesn't go below 0 if data starts at 0
                        rpm_range_min = max(0, rpm_min - rpm_padding) if rpm_min >= 0 else rpm_min - rpm_padding
                        rpm_range_max = rpm_max + rpm_padding
                        
                        # Ensure ranges are not all zeros or too small
                        if speed_max <= speed_min:
                            speed_range_min = 0
                            speed_range_max = max(10, speed_max + 1)
                        if rpm_max <= rpm_min:
                            rpm_range_min = 0
                            rpm_range_max = max(100, rpm_max + 10)
                        
                        logger.info(f"[GEAR PLOT] Setting axis ranges: Speed [{speed_range_min:.2f}, {speed_range_max:.2f}], RPM [{rpm_range_min:.2f}, {rpm_range_max:.2f}]")
                        logger.info(f"[GEAR PLOT] Speed trace added: {speed_trace_added}, RPM trace added: {rpm_trace_added}")
                        
                        # Configure y2 (Speed axis) FIRST with explicit range
                        fig.update_yaxes(
                            title_text="Speed (km/h)",
                            range=[speed_range_min, speed_range_max],
                            row=2, col=1
                        )
                        
                        # Configure y4 axis (RPM) with proper range and ensure it overlays y2 correctly
                        # For subplots with shared x-axis, we need to anchor to x2
                        fig.update_layout(
                            yaxis4=dict(
                                title=dict(text="RPM", font=dict(color='#FF9800', size=11)),
                                overlaying="y2",
                                side="right",
                                anchor="x2",  # Anchor to x-axis of row 2
                                range=[rpm_range_min, rpm_range_max],
                                showgrid=True,
                                gridcolor='rgba(255,255,255,0.1)',
                                gridwidth=1,
                                linecolor='rgba(255,255,255,0.3)',
                                tickfont=dict(color='#FF9800', size=10)  # Orange color to match RPM line
                            )
                        )
                        
                        logger.info(f"[GEAR PLOT] Configured dual y-axes: y2 (Speed) range=[{speed_range_min:.2f}, {speed_range_max:.2f}], y4 (RPM) range=[{rpm_range_min:.2f}, {rpm_range_max:.2f}]")
                    elif has_speed and speed_ranges:
                        logger.info(f"[GEAR PLOT] Configuring single y-axis for Speed only")
                        # Only speed - set range
                        speed_min = min(r[0] for r in speed_ranges)
                        speed_max = max(r[1] for r in speed_ranges)
                        speed_range = speed_max - speed_min
                        speed_padding = max(speed_range * 0.1, abs(speed_max) * 0.05) if speed_range > 0 else max(abs(speed_max), 5) * 0.1
                        if speed_padding == 0 or speed_padding < 0.5:
                            speed_padding = max(1, abs(speed_max) * 0.05)
                        speed_range_min = max(0, speed_min - speed_padding) if speed_min >= 0 else speed_min - speed_padding
                        speed_range_max = speed_max + speed_padding
                        if speed_max <= speed_min:
                            speed_range_min = 0
                            speed_range_max = max(10, speed_max + 1)
                        fig.update_yaxes(
                            title_text="Speed (km/h)",
                            range=[speed_range_min, speed_range_max],
                            row=2, col=1
                        )
                        logger.info(f"[GEAR PLOT] Configured single y-axis for Speed: [{speed_range_min:.2f}, {speed_range_max:.2f}]")
                    elif has_rpm and rpm_ranges:
                        logger.info(f"[GEAR PLOT] Configuring single y-axis for RPM only")
                        # Only RPM - set range
                        rpm_min = min(r[0] for r in rpm_ranges)
                        rpm_max = max(r[1] for r in rpm_ranges)
                        rpm_range = rpm_max - rpm_min
                        rpm_padding = max(rpm_range * 0.1, abs(rpm_max) * 0.05) if rpm_range > 0 else max(abs(rpm_max), 50) * 0.1
                        if rpm_padding == 0 or rpm_padding < 5:
                            rpm_padding = max(10, abs(rpm_max) * 0.05)
                        rpm_range_min = max(0, rpm_min - rpm_padding) if rpm_min >= 0 else rpm_min - rpm_padding
                        rpm_range_max = rpm_max + rpm_padding
                        if rpm_max <= rpm_min:
                            rpm_range_min = 0
                            rpm_range_max = max(100, rpm_max + 10)
                        fig.update_yaxes(
                            title_text="RPM",
                            range=[rpm_range_min, rpm_range_max],
                            row=2, col=1
                        )
                        logger.info(f"[GEAR PLOT] Configured single y-axis for RPM: [{rpm_range_min:.2f}, {rpm_range_max:.2f}]")
                    
                    # Throttle plot (row 3)
                    try:
                        if 'throttle' in full_df.columns and full_df['throttle'].notna().any():
                            for file_name in full_df['file'].unique():
                                file_df = full_df[full_df['file'] == file_name].copy()
                                # Filter out NaN values
                                throttle_valid = file_df[['time', 'throttle']].dropna()
                                if len(throttle_valid) > 0:
                                    time_throttle = np.asarray(throttle_valid['time'].values, dtype=np.float64).tolist()
                                    throttle_vals = np.asarray(throttle_valid['throttle'].values, dtype=np.float64).tolist()
                                    if len(time_throttle) > 0 and len(throttle_vals) > 0:
                                        fig.add_trace(
                                            go.Scatter(
                                                x=time_throttle,
                                                y=throttle_vals,
                                                mode='lines',
                                                name='Throttle',
                                                line=dict(width=1.5),
                                                showlegend=False
                                            ),
                                            row=3, col=1
                                        )
                                        logger.debug(f"[GEAR PLOT] Added throttle trace for {file_name}")
                    except Exception as throttle_err:
                        logger.warning(f"[GEAR PLOT] Error adding throttle trace: {throttle_err}")
                    
                    # Add misfire events as markers on Speed/RPM subplot (row 2)
                    if misfire_events and len(misfire_events) > 0:
                        try:
                            misfire_times = [evt['time'] for evt in misfire_events if 'time' in evt]
                            misfire_rpms = [evt.get('rpm', 0) for evt in misfire_events if 'time' in evt]
                            
                            if misfire_times and misfire_rpms:
                                # Interpolate RPM values from full_df for better accuracy
                                misfire_rpm_values = []
                                for mf_time, mf_rpm in zip(misfire_times, misfire_rpms):
                                    # Find closest RPM value in dataframe
                                    time_mask = full_df['time'].notna()
                                    if time_mask.any() and 'rpm' in full_df.columns:
                                        time_diffs = np.abs(full_df.loc[time_mask, 'time'] - mf_time)
                                        closest_idx = time_diffs.idxmin()
                                        if time_diffs[closest_idx] < 0.5:  # Within 0.5 seconds
                                            rpm_val = full_df.loc[closest_idx, 'rpm']
                                            if pd.notna(rpm_val):
                                                misfire_rpm_values.append((mf_time, rpm_val))
                                            else:
                                                misfire_rpm_values.append((mf_time, mf_rpm))
                                        else:
                                            misfire_rpm_values.append((mf_time, mf_rpm))
                                    else:
                                        misfire_rpm_values.append((mf_time, mf_rpm))
                                
                                if misfire_rpm_values:
                                    mf_times_only = [t for t, _ in misfire_rpm_values]
                                    mf_rpms_only = [r for _, r in misfire_rpm_values]
                                    
                                    # Add misfire markers on RPM axis (y4)
                                    fig.add_trace(
                                        go.Scatter(
                                            x=mf_times_only,
                                            y=mf_rpms_only,
                                            mode='markers',
                                            name='Misfire',
                                            marker=dict(
                                                symbol='x',
                                                size=12,
                                                color='#FF4444',  # Red for misfires
                                                line=dict(width=2, color='#FF0000')
                                            ),
                                            showlegend=True,
                                            yaxis='y4',  # Use RPM axis
                                            hovertemplate='<b>Misfire Event</b><br>' +
                                                         'Time: %{x:.2f} s<br>' +
                                                         'RPM: %{y:.0f}<br>' +
                                                         '<extra></extra>'
                                        ),
                                        row=2, col=1
                                    )
                                    logger.info(f"[GEAR PLOT] ✅ Added {len(misfire_rpm_values)} misfire markers to Speed/RPM subplot")
                        except Exception as misfire_plot_err:
                            logger.warning(f"[GEAR PLOT] Error adding misfire markers: {misfire_plot_err}")
                    
                    logger.info(f"[GEAR PLOT] ✅ Completed trace addition phase. Total traces: {len(fig.data)}")
                    
                    # Dark theme layout - deep black background like IUPR and fuel consumption
                    # Use plotly_dark template with explicit deep black backgrounds
                    try:
                        logger.info(f"[GEAR PLOT] Updating layout...")
                        fig.update_layout(
                            title='Gear Hunting Analysis - Multi-Signal View',
                            height=1600,  # Increased from 1200 for bigger graph
                            template="plotly_dark",  # Dark background to match misfire section
                            paper_bgcolor='black',  # Deep black background like IUPR/fuel
                            plot_bgcolor='black',  # Deep black background like IUPR/fuel
                            hovermode='x unified',
                            autosize=True,
                            font=dict(color='#dce1e6')  # Light text for dark mode
                        )
                        
                        # Update x-axis (only on bottom subplot)
                        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
                        
                        # Update y-axes
                        fig.update_yaxes(title_text="Gear", row=1, col=1)
                        
                        # Row 2: Combined Speed & RPM panel
                        # Note: Titles and ranges were already set above during axis configuration
                        # Only update titles here if they weren't set above (single signal case)
                        if has_speed and not has_rpm:
                            # Only speed was plotted - title should already be set, but ensure it
                            pass  # Already handled above
                        elif has_rpm and not has_speed:
                            # Only RPM was plotted - title should already be set, but ensure it
                            pass  # Already handled above
                        # If both exist, everything is already configured above
                        
                        # Row 3: Throttle
                        if 'throttle' in full_df.columns and full_df['throttle'].notna().any():
                            fig.update_yaxes(title_text="Throttle Position", row=3, col=1)
                        logger.info(f"[GEAR PLOT] ✅ Layout updated successfully")
                    except Exception as layout_err:
                        logger.error(f"[GEAR PLOT] Error updating layout: {layout_err}", exc_info=True)
                
                    # Log trace count BEFORE plot creation attempt
                    trace_count = len(fig.data)
                    logger.info(f"[GEAR PLOT] Figure has {trace_count} traces before plot creation attempt")
                    
                    # CRITICAL: Always create the main plot even if some subplots are empty
                    # This is the main plot - it MUST be created if we have any data
                    logger.info(f"[GEAR PLOT] 🔴 CRITICAL: About to enter plot creation block. fig exists: {'fig' in locals()}, fig.data length: {trace_count}")
                    
                    # Always try to create the plot if we have any traces
                    if len(fig.data) > 0:
                        logger.info(f"[GEAR PLOT] ✅ Trace count > 0, proceeding with plot creation")
                        logger.info(f"[GEAR PLOT] Attempting to create plot with {len(fig.data)} traces")
                        
                        # Validate traces before creating plot
                        speed_traces = [t for t in fig.data if 'Speed' in str(t.name)]
                        rpm_traces = [t for t in fig.data if 'RPM' in str(t.name) or 'rpm' in str(t.name).lower()]
                        gear_traces = [t for t in fig.data if 'Gear' in str(t.name) or 'gear' in str(t.name).lower()]
                        logger.info(f"[GEAR PLOT] Trace breakdown: {len(gear_traces)} Gear, {len(speed_traces)} Speed, {len(rpm_traces)} RPM traces")
                        
                        # Log axis configuration
                        try:
                            if hasattr(fig, 'layout') and fig.layout:
                                if hasattr(fig.layout, 'yaxis2'):
                                    y2_range = getattr(fig.layout.yaxis2, 'range', None)
                                    logger.info(f"[GEAR PLOT] yaxis2 (Speed) range: {y2_range}")
                                if hasattr(fig.layout, 'yaxis4'):
                                    y4_range = getattr(fig.layout.yaxis4, 'range', None)
                                    logger.info(f"[GEAR PLOT] yaxis4 (RPM) range: {y4_range}, overlaying: {getattr(fig.layout.yaxis4, 'overlaying', None)}")
                        except Exception as layout_err:
                            logger.warning(f"[GEAR PLOT] Error reading layout: {layout_err}")
                        
                        # Convert figure to JSON - this should always work if figure was created
                        try:
                            plot_json_str = fig.to_json()
                            logger.info(f"[GEAR PLOT] Successfully converted figure to JSON ({len(plot_json_str)} chars)")
                            
                            # Parse and validate the JSON structure
                            import json as json_module
                            plot_json_obj = json_module.loads(plot_json_str)
                            
                            # Always add the plot if JSON was created successfully
                            if plot_json_obj.get('data') and len(plot_json_obj['data']) > 0:
                                plots['Gear Analysis Multi-Signal'] = {"plotly_json": plot_json_str}
                                logger.info(f"[GEAR PLOT] ✅ Successfully created gear analysis plot with {len(fig.data)} traces, {len(plot_json_obj['data'])} data traces in JSON")
                                
                                # Validate axis ranges in JSON
                                layout = plot_json_obj.get('layout', {})
                                if 'yaxis2' in layout:
                                    y2_range = layout['yaxis2'].get('range')
                                    logger.info(f"[GEAR PLOT] JSON yaxis2 range: {y2_range}")
                                if 'yaxis4' in layout:
                                    y4_range = layout['yaxis4'].get('range')
                                    logger.info(f"[GEAR PLOT] JSON yaxis4 range: {y4_range}")
                            else:
                                logger.error(f"[GEAR PLOT] ❌ Plot JSON has no data: {list(plot_json_obj.keys())}")
                                # Still try to add it - maybe frontend can handle it
                                plots['Gear Analysis Multi-Signal'] = {"plotly_json": plot_json_str}
                                logger.warning(f"[GEAR PLOT] Added plot anyway despite missing data in JSON")
                        except Exception as json_err:
                            logger.error(f"[GEAR PLOT] ❌ Failed to convert/validate plot JSON: {json_err}", exc_info=True)
                            # Try to create a minimal plot anyway
                            try:
                                minimal_json = fig.to_json()
                                plots['Gear Analysis Multi-Signal'] = {"plotly_json": minimal_json}
                                logger.warning(f"[GEAR PLOT] Added minimal plot despite JSON error")
                            except Exception as minimal_err:
                                logger.error(f"[GEAR PLOT] ❌ Failed to create even minimal plot: {minimal_err}")
                    else:
                        logger.error(f"[GEAR PLOT] ❌ Gear analysis plot has no traces - cannot create plot")
                        logger.error(f"[GEAR PLOT] Full dataframe info: has_speed={has_speed}, has_rpm={has_rpm}, full_df shape={full_df.shape if full_df is not None else 'None'}")
                        # Try to create plot anyway with empty figure if we have the figure object
                        try:
                            if 'fig' in locals():
                                logger.warning(f"[GEAR PLOT] Attempting to create plot with empty figure as fallback")
                                plots['Gear Analysis Multi-Signal'] = {"plotly_json": fig.to_json()}
                                logger.warning(f"[GEAR PLOT] Created empty plot as fallback")
                        except Exception as fallback_err:
                            logger.error(f"[GEAR PLOT] ❌ Fallback plot creation also failed: {fallback_err}")
                except Exception as e:
                    logger.error(f"[GEAR PLOT] ❌ Failed to create gear analysis plot: {e}", exc_info=True)
                    # Try to create a basic plot anyway if figure exists
                    try:
                        if 'fig' in locals():
                            logger.warning(f"[GEAR PLOT] Attempting emergency plot creation")
                            plots['Gear Analysis Multi-Signal'] = {"plotly_json": fig.to_json()}
                            logger.warning(f"[GEAR PLOT] ✅ Emergency plot creation succeeded")
                    except Exception as emergency_err:
                        logger.error(f"[GEAR PLOT] ❌ Emergency plot creation failed: {emergency_err}", exc_info=True)
                    else:
                        logger.info(f"[GEAR PLOT] Plot creation block completed successfully")
                        
                    # CRITICAL CHECK: Ensure plot was actually added
                    if 'Gear Analysis Multi-Signal' not in plots:
                        logger.error(f"[GEAR PLOT] 🔴🔴🔴 CRITICAL ERROR: Plot was NOT added to plots dict! Attempting emergency creation...")
                        try:
                            if 'fig' in locals():
                                logger.warning(f"[GEAR PLOT] Emergency: Creating plot with {len(fig.data) if hasattr(fig, 'data') else 0} traces")
                                plots['Gear Analysis Multi-Signal'] = {"plotly_json": fig.to_json()}
                                logger.warning(f"[GEAR PLOT] ✅ Emergency plot creation succeeded")
                            else:
                                logger.error(f"[GEAR PLOT] 🔴 fig object does not exist in locals!")
                        except Exception as emergency_final:
                            logger.error(f"[GEAR PLOT] 🔴🔴🔴 FINAL EMERGENCY FAILED: {emergency_final}", exc_info=True)
                    else:
                        logger.info(f"[GEAR PLOT] ✅ Confirmed: Plot 'Gear Analysis Multi-Signal' is in plots dict")
                except Exception as plot_creation_error:
                    logger.error(f"[GEAR PLOT] 🔴🔴🔴 CRITICAL: Exception in plot creation block: {plot_creation_error}", exc_info=True)
                    # EMERGENCY: Try to create plot anyway if figure exists
                    try:
                        if 'fig' in locals():
                            logger.warning(f"[GEAR PLOT] EMERGENCY: Attempting to create plot despite error")
                            plots['Gear Analysis Multi-Signal'] = {"plotly_json": fig.to_json()}
                            logger.warning(f"[GEAR PLOT] ✅ EMERGENCY plot creation succeeded")
                    except Exception as emergency_err:
                        logger.error(f"[GEAR PLOT] 🔴🔴🔴 EMERGENCY PLOT CREATION ALSO FAILED: {emergency_err}", exc_info=True)
                finally:
                    logger.info(f"[GEAR PLOT] 🔵 EXITING PLOT CREATION BLOCK - plots dict now has {len(plots)} plots: {list(plots.keys())}")
            else:
                logger.warning(f"[GEAR PLOT] ⚠️  full_df is None or empty - skipping plot creation")
        
        # Severity distribution plot
        if all_hunting_events:
            severity_df = pd.DataFrame(all_hunting_events)
            if 'severity' in severity_df.columns:
                # Filter out NaN and invalid severity values
                severity_valid = severity_df['severity'].dropna()
                if len(severity_valid) > 0:
                    # Convert to list for proper serialization
                    severity_list = severity_valid.values.tolist()
                    fig_severity = go.Figure()
                    
                    fig_severity.add_trace(go.Histogram(
                        x=severity_list,
                        nbinsx=20,
                        name='Hunting Event Severity',
                        marker_color='coral',
                        opacity=0.7
                    ))
                    
                    fig_severity.update_layout(
                        title='Hunting Event Severity Distribution',
                        xaxis_title='Severity Score (0-100)',
                        yaxis_title='Number of Events',
                        template="plotly_dark",  # Dark background - match misfire style
                        paper_bgcolor='black',  # Deep black background like IUPR/fuel
                        plot_bgcolor='black',  # Deep black background like IUPR/fuel
                        font=dict(color='#dce1e6')  # Light text for dark mode
                    )
                    
                    try:
                        plots['Severity Distribution'] = {"plotly_json": fig_severity.to_json()}
                        logger.info(f"[GEAR PLOT] ✅ Created: Severity Distribution with {len(severity_valid)} events")
                    except Exception as e:
                        logger.error(f"[GEAR PLOT] ❌ Failed to create severity distribution plot: {e}", exc_info=True)
                else:
                    logger.warning("No valid severity data for distribution plot")
        
        # Gear shift frequency plot
        shift_freq_data = []
        for file_name in full_df['file'].unique():
            file_df = full_df[full_df['file'] == file_name]
            time_range = file_df['time'].max() - file_df['time'].min()
            shift_count = file_df['gear_change'].sum()
            freq = shift_count / (time_range / 60.0) if time_range > 0 else 0  # shifts per minute
            shift_freq_data.append({
                'file': file_name[:30],
                'frequency': round(freq, 2)
            })
        
        if shift_freq_data:
            freq_df = pd.DataFrame(shift_freq_data)
            # Filter out invalid frequency values
            freq_df = freq_df[freq_df['frequency'].notna() & (freq_df['frequency'] >= 0)]
            if len(freq_df) > 0:
                # Convert to lists for proper serialization
                file_names = freq_df['file'].values.tolist()
                frequencies = freq_df['frequency'].values.tolist()
                fig_freq = go.Figure()
                fig_freq.add_trace(go.Bar(
                    x=file_names,
                    y=frequencies,
                    name='Shift Frequency',
                    marker_color='steelblue'
                ))
                fig_freq.update_layout(
                    title='Gear Shift Frequency by File',
                    xaxis_title='File',
                    yaxis_title='Shifts per Minute',
                    template="plotly_dark",  # Dark background - match misfire style
                    paper_bgcolor='black',  # Deep black background like IUPR/fuel
                    plot_bgcolor='black',  # Deep black background like IUPR/fuel
                    font=dict(color='#dce1e6'),  # Light text for dark mode
                    xaxis_tickangle=-45
                )
                try:
                    plots['Shift Frequency'] = {"plotly_json": fig_freq.to_json()}
                    logger.info(f"[GEAR PLOT] ✅ Created: Shift Frequency with {len(freq_df)} files")
                except Exception as e:
                    logger.error(f"[GEAR PLOT] ❌ Failed to create shift frequency plot: {e}", exc_info=True)
            else:
                logger.warning("No valid frequency data for shift frequency plot")
        
    # Summary of all plots created (even if plots dict is empty)
    if len(plots) > 0:
        logger.info(f"[GEAR PLOT] 📊 Plot creation summary: Created {len(plots)} plots: {list(plots.keys())}")
    else:
        logger.warning(f"[GEAR PLOT] ⚠️  WARNING: No plots were created! Check above logs for errors.")
        if 'full_df' in locals():
            logger.warning(f"[GEAR PLOT] Debug: full_df exists={full_df is not None}, all_hunting_events={len(all_hunting_events) if all_hunting_events else 0}")
    
    # Statistics summary
    statistics = []
    if all_hunting_events:
        events_df = pd.DataFrame(all_hunting_events)
        statistics = [
            {"metric": "Total Hunting Events", "value": len(all_hunting_events)},
            {"metric": "Average Event Duration (s)", "value": round(events_df['duration'].mean(), 2) if 'duration' in events_df.columns else 0},
            {"metric": "Average Shifts per Event", "value": round(events_df['shift_count'].mean(), 1) if 'shift_count' in events_df.columns else 0},
            {"metric": "Average Shift Rate (shifts/s)", "value": round(events_df['shift_rate'].mean(), 2) if 'shift_rate' in events_df.columns else 0},
            {"metric": "Max Severity Score", "value": round(events_df['severity'].max(), 1) if 'severity' in events_df.columns else 0},
            {"metric": "Average Severity Score", "value": round(events_df['severity'].mean(), 1) if 'severity' in events_df.columns else 0},
        ]
    
    statistics.extend([
        {"metric": "Files Processed", "value": global_stats["files_processed"]},
        {"metric": "Total Gear Shifts", "value": global_stats["total_shifts"]},
        {"metric": "Total Hunting Events", "value": global_stats["total_hunting_events"]}
    ])
    
    return {
        "tables": {
            "Hunting Events": all_hunting_events[:500],  # Limit to first 500 for display
            "File Summary": file_summaries,
            "Statistics": statistics,
            "Signal Mapping": signal_mapping_report[:100]  # Show signal mapping report
        },
        "plots": plots,
        "meta": {
            "ok": True,
            "files_processed": global_stats["files_processed"],
            "total_events": len(all_hunting_events),
            "total_shifts": global_stats["total_shifts"],
            "signal_mapping": {row["signal_role"]: row["channel_name"] for row in signal_mapping_report if row.get("found")}
        }
    }
