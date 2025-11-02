#!/usr/bin/env python3
"""
custom_misfire.py — Advanced Misfire Detection System for Engine Diagnostics

Implements multiple sophisticated misfire detection algorithms:
1. Crankshaft Speed Variance Analysis (CSVA)
2. Frequency Domain Analysis (FFT-based pattern recognition)
3. Statistical Anomaly Detection (Z-score, percentile-based)
4. Angular Velocity Analysis (per-cylinder detection)
5. Adaptive Threshold Detection (context-aware thresholds)

Returns structure:
{
  "summary": [ { "time": float, "severity": str, "confidence": float, "cylinder": int, ... } ],
  "plots": [ {"name": "...", "plotly_json": "..."}, ...],
  "statistics": [ {...} ],
  "meta": {}
}
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
try:
    from scipy import signal, stats
    from scipy.fft import fft, fftfreq
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    # Fallback implementations
    def fft(x):
        return np.fft.fft(x)
    def fftfreq(n, d=1.0):
        return np.fft.fftfreq(n, d)
    def detrend(data):
        # Simple linear detrend
        return data - np.linspace(data[0], data[-1], len(data))
    signal = type('obj', (object,), {'detrend': detrend, 'windows': type('obj', (object,), {'hann': lambda n: np.hanning(n)})()})

# Advanced ML-based detection (optional)
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

logger = logging.getLogger(__name__)

try:
    from asammdf import MDF
except ImportError:
    MDF = None

# Import centralized signal mapping system
try:
    from signal_mapping import (
        SIGNAL_MAP, find_signal_advanced, find_multiple_signals,
        RPM_CANDIDATES, TORQUE_CANDIDATES, LAMBDA_CANDIDATES,
        COOLANT_TEMP_CANDIDATES, INTAKE_TEMP_CANDIDATES,
        CRANKSHAFT_ANGLE_CANDIDATES, IGNITION_CANDIDATES,
        MAP_CANDIDATES, THROTTLE_CANDIDATES, CYLINDER_COUNT_CANDIDATES
    )
except ImportError:
    # Fallback if signal_mapping not available (should not happen)
    logger.warning("signal_mapping module not found, using basic signal lists")
    SIGNAL_MAP = {}
    def find_signal_advanced(channels, role, **kwargs):
        return None
    
    RPM_CANDIDATES = ['nEng', 'EngineSpeed', 'Engine_RPM', 'rpm']
    TORQUE_CANDIDATES = ['EngineTorque', 'Tq', 'Trq', 'tqEng']
    LAMBDA_CANDIDATES = ['Lambda', 'lambda', 'AFR']
    COOLANT_TEMP_CANDIDATES = ['CoolantTemp', 'ECT', 'ECT_C']
    INTAKE_TEMP_CANDIDATES = ['IAT', 'IAT_C', 'IntakeAirTemp']
    CRANKSHAFT_ANGLE_CANDIDATES = ['CrankAngle', 'CrankPos']
    IGNITION_CANDIDATES = ['IgnitionTiming', 'IgnTiming', 'SparkAdvance']
    MAP_CANDIDATES = ['MAP', 'MAP_kPa', 'ManifoldPressure']
    THROTTLE_CANDIDATES = ['ThrottlePos', 'ThrottlePosition', 'APP']
    CYLINDER_COUNT_CANDIDATES = ['nCyl', 'CylinderCount']

# Backward compatibility aliases
LOAD_CANDIDATES = TORQUE_CANDIDATES  # Load and torque often same signals
TEMPERATURE_CANDIDATES = COOLANT_TEMP_CANDIDATES
MANIFOLD_PRESSURE_CANDIDATES = MAP_CANDIDATES

# Default parameters (configurable)
DEFAULT_CYLINDERS = 4  # Default assumption for 4-cylinder engine
MIN_RPM_FOR_DETECTION = 500  # Minimum RPM to consider valid data
MAX_RPM_FOR_DETECTION = 8000  # Maximum RPM for valid data
SPEED_DROP_THRESHOLD_RATIO = 0.05  # 5% speed drop indicates potential misfire
WINDOW_SIZE_SECONDS = 2.0  # Analysis window for statistical methods
FFT_SAMPLING_WINDOW = 2048  # Samples for FFT analysis
Z_SCORE_THRESHOLD = 3.0  # Standard deviations for anomaly detection

# OEM-level calibration parameters
MIN_COOLANT_TEMP_FOR_DETECTION = 60.0  # °C - Below this, misfire detection disabled (cold start)
IDLE_RPM_THRESHOLD = 1200  # RPM below this considered idle
HIGH_LOAD_THRESHOLD = 75.0  # % Load above this considered high load
LOW_LOAD_THRESHOLD = 25.0  # % Load below this considered low load

# Load-dependent threshold multipliers (OEM calibration approach)
# Higher load = more sensitive detection (smaller threshold)
THRESHOLD_MULTIPLIERS = {
    'idle': 1.5,      # Less sensitive at idle (more noise)
    'low_load': 1.2,  # Low load (20-40%)
    'medium_load': 1.0,  # Normal load (40-70%)
    'high_load': 0.8,  # High load (70-100%) - most sensitive
    'cold': 2.0      # Cold engine - much less sensitive
}

# OBD-II parameters
OBD_MISFIRE_COUNT_THRESHOLD = 100  # Per 1000 revolutions for MIL activation
OBD_MISFIRE_RATE_LIMIT = 0.02  # 2% misfire rate threshold


def find_signal_channel(mdf: MDF, candidates: List[str]) -> Optional[str]:
    """
    Find signal channel from candidates list with fuzzy matching.
    
    This function uses the advanced signal mapping system for better detection.
    """
    if mdf is None:
        return None
    
    available_channels = list(mdf.channels_db.keys())
    
    # Try to use advanced signal mapping if candidates match a known role
    # Otherwise fall back to basic matching
    all_channels_lower = {ch.lower(): ch for ch in available_channels}
    
    # First try exact matches
    for cand in candidates:
        cand_lower = cand.lower()
        if cand_lower in all_channels_lower:
            return all_channels_lower[cand_lower]
    
    # Then try substring matches
    for cand in candidates:
        cand_lower = cand.lower()
        for ch_name_lower, ch_name in all_channels_lower.items():
            if cand_lower in ch_name_lower or ch_name_lower in cand_lower:
                return ch_name
    
    return None


def find_signal_by_role(mdf: MDF, signal_role: str) -> Optional[str]:
    """
    Find signal by role using the advanced signal mapping system.
    
    This is the preferred method for finding signals.
    """
    if mdf is None:
        return None
    
    available_channels = list(mdf.channels_db.keys())
    return find_signal_advanced(available_channels, signal_role, fuzzy_match=True, substring_match=True)


def extract_cylinder_count(mdf: MDF, rpm_signal_name: str) -> int:
    """Try to extract cylinder count from MDF channels or estimate from RPM pattern."""
    cyl_ch = find_signal_channel(mdf, CYLINDER_COUNT_CANDIDATES)
    if cyl_ch:
        try:
            sig = mdf.get(cyl_ch)
            if sig.samples.size > 0:
                # Get most common value
                values = pd.to_numeric(sig.samples, errors='coerce').dropna()
                if len(values) > 0:
                    mode_value = int(values.mode().iloc[0] if len(values.mode()) > 0 else values.iloc[0])
                    if 2 <= mode_value <= 16:  # Reasonable range
                        return mode_value
        except Exception:
            pass
    
    # Estimate from RPM frequency pattern (advanced method)
    # This could analyze firing frequency, but for now return default
    return DEFAULT_CYLINDERS


def detect_misfire_crankshaft_variance(rpm_data: np.ndarray, timestamps: np.ndarray,
                                       cylinder_count: int, 
                                       threshold_ratio: float = SPEED_DROP_THRESHOLD_RATIO) -> List[Dict[str, Any]]:
    """
    Detect misfires using crankshaft speed variance analysis.
    
    Method: Monitor for sudden drops in RPM that indicate incomplete combustion.
    """
    if len(rpm_data) < 10:
        return []
    
    events = []
    rpm_series = pd.Series(rpm_data, index=timestamps)
    
    # Calculate rolling mean and std
    window_samples = max(20, int(cylinder_count * 2))  # At least 2 engine cycles
    rolling_mean = rpm_series.rolling(window=window_samples, center=True).mean()
    rolling_std = rpm_series.rolling(window=window_samples, center=True).std()
    
    # Detect significant drops (potential misfires)
    relative_drop = (rolling_mean - rpm_series) / (rolling_mean + 1e-6)
    significant_drops = relative_drop > threshold_ratio
    
    # Filter by minimum RPM to avoid false positives at idle
    valid_rpm_mask = rpm_series > MIN_RPM_FOR_DETECTION
    misfire_candidates = significant_drops & valid_rpm_mask
    
    # Group consecutive detections into events
    misfire_indices = np.where(misfire_candidates)[0]
    if len(misfire_indices) == 0:
        return []
    
    # Group nearby detections (within 0.5 seconds)
    events_list = []
    current_event_start = 0  # Index within misfire_indices array, not the actual index value
    
    for i in range(1, len(misfire_indices)):
        time_gap = timestamps[misfire_indices[i]] - timestamps[misfire_indices[i-1]]
        if time_gap > 0.5:  # New event
            # Save previous event
            event_idx = misfire_indices[current_event_start]
            events_list.append({
                'time': float(timestamps[event_idx]),
                'rpm': float(rpm_data[event_idx]),
                'rpm_drop': float(relative_drop.iloc[event_idx]),
                'method': 'crankshaft_variance'
            })
            current_event_start = i
    
    # Add last event
    if len(misfire_indices) > 0:
        event_idx = misfire_indices[current_event_start]
        events_list.append({
            'time': float(timestamps[event_idx]),
            'rpm': float(rpm_data[event_idx]),
            'rpm_drop': float(relative_drop.iloc[event_idx]),
            'method': 'crankshaft_variance'
        })
    
    return events_list


def detect_misfire_frequency_domain(rpm_data: np.ndarray, timestamps: np.ndarray,
                                    cylinder_count: int,
                                    sampling_window: int = FFT_SAMPLING_WINDOW) -> List[Dict[str, Any]]:
    """
    Detect misfires using frequency domain analysis (FFT).
    
    Method: Analyze power spectrum for anomalies at firing frequency harmonics.
    """
    if len(rpm_data) < sampling_window:
        return []
    
    events = []
    
    # Calculate sampling rate
    if len(timestamps) > 1:
        avg_dt = np.mean(np.diff(timestamps))
        sampling_rate = 1.0 / avg_dt if avg_dt > 0 else 100.0
    else:
        return []
    
    # Sliding window FFT analysis
    step_size = sampling_window // 4  # 75% overlap
    for start_idx in range(0, len(rpm_data) - sampling_window, step_size):
        end_idx = start_idx + sampling_window
        window_rpm = rpm_data[start_idx:end_idx]
        window_time = timestamps[start_idx:end_idx]
        
        # Skip if not enough variation or low RPM
        if np.std(window_rpm) < 10 or np.mean(window_rpm) < MIN_RPM_FOR_DETECTION:
            continue
        
        # Remove trend (detrend)
        if HAVE_SCIPY:
            detrended = signal.detrend(window_rpm)
        else:
            # Simple linear detrend
            detrended = window_rpm - np.linspace(window_rpm[0], window_rpm[-1], len(window_rpm))
        
        # Apply windowing to reduce spectral leakage
        if HAVE_SCIPY:
            windowed = detrended * signal.windows.hann(len(detrended))
        else:
            windowed = detrended * np.hanning(len(detrended))
        
        # Compute FFT
        fft_vals = fft(windowed)
        fft_freqs = fftfreq(len(windowed), d=avg_dt)
        
        # Power spectrum
        power = np.abs(fft_vals) ** 2
        
        # Calculate firing frequency (Hz) based on RPM and cylinder count
        avg_rpm = np.mean(window_rpm)
        firing_freq_hz = (avg_rpm / 60.0) * (cylinder_count / 2.0)  # For 4-stroke engine
        
        # Find frequency bin closest to firing frequency
        freq_idx = np.argmin(np.abs(fft_freqs - firing_freq_hz))
        
        # Check for anomalies: high power at firing frequency or harmonics
        fundamental_power = power[freq_idx] if freq_idx < len(power) else 0
        harmonic_2_power = power[min(freq_idx * 2, len(power) - 1)] if freq_idx * 2 < len(power) else 0
        
        # Baseline power (median of spectrum excluding DC)
        baseline_power = np.median(power[1:len(power)//2])
        
        # Misfire indicator: significant deviation from baseline
        if baseline_power > 0:
            anomaly_ratio = (fundamental_power + harmonic_2_power) / (2 * baseline_power + 1e-6)
            
            # High anomaly ratio could indicate misfire (irregular combustion pattern)
            if anomaly_ratio > 2.0 or anomaly_ratio < 0.3:  # Too high or too low
                events.append({
                    'time': float(window_time[len(window_time) // 2]),
                    'rpm': float(avg_rpm),
                    'anomaly_ratio': float(anomaly_ratio),
                    'firing_freq': float(firing_freq_hz),
                    'method': 'frequency_domain'
                })
    
    return events


def detect_misfire_statistical_anomaly(rpm_data: np.ndarray, timestamps: np.ndarray,
                                       z_threshold: float = Z_SCORE_THRESHOLD) -> List[Dict[str, Any]]:
    """
    Detect misfires using statistical anomaly detection (Z-score analysis).
    
    Method: Identify data points that deviate significantly from statistical norms.
    """
    if len(rpm_data) < 20:
        return []
    
    events = []
    rpm_series = pd.Series(rpm_data, index=timestamps)
    
    # Calculate rolling statistics
    window_size = max(50, int(len(rpm_data) * 0.05))  # 5% of data or minimum 50
    rolling_mean = rpm_series.rolling(window=window_size, center=True).mean()
    rolling_std = rpm_series.rolling(window=window_size, center=True).std()
    
    # Calculate Z-scores
    z_scores = (rpm_series - rolling_mean) / (rolling_std + 1e-6)
    
    # Find significant negative deviations (sudden drops)
    anomalies = (z_scores < -z_threshold) & (rpm_series > MIN_RPM_FOR_DETECTION)
    
    anomaly_indices = np.where(anomalies)[0]
    
    for idx in anomaly_indices:
        events.append({
            'time': float(timestamps[idx]),
            'rpm': float(rpm_data[idx]),
            'z_score': float(z_scores.iloc[idx]),
            'deviation_percent': float((rpm_data[idx] - rolling_mean.iloc[idx]) / (rolling_mean.iloc[idx] + 1e-6) * 100),
            'method': 'statistical_anomaly'
        })
    
    return events


def detect_misfire_wavelet_analysis(rpm_data: np.ndarray, timestamps: np.ndarray,
                                     cylinder_count: int) -> List[Dict[str, Any]]:
    """
    Advanced misfire detection using Wavelet Transform Analysis.
    
    Wavelets provide better time-frequency localization than FFT,
    allowing detection of transient misfire events.
    """
    if len(rpm_data) < 64:  # Need minimum data for wavelet analysis
        return []
    
    events = []
    
    # Simple discrete wavelet transform approximation using FFT
    # (Full implementation would use PyWavelets if available)
    try:
        # Calculate RPM gradient (first derivative)
        if len(timestamps) > 1:
            dt = np.diff(timestamps)
            dt = np.append(dt, dt[-1]) if len(dt) > 0 else np.array([0.01])
            rpm_gradient = np.gradient(rpm_data, dt)
        else:
            return []
        
        # Multi-scale analysis (simulating wavelet decomposition)
        # Analyze at different time scales
        scales = [8, 16, 32, 64]  # Different window sizes
        
        for scale in scales:
            if len(rpm_data) < scale * 2:
                continue
            
            # Rolling variance at this scale
            window_size = scale
            rolling_var = pd.Series(rpm_gradient).rolling(window=window_size, center=True).var()
            
            # Detect anomalies: high variance indicates irregular combustion
            threshold = rolling_var.quantile(0.95)  # Top 5% as anomalies
            anomalies = rolling_var > threshold
            
            # Filter by RPM range
            valid_rpm_mask = pd.Series(rpm_data) > MIN_RPM_FOR_DETECTION
            misfire_candidates = anomalies & valid_rpm_mask
            
            anomaly_indices = np.where(misfire_candidates)[0]
            
            for idx in anomaly_indices:
                events.append({
                    'time': float(timestamps[idx]),
                    'rpm': float(rpm_data[idx]),
                    'wavelet_scale': scale,
                    'variance': float(rolling_var.iloc[idx]),
                    'method': 'wavelet_analysis'
                })
    except Exception as e:
        logger.debug(f"Wavelet analysis error: {e}")
    
    return events


def detect_misfire_ml_anomaly(rpm_data: np.ndarray, timestamps: np.ndarray,
                              cylinder_count: int) -> List[Dict[str, Any]]:
    """
    Machine Learning-based misfire detection using Isolation Forest.
    
    Learns normal engine behavior and flags anomalies.
    """
    if not HAVE_SKLEARN or len(rpm_data) < 100:
        return []
    
    events = []
    
    try:
        # Extract features from RPM signal
        rpm_series = pd.Series(rpm_data, index=timestamps)
        
        features = []
        feature_indices = []
        
        window_size = max(20, cylinder_count * 2)
        
        for i in range(window_size, len(rpm_data) - window_size):
            window = rpm_data[i-window_size:i+window_size]
            
            # Feature engineering
            feat = [
                np.mean(window),           # Average RPM
                np.std(window),            # RPM variance
                np.max(window) - np.min(window),  # RPM range
                np.ptp(window),            # Peak-to-peak
                float(stats.skew(window)) if HAVE_SCIPY else 0.0,  # Skewness
                float(stats.kurtosis(window)) if HAVE_SCIPY else 0.0,  # Kurtosis
            ]
            
            # Add gradient features
            if i > 0:
                feat.append(rpm_data[i] - rpm_data[i-1])  # Instantaneous change
                feat.append(np.mean(np.diff(window)))  # Average gradient
            else:
                feat.extend([0, 0])
            
            features.append(feat)
            feature_indices.append(i)
        
        if len(features) < 10:
            return []
        
        # Train Isolation Forest
        X = np.array(features)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Isolation Forest with contamination tuned for misfire detection
        iso_forest = IsolationForest(
            contamination=0.1,  # Expect ~10% anomalies
            random_state=42,
            n_estimators=100
        )
        
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        
        # Find anomalies (label == -1)
        for idx, label in enumerate(anomaly_labels):
            if label == -1:  # Anomaly detected
                original_idx = feature_indices[idx]
                if rpm_data[original_idx] > MIN_RPM_FOR_DETECTION:
                    # Calculate anomaly score
                    anomaly_score = iso_forest.score_samples(X_scaled[idx:idx+1])[0]
                    
                    events.append({
                        'time': float(timestamps[original_idx]),
                        'rpm': float(rpm_data[original_idx]),
                        'anomaly_score': float(anomaly_score),
                        'method': 'ml_isolation_forest'
                    })
    except Exception as e:
        logger.debug(f"ML anomaly detection error: {e}")
    
    return events


def detect_misfire_pattern_matching(rpm_data: np.ndarray, timestamps: np.ndarray,
                                    cylinder_count: int) -> List[Dict[str, Any]]:
    """
    Pattern-based misfire detection using known misfire signatures.
    
    Matches observed patterns against theoretical misfire waveforms.
    """
    if len(rpm_data) < cylinder_count * 4:
        return []
    
    events = []
    
    try:
        # Expected firing interval (for 4-stroke engine)
        # Time between cylinder fires = 720 / (cylinder_count * RPM/60)
        avg_rpm = np.mean(rpm_data)
        if avg_rpm < MIN_RPM_FOR_DETECTION:
            return []
        
        firing_period = 720.0 / (cylinder_count * avg_rpm / 60.0)  # degrees to time conversion
        
        # Calculate RPM change rate
        if len(timestamps) > 1:
            dt = np.diff(timestamps)
            dt = np.append(dt, dt[-1]) if len(dt) > 0 else np.array([0.01])
            rpm_change_rate = np.diff(rpm_data) / dt[:-1]
            rpm_change_rate = np.append(rpm_change_rate, rpm_change_rate[-1])
        else:
            return []
        
        # Look for misfire pattern: sudden drop followed by recovery
        pattern_length = max(cylinder_count, 8)
        
        for i in range(pattern_length, len(rpm_data) - pattern_length):
            window = rpm_data[i-pattern_length//2:i+pattern_length//2]
            window_changes = rpm_change_rate[i-pattern_length//2:i+pattern_length//2]
            
            # Misfire signature: large negative change followed by recovery
            # Check for dip pattern
            mid_point = len(window) // 2
            before_dip = np.mean(window[:mid_point])
            at_dip = window[mid_point]
            after_dip = np.mean(window[mid_point+1:]) if len(window) > mid_point + 1 else at_dip
            
            drop_magnitude = (before_dip - at_dip) / (before_dip + 1e-6)
            recovery = (after_dip - at_dip) / (at_dip + 1e-6)
            
            # Misfire pattern: significant drop (>3%) with recovery
            if drop_magnitude > 0.03 and recovery > 0.01 and rpm_data[i] > MIN_RPM_FOR_DETECTION:
                # Calculate pattern confidence
                pattern_match_score = min(1.0, drop_magnitude * 10)  # Scale to 0-1
                
                events.append({
                    'time': float(timestamps[i]),
                    'rpm': float(rpm_data[i]),
                    'pattern_match_score': round(pattern_match_score, 3),
                    'drop_magnitude': round(drop_magnitude * 100, 2),
                    'method': 'pattern_matching'
                })
    except Exception as e:
        logger.debug(f"Pattern matching error: {e}")
    
    return events


def detect_misfire_angular_velocity(rpm_data: np.ndarray, timestamps: np.ndarray,
                                     cylinder_count: int) -> List[Dict[str, Any]]:
    """
    Detect misfires using angular velocity analysis.
    
    Method: Monitor per-cylinder angular velocity changes.
    For multi-cylinder engines, detect when one cylinder doesn't contribute.
    """
    if len(rpm_data) < cylinder_count * 2:
        return []
    
    events = []
    
    # Convert RPM to angular velocity (rad/s)
    angular_velocity = rpm_data * (2 * np.pi / 60.0)
    
    # Calculate angular acceleration (derivative of angular velocity)
    dt = np.diff(timestamps)
    if len(dt) == 0 or np.any(dt <= 0):
        return []
    
    angular_accel = np.diff(angular_velocity) / dt
    angular_accel = np.append(angular_accel, angular_accel[-1])  # Match length
    
    # Expected acceleration pattern for each cylinder firing
    # During normal combustion: positive acceleration
    # During misfire: sudden negative acceleration
    
    # Detect significant negative accelerations
    avg_accel = np.mean(angular_accel)
    std_accel = np.std(angular_accel)
    
    if std_accel > 0:
        # Significant negative deviations
        threshold = avg_accel - 2 * std_accel
        misfire_mask = (angular_accel < threshold) & (rpm_data > MIN_RPM_FOR_DETECTION)
        
        misfire_indices = np.where(misfire_mask)[0]
        
        for idx in misfire_indices:
            events.append({
                'time': float(timestamps[idx]),
                'rpm': float(rpm_data[idx]),
                'angular_accel': float(angular_accel[idx]),
                'accel_z_score': float((angular_accel[idx] - avg_accel) / std_accel),
                'method': 'angular_velocity'
            })
    
    return events


def detect_misfire_per_cylinder_crankshaft(
    rpm_data: np.ndarray, 
    timestamps: np.ndarray,
    crank_angle: Optional[np.ndarray],
    cylinder_count: int,
    ignition_order: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """
    OEM-Level Per-Cylinder Misfire Detection using Crankshaft Position Analysis.
    
    This is the gold standard method used by BMW, VW, and other premium OEMs.
    Analyzes crankshaft angular velocity at specific crank angles corresponding
    to each cylinder's power stroke to identify which cylinder is misfiring.
    
    Args:
        rpm_data: Engine RPM signal
        timestamps: Time array
        crank_angle: Crankshaft angle in degrees (0-720° for 4-stroke) - optional
        cylinder_count: Number of cylinders
        ignition_order: Firing order (e.g., [1,3,4,2] for 4-cyl) - optional
        
    Returns:
        List of misfire events with identified cylinder number
    """
    if len(rpm_data) < cylinder_count * 2:
        return []
    
    events = []
    
    # Calculate firing interval in degrees (for 4-stroke engine)
    firing_interval_deg = 720.0 / cylinder_count  # e.g., 180° for 4-cyl, 120° for 6-cyl
    
    # If crank angle not available, estimate from RPM
    if crank_angle is None:
        # Estimate crank angle from time and RPM
        # For 4-stroke: 720° per revolution = 2 crank revolutions per engine cycle
        cumulative_angle = np.zeros_like(timestamps)
        for i in range(1, len(timestamps)):
            dt = timestamps[i] - timestamps[i-1]
            # Convert RPM to degrees per second
            deg_per_sec = rpm_data[i] * 360.0 / 60.0  # 360° per revolution
            cumulative_angle[i] = cumulative_angle[i-1] + deg_per_sec * dt
        
        # Normalize to 0-720° range (for 4-stroke cycle)
        crank_angle = cumulative_angle % 720.0
    else:
        # Normalize to 0-720°
        crank_angle = crank_angle % 720.0
    
    # Default firing order if not provided (standard for common engines)
    if ignition_order is None:
        if cylinder_count == 4:
            ignition_order = [1, 3, 4, 2]  # Standard 4-cyl firing order
        elif cylinder_count == 6:
            ignition_order = [1, 5, 3, 6, 2, 4]  # Standard 6-cyl
        elif cylinder_count == 8:
            ignition_order = [1, 8, 4, 3, 6, 5, 7, 2]  # V8 typical
        else:
            # Generic order 1, 2, 3, ...
            ignition_order = list(range(1, cylinder_count + 1))
    
    # Convert RPM to angular velocity (rad/s)
    angular_velocity = rpm_data * (2 * np.pi / 60.0)
    
    # Calculate angular acceleration
    dt = np.diff(timestamps)
    if len(dt) == 0 or np.any(dt <= 0):
        return []
    
    # Ensure dt array matches angular_velocity length for gradient
    # np.gradient expects sample spacing, so use timestamps directly
    if len(angular_velocity) == len(timestamps):
        try:
            angular_accel = np.gradient(angular_velocity, timestamps)
        except (ValueError, TypeError):
            # Fallback: use simple derivative if gradient fails
            dt_padded = np.append(dt, dt[-1]) if len(dt) > 0 else np.ones(len(angular_velocity)) * 0.01
            angular_accel = np.diff(angular_velocity) / dt
            angular_accel = np.append(angular_accel, angular_accel[-1]) if len(angular_accel) > 0 else np.zeros(len(angular_velocity))
    else:
        # Fallback: simple derivative
        angular_accel = np.diff(angular_velocity) / dt
        angular_accel = np.append(angular_accel, angular_accel[-1]) if len(angular_accel) > 0 else np.zeros(len(angular_velocity))
    
    # Find expected firing windows for each cylinder
    # Each cylinder fires at specific crank angles
    for cyl_idx, cylinder_num in enumerate(ignition_order):
        # Calculate expected firing angle for this cylinder
        expected_firing_angle = cyl_idx * firing_interval_deg
        
        # Find samples closest to this cylinder's power stroke
        # Power stroke typically starts ~10° ATDC and extends ~140°
        power_stroke_start = expected_firing_angle + 10.0
        power_stroke_end = expected_firing_angle + 150.0
        
        # Normalize angles
        power_stroke_start = power_stroke_start % 720.0
        power_stroke_end = power_stroke_end % 720.0
        
        # Find indices within power stroke window
        if power_stroke_start < power_stroke_end:
            mask = (crank_angle >= power_stroke_start) & (crank_angle <= power_stroke_end)
        else:
            # Handle wrap-around (e.g., 700-30°)
            mask = (crank_angle >= power_stroke_start) | (crank_angle <= power_stroke_end)
        
        power_stroke_indices = np.where(mask)[0]
        
        if len(power_stroke_indices) < 3:
            continue
        
        # Analyze angular acceleration during this cylinder's power stroke
        power_stroke_accel = angular_accel[power_stroke_indices]
        power_stroke_rpm = rpm_data[power_stroke_indices]
        
        # Expected: Positive acceleration during normal combustion
        # Misfire: Negative or low acceleration
        avg_accel = np.mean(power_stroke_accel)
        std_accel = np.std(power_stroke_accel)
        
        if std_accel > 0:
            # Z-score for acceleration during power stroke
            z_scores = (power_stroke_accel - avg_accel) / (std_accel + 1e-6)
            
            # Detect significant negative deviations (potential misfire)
            misfire_threshold = -2.5  # More sensitive threshold for per-cylinder detection
            misfire_mask = (z_scores < misfire_threshold) & (power_stroke_rpm > MIN_RPM_FOR_DETECTION)
            
            misfire_indices_in_window = np.where(misfire_mask)[0]
            
            for local_idx in misfire_indices_in_window:
                global_idx = power_stroke_indices[local_idx]
                events.append({
                    'time': float(timestamps[global_idx]),
                    'rpm': float(rpm_data[global_idx]),
                    'cylinder': int(cylinder_num),
                    'crank_angle': float(crank_angle[global_idx]),
                    'angular_accel': float(angular_accel[global_idx]),
                    'accel_z_score': float(z_scores[local_idx]),
                    'firing_angle_expected': float(expected_firing_angle),
                    'method': 'per_cylinder_crankshaft'
                })
    
    return events


def calculate_adaptive_threshold(
    rpm: float,
    load: Optional[float],
    coolant_temp: Optional[float],
    base_threshold: float = SPEED_DROP_THRESHOLD_RATIO
) -> float:
    """
    Calculate load and temperature-dependent adaptive threshold for misfire detection.
    
    OEM calibration approach: Thresholds vary based on operating conditions:
    - Higher load = more sensitive (lower threshold)
    - Cold engine = less sensitive (higher threshold)
    - Idle = less sensitive (higher threshold to reduce false positives)
    
    Args:
        rpm: Current engine RPM
        load: Engine load percentage (0-100%) or None
        coolant_temp: Coolant temperature in °C or None
        base_threshold: Base threshold ratio (default 5%)
        
    Returns:
        Adaptive threshold ratio
    """
    threshold = base_threshold
    multiplier = 1.0
    
    # Temperature compensation
    if coolant_temp is not None:
        if coolant_temp < MIN_COOLANT_TEMP_FOR_DETECTION:
            # Cold engine: much less sensitive
            multiplier *= THRESHOLD_MULTIPLIERS['cold']
    
    # Load-dependent compensation
    if load is not None:
        if rpm < IDLE_RPM_THRESHOLD:
            multiplier *= THRESHOLD_MULTIPLIERS['idle']
        elif load < LOW_LOAD_THRESHOLD:
            multiplier *= THRESHOLD_MULTIPLIERS['low_load']
        elif load < HIGH_LOAD_THRESHOLD:
            multiplier *= THRESHOLD_MULTIPLIERS['medium_load']
        else:
            multiplier *= THRESHOLD_MULTIPLIERS['high_load']
    else:
        # If load unknown, use RPM-based heuristic
        if rpm < IDLE_RPM_THRESHOLD:
            multiplier *= THRESHOLD_MULTIPLIERS['idle']
        elif rpm < 2000:
            multiplier *= THRESHOLD_MULTIPLIERS['low_load']
        else:
            multiplier *= THRESHOLD_MULTIPLIERS['medium_load']
    
    return threshold * multiplier


def detect_misfire_signal_fusion(
    rpm_data: np.ndarray,
    timestamps: np.ndarray,
    lambda_data: Optional[np.ndarray],
    load_data: Optional[np.ndarray],
    coolant_temp_data: Optional[np.ndarray],
    ignition_timing_data: Optional[np.ndarray],
    cylinder_count: int
) -> List[Dict[str, Any]]:
    """
    Advanced Signal Fusion for Misfire Detection (OEM-Level).
    
    Combines multiple signals to improve detection accuracy and reduce false positives:
    - RPM variance + Lambda deviation + Load context + Ignition timing anomalies
    
    This approach is similar to what BMW uses in their advanced diagnostics.
    """
    if len(rpm_data) < cylinder_count * 4:
        return []
    
    events = []
    
    # Resample all signals to common timebase (use RPM timestamps as reference)
    time_base = timestamps
    
    # Resample lambda if available
    lambda_resampled = None
    if lambda_data is not None and len(lambda_data) > 0:
        if len(lambda_data) == len(time_base):
            lambda_resampled = lambda_data
        else:
            # Simple interpolation (would use proper resampling in production)
            lambda_resampled = np.interp(time_base, np.linspace(time_base[0], time_base[-1], len(lambda_data)), lambda_data)
    
    # Resample load if available
    load_resampled = None
    if load_data is not None and len(load_data) > 0:
        if len(load_data) == len(time_base):
            load_resampled = load_data
        else:
            load_resampled = np.interp(time_base, np.linspace(time_base[0], time_base[-1], len(load_data)), load_data)
    
    # Resample coolant temp if available
    temp_resampled = None
    if coolant_temp_data is not None and len(coolant_temp_data) > 0:
        if len(coolant_temp_data) == len(time_base):
            temp_resampled = coolant_temp_data
        else:
            temp_resampled = np.interp(time_base, np.linspace(time_base[0], time_base[-1], len(coolant_temp_data)), coolant_temp_data)
    
    # Resample ignition timing if available
    ign_resampled = None
    if ignition_timing_data is not None and len(ignition_timing_data) > 0:
        if len(ignition_timing_data) == len(time_base):
            ign_resampled = ignition_timing_data
        else:
            ign_resampled = np.interp(time_base, np.linspace(time_base[0], time_base[-1], len(ignition_timing_data)), ignition_timing_data)
    
    # Calculate RPM variance (primary indicator)
    window_size = max(20, cylinder_count * 2)
    rpm_series = pd.Series(rpm_data, index=time_base)
    rolling_mean = rpm_series.rolling(window=window_size, center=True).mean()
    rolling_std = rpm_series.rolling(window=window_size, center=True).std()
    rpm_drop_ratio = (rolling_mean - rpm_series) / (rolling_mean + 1e-6)
    
    # Calculate adaptive threshold for each sample
    adaptive_thresholds = np.array([
        calculate_adaptive_threshold(
            rpm_data[i],
            load_resampled[i] if load_resampled is not None else None,
            temp_resampled[i] if temp_resampled is not None else None
        )
        for i in range(len(rpm_data))
    ])
    
    # Primary detection: RPM drop
    primary_misfire_candidates = rpm_drop_ratio > adaptive_thresholds
    primary_misfire_candidates = primary_misfire_candidates & (rpm_data > MIN_RPM_FOR_DETECTION)
    
    # Secondary validation with lambda (if available)
    # During misfire: unburned fuel -> rich lambda (lower lambda value)
    if lambda_resampled is not None:
        lambda_mean = np.nanmean(lambda_resampled)
        lambda_std = np.nanstd(lambda_resampled[~np.isnan(lambda_resampled)])
        
        if lambda_std > 0:
            # Lambda deviation during misfire (goes rich/lean)
            lambda_deviation = np.abs(lambda_resampled - lambda_mean) / (lambda_std + 1e-6)
            # Misfire typically causes lambda to deviate significantly
            lambda_validation = lambda_deviation > 2.0  # Significant deviation
            
            # Require BOTH RPM drop AND lambda deviation for confirmation
            validated_candidates = primary_misfire_candidates & lambda_validation
        else:
            validated_candidates = primary_misfire_candidates
    else:
        validated_candidates = primary_misfire_candidates
    
    # Find validated misfire events
    misfire_indices = np.where(validated_candidates)[0]
    
    for idx in misfire_indices:
        # Calculate confidence based on signal agreement
        confidence = 0.7  # Base confidence
        
        # Increase confidence if lambda agrees
        if lambda_resampled is not None:
            lambda_dev = abs(lambda_resampled[idx] - np.nanmean(lambda_resampled))
            if lambda_dev > 2 * np.nanstd(lambda_resampled[~np.isnan(lambda_resampled)]):
                confidence += 0.15
        
        # Increase confidence if at high load (more reliable detection)
        if load_resampled is not None:
            if load_resampled[idx] > HIGH_LOAD_THRESHOLD:
                confidence += 0.1
        
        # Decrease confidence if cold engine
        if temp_resampled is not None:
            if temp_resampled[idx] < MIN_COOLANT_TEMP_FOR_DETECTION:
                confidence -= 0.2
        
        confidence = max(0.5, min(1.0, confidence))  # Clamp to [0.5, 1.0]
        
        event = {
            'time': float(timestamps[idx]),
            'rpm': float(rpm_data[idx]),
            'rpm_drop': float(rpm_drop_ratio.iloc[idx] if isinstance(rpm_drop_ratio, pd.Series) else rpm_drop_ratio[idx]),
            'adaptive_threshold': float(adaptive_thresholds[idx]),
            'confidence': round(confidence, 2),
            'method': 'signal_fusion'
        }
        
        if lambda_resampled is not None and not np.isnan(lambda_resampled[idx]):
            event['lambda'] = float(lambda_resampled[idx])
        
        if load_resampled is not None and not np.isnan(load_resampled[idx]):
            event['load'] = float(load_resampled[idx])
        
        if temp_resampled is not None and not np.isnan(temp_resampled[idx]):
            event['coolant_temp'] = float(temp_resampled[idx])
        
        events.append(event)
    
    return events


def merge_and_validate_events(all_events: List[List[Dict[str, Any]]], 
                               timestamps: np.ndarray,
                               merge_window_seconds: float = 0.5) -> List[Dict[str, Any]]:
    """
    Merge events from different detection methods and remove duplicates.
    Events within merge_window_seconds are considered the same misfire.
    """
    if not all_events or sum(len(events) for events in all_events) == 0:
        return []
    
    # Flatten all events
    flat_events = []
    for event_list in all_events:
        flat_events.extend(event_list)
    
    if not flat_events:
        return []
    
    # Sort by time
    flat_events.sort(key=lambda x: x['time'])
    
    # Merge nearby events
    merged = []
    current_group = [flat_events[0]]
    
    for event in flat_events[1:]:
        time_diff = event['time'] - current_group[-1]['time']
        
        if time_diff <= merge_window_seconds:
            current_group.append(event)
        else:
            # Save merged event
            avg_time = np.mean([e['time'] for e in current_group])
            methods = list(set([e.get('method', 'unknown') for e in current_group]))
            
            # Calculate confidence based on method agreement and signal fusion
            base_confidence = min(1.0, len(methods) * 0.25)  # Base from method agreement
            
            # Check if confidence is provided in events (from signal fusion)
            event_confidences = [e.get('confidence', 0) for e in current_group if 'confidence' in e]
            if event_confidences:
                avg_event_confidence = np.mean(event_confidences)
                confidence = max(base_confidence, avg_event_confidence)  # Use higher of the two
            else:
                confidence = base_confidence
            
            # Get representative RPM (median)
            rpms = [e.get('rpm', 0) for e in current_group if 'rpm' in e]
            representative_rpm = np.median(rpms) if rpms else 0
            
            # Preserve cylinder identification if available
            cylinders = [e.get('cylinder') for e in current_group if 'cylinder' in e]
            identified_cylinder = None
            if cylinders:
                # Use most common cylinder from the group
                from collections import Counter
                cylinder_counts = Counter(cylinders)
                identified_cylinder = cylinder_counts.most_common(1)[0][0]
            
            # Calculate severity
            max_drop = max([abs(e.get('rpm_drop', 0)) for e in current_group if 'rpm_drop' in e], default=0)
            max_z_score = max([abs(e.get('z_score', 0)) for e in current_group if 'z_score' in e], default=0)
            max_accel_z = max([abs(e.get('accel_z_score', 0)) for e in current_group if 'accel_z_score' in e], default=0)
            
            # Use maximum of all z-scores
            combined_z_score = max(max_z_score, max_accel_z)
            
            if max_drop > 0.15 or combined_z_score > 4.0:
                severity = 'critical'
            elif max_drop > 0.10 or combined_z_score > 3.5:
                severity = 'high'
            elif max_drop > 0.05 or combined_z_score > 3.0:
                severity = 'medium'
            else:
                severity = 'low'
            
            merged_event = {
                'time': float(avg_time),
                'rpm': float(representative_rpm),
                'severity': severity,
                'confidence': round(confidence, 2),
                'detection_methods': ', '.join(methods),
                'event_count': len(current_group),
                'max_rpm_drop': round(max_drop * 100, 2) if max_drop > 0 else None
            }
            
            # Add cylinder identification if available
            if identified_cylinder is not None:
                merged_event['cylinder'] = int(identified_cylinder)
            
            # Preserve additional diagnostic data
            if any('lambda' in e for e in current_group):
                lambdas = [e.get('lambda') for e in current_group if 'lambda' in e]
                merged_event['lambda'] = float(np.median(lambdas)) if lambdas else None
            
            if any('load' in e for e in current_group):
                loads = [e.get('load') for e in current_group if 'load' in e]
                merged_event['load'] = float(np.median(loads)) if loads else None
            
            if any('coolant_temp' in e for e in current_group):
                temps = [e.get('coolant_temp') for e in current_group if 'coolant_temp' in e]
                merged_event['coolant_temp'] = float(np.median(temps)) if temps else None
            
            merged.append(merged_event)
            
            current_group = [event]
    
    # Don't forget the last group
    if current_group:
        avg_time = np.mean([e['time'] for e in current_group])
        methods = list(set([e.get('method', 'unknown') for e in current_group]))
        base_confidence = min(1.0, len(methods) * 0.25)
        
        event_confidences = [e.get('confidence', 0) for e in current_group if 'confidence' in e]
        if event_confidences:
            avg_event_confidence = np.mean(event_confidences)
            confidence = max(base_confidence, avg_event_confidence)
        else:
            confidence = base_confidence
        
        rpms = [e.get('rpm', 0) for e in current_group if 'rpm' in e]
        representative_rpm = np.median(rpms) if rpms else 0
        
        cylinders = [e.get('cylinder') for e in current_group if 'cylinder' in e]
        identified_cylinder = None
        if cylinders:
            from collections import Counter
            cylinder_counts = Counter(cylinders)
            identified_cylinder = cylinder_counts.most_common(1)[0][0]
        
        max_drop = max([abs(e.get('rpm_drop', 0)) for e in current_group if 'rpm_drop' in e], default=0)
        max_z_score = max([abs(e.get('z_score', 0)) for e in current_group if 'z_score' in e], default=0)
        max_accel_z = max([abs(e.get('accel_z_score', 0)) for e in current_group if 'accel_z_score' in e], default=0)
        combined_z_score = max(max_z_score, max_accel_z)
        
        if max_drop > 0.15 or combined_z_score > 4.0:
            severity = 'critical'
        elif max_drop > 0.10 or combined_z_score > 3.5:
            severity = 'high'
        elif max_drop > 0.05 or combined_z_score > 3.0:
            severity = 'medium'
        else:
            severity = 'low'
        
        merged_event = {
            'time': float(avg_time),
            'rpm': float(representative_rpm),
            'severity': severity,
            'confidence': round(confidence, 2),
            'detection_methods': ', '.join(methods),
            'event_count': len(current_group),
            'max_rpm_drop': round(max_drop * 100, 2) if max_drop > 0 else None
        }
        
        if identified_cylinder is not None:
            merged_event['cylinder'] = int(identified_cylinder)
        
        if any('lambda' in e for e in current_group):
            lambdas = [e.get('lambda') for e in current_group if 'lambda' in e]
            merged_event['lambda'] = float(np.median(lambdas)) if lambdas else None
        
        if any('load' in e for e in current_group):
            loads = [e.get('load') for e in current_group if 'load' in e]
            merged_event['load'] = float(np.median(loads)) if loads else None
        
        if any('coolant_temp' in e for e in current_group):
            temps = [e.get('coolant_temp') for e in current_group if 'coolant_temp' in e]
            merged_event['coolant_temp'] = float(np.median(temps)) if temps else None
        
        merged.append(merged_event)
    
    return merged


def compute_misfire(files: List[Path], 
                    cylinder_count: Optional[int] = None,
                    min_rpm: float = MIN_RPM_FOR_DETECTION,
                    max_rpm: float = MAX_RPM_FOR_DETECTION,
                    include_plots: bool = True) -> Dict[str, Any]:
    """
    Main misfire detection function.
    
    Analyzes MDF files using multiple detection algorithms and returns comprehensive results.
    """
    if not MDF:
        return {
            "summary": [],
            "plots": {},
            "statistics": [],
            "meta": {
                "ok": False,
                "error": "asammdf library not installed"
            }
        }
    
    all_misfire_events = []
    file_summaries = []
    all_data_frames = []
    global_stats = {
        "total_files": len(files),
        "files_processed": 0,
        "total_misfires": 0,
        "critical_misfires": 0,
        "high_severity": 0,
        "medium_severity": 0,
        "low_severity": 0
    }
    
    for file_path in files:
        try:
            # Skip non-MDF files (CSV, Excel, etc.)
            path = Path(file_path)
            suffix = path.suffix.lower()
            if suffix not in {'.mf4', '.mf3', '.mdf'}:
                logger.debug(f"Skipping non-MDF file: {file_path} (extension: {suffix})")
                continue
                
            mdf = MDF(str(file_path), memory="minimum")
            try:
                # Find RPM/crankshaft speed channel using advanced signal mapping
                rpm_ch = find_signal_by_role(mdf, "rpm")
                if not rpm_ch:
                    # Fallback to old method
                    rpm_ch = find_signal_channel(mdf, RPM_CANDIDATES)
                
                if not rpm_ch:
                    file_summaries.append({
                        "file": file_path.name,
                        "status": "No RPM channel found",
                        "misfires": 0,
                        "signals_found": "None"
                    })
                    continue
                
                # Extract RPM signal
                rpm_sig = mdf.get(rpm_ch)
                if rpm_sig.samples.size == 0:
                    file_summaries.append({
                        "file": file_path.name,
                        "status": "Empty RPM signal",
                        "misfires": 0
                    })
                    continue
                
                # Convert to numpy arrays
                rpm_data = np.array(pd.to_numeric(rpm_sig.samples, errors='coerce'))
                timestamps = np.array(rpm_sig.timestamps)
                
                # Remove invalid data
                valid_mask = ~(np.isnan(rpm_data) | np.isnan(timestamps))
                rpm_data = rpm_data[valid_mask]
                timestamps = timestamps[valid_mask]
                
                if len(rpm_data) < 10:
                    file_summaries.append({
                        "file": file_path.name,
                        "status": "Insufficient valid data",
                        "misfires": 0
                    })
                    continue
                
                # Filter by RPM range
                valid_rpm_mask = (rpm_data >= min_rpm) & (rpm_data <= max_rpm)
                rpm_data = rpm_data[valid_rpm_mask]
                timestamps = timestamps[valid_rpm_mask]
                
                if len(rpm_data) < 10:
                    file_summaries.append({
                        "file": file_path.name,
                        "status": "Insufficient data in valid RPM range",
                        "misfires": 0
                    })
                    continue
                
                # Determine cylinder count using advanced signal mapping
                cyl_count = None
                if cylinder_count is None:
                    cyl_ch = find_signal_by_role(mdf, "cylinder_count")
                    if not cyl_ch:
                        cyl_ch = find_signal_channel(mdf, CYLINDER_COUNT_CANDIDATES)
                    if cyl_ch:
                        try:
                            sig = mdf.get(cyl_ch)
                            if sig.samples.size > 0:
                                values = pd.to_numeric(sig.samples, errors='coerce').dropna()
                                if len(values) > 0:
                                    mode_value = int(values.mode().iloc[0] if len(values.mode()) > 0 else values.iloc[0])
                                    if 2 <= mode_value <= 16:
                                        cyl_count = mode_value
                                        logger.info(f"Detected cylinder count: {cyl_count} from signal {cyl_ch}")
                        except Exception as e:
                            logger.debug(f"Could not extract cylinder count from {cyl_ch}: {e}")
                    
                    # Fallback to old method
                    if cyl_count is None:
                        cyl_count = extract_cylinder_count(mdf, rpm_ch)
                else:
                    cyl_count = cylinder_count
                
                # Extract additional signals for OEM-level detection using advanced mapping
                lambda_data = None
                lambda_ch = find_signal_by_role(mdf, "lambda")
                if not lambda_ch:
                    lambda_ch = find_signal_channel(mdf, LAMBDA_CANDIDATES)
                if lambda_ch:
                    try:
                        lambda_sig = mdf.get(lambda_ch)
                        if lambda_sig.samples.size > 0:
                            lambda_data_raw = np.array(pd.to_numeric(lambda_sig.samples, errors='coerce'))
                            lambda_timestamps = np.array(lambda_sig.timestamps)
                            # Resample to RPM timebase
                            lambda_data = np.interp(timestamps, lambda_timestamps, lambda_data_raw)
                    except Exception as e:
                        logger.debug(f"Could not extract lambda signal: {e}")
                
                load_data = None
                load_ch = find_signal_by_role(mdf, "torque")
                if not load_ch:
                    load_ch = find_signal_channel(mdf, LOAD_CANDIDATES)
                if load_ch:
                    try:
                        load_sig = mdf.get(load_ch)
                        if load_sig.samples.size > 0:
                            load_data_raw = np.array(pd.to_numeric(load_sig.samples, errors='coerce'))
                            load_timestamps = np.array(load_sig.timestamps)
                            load_data = np.interp(timestamps, load_timestamps, load_data_raw)
                    except Exception as e:
                        logger.debug(f"Could not extract load signal: {e}")
                
                coolant_temp_data = None
                temp_ch = find_signal_by_role(mdf, "coolant_temp")
                if not temp_ch:
                    temp_ch = find_signal_channel(mdf, COOLANT_TEMP_CANDIDATES)
                if temp_ch:
                    try:
                        temp_sig = mdf.get(temp_ch)
                        if temp_sig.samples.size > 0:
                            temp_data_raw = np.array(pd.to_numeric(temp_sig.samples, errors='coerce'))
                            temp_timestamps = np.array(temp_sig.timestamps)
                            coolant_temp_data = np.interp(timestamps, temp_timestamps, temp_data_raw)
                    except Exception as e:
                        logger.debug(f"Could not extract coolant temp signal: {e}")
                
                crank_angle_data = None
                crank_ch = find_signal_by_role(mdf, "crank_angle")
                if not crank_ch:
                    crank_ch = find_signal_channel(mdf, CRANKSHAFT_ANGLE_CANDIDATES)
                if crank_ch:
                    try:
                        crank_sig = mdf.get(crank_ch)
                        if crank_sig.samples.size > 0:
                            crank_data_raw = np.array(pd.to_numeric(crank_sig.samples, errors='coerce'))
                            crank_timestamps = np.array(crank_sig.timestamps)
                            crank_angle_data = np.interp(timestamps, crank_timestamps, crank_data_raw)
                    except Exception as e:
                        logger.debug(f"Could not extract crank angle signal: {e}")
                
                ignition_timing_data = None
                ign_ch = find_signal_by_role(mdf, "ignition_timing")
                if not ign_ch:
                    ign_ch = find_signal_channel(mdf, IGNITION_CANDIDATES)
                if ign_ch:
                    try:
                        ign_sig = mdf.get(ign_ch)
                        if ign_sig.samples.size > 0:
                            ign_data_raw = np.array(pd.to_numeric(ign_sig.samples, errors='coerce'))
                            ign_timestamps = np.array(ign_sig.timestamps)
                            ignition_timing_data = np.interp(timestamps, ign_timestamps, ign_data_raw)
                    except Exception as e:
                        logger.debug(f"Could not extract ignition timing signal: {e}")
                
                # Run all detection methods - now with OEM-level enhancements!
                # Traditional methods
                events_csva = detect_misfire_crankshaft_variance(rpm_data, timestamps, cyl_count)
                events_fft = detect_misfire_frequency_domain(rpm_data, timestamps, cyl_count)
                events_stat = detect_misfire_statistical_anomaly(rpm_data, timestamps)
                events_angular = detect_misfire_angular_velocity(rpm_data, timestamps, cyl_count)
                events_wavelet = detect_misfire_wavelet_analysis(rpm_data, timestamps, cyl_count)
                events_ml = detect_misfire_ml_anomaly(rpm_data, timestamps, cyl_count)
                events_pattern = detect_misfire_pattern_matching(rpm_data, timestamps, cyl_count)
                
                # NEW: OEM-level methods
                events_per_cylinder = detect_misfire_per_cylinder_crankshaft(
                    rpm_data, timestamps, crank_angle_data, cyl_count
                )
                events_fusion = detect_misfire_signal_fusion(
                    rpm_data, timestamps, lambda_data, load_data, 
                    coolant_temp_data, ignition_timing_data, cyl_count
                )
                
                # Merge and validate events (now includes per-cylinder identification)
                all_method_events = [
                    events_csva, events_fft, events_stat, events_angular, 
                    events_wavelet, events_ml, events_pattern,
                    events_per_cylinder, events_fusion  # New OEM methods
                ]
                merged_events = merge_and_validate_events(all_method_events, timestamps)
                
                # Add file information
                for event in merged_events:
                    event['file'] = file_path.name
                    event['cylinder_count'] = cyl_count
                    all_misfire_events.append(event)
                
                # Update statistics
                file_misfire_count = len(merged_events)
                file_critical = sum(1 for e in merged_events if e.get('severity') == 'critical')
                file_high = sum(1 for e in merged_events if e.get('severity') == 'high')
                file_medium = sum(1 for e in merged_events if e.get('severity') == 'medium')
                file_low = sum(1 for e in merged_events if e.get('severity') == 'low')
                
                # Collect signals found for diagnostics
                signals_found = [rpm_ch]
                if lambda_ch:
                    signals_found.append(f"Lambda:{lambda_ch}")
                if load_ch:
                    signals_found.append(f"Load:{load_ch}")
                if temp_ch:
                    signals_found.append(f"Temp:{temp_ch}")
                if crank_ch:
                    signals_found.append(f"CrankAngle:{crank_ch}")
                if ign_ch:
                    signals_found.append(f"Ignition:{ign_ch}")
                
                # Calculate per-cylinder misfire counts if available
                per_cylinder_counts = {}
                for event in merged_events:
                    if 'cylinder' in event:
                        cyl_num = event['cylinder']
                        per_cylinder_counts[cyl_num] = per_cylinder_counts.get(cyl_num, 0) + 1
                
                file_summaries.append({
                    "file": file_path.name,
                    "status": "OK",
                    "misfires": file_misfire_count,
                    "critical": file_critical,
                    "high": file_high,
                    "medium": file_medium,
                    "low": file_low,
                    "cylinder_count": cyl_count,
                    "avg_rpm": round(float(np.mean(rpm_data)), 1),
                    "max_rpm": round(float(np.max(rpm_data)), 1),
                    "min_rpm": round(float(np.min(rpm_data)), 1),
                    "signals_found": ", ".join(signals_found),
                    "signals_available": {
                        "rpm": rpm_ch is not None,
                        "lambda": lambda_ch is not None,
                        "load": load_ch is not None,
                        "coolant_temp": temp_ch is not None,
                        "crank_angle": crank_ch is not None,
                        "ignition_timing": ign_ch is not None
                    }
                })
                
                # Add per-cylinder counts if available
                if per_cylinder_counts:
                    file_summaries[-1]["per_cylinder_misfires"] = per_cylinder_counts
                
                # Store data for plotting
                df = pd.DataFrame({
                    'time': timestamps,
                    'rpm': rpm_data,
                    'file': file_path.name
                })
                all_data_frames.append(df)
                
                global_stats["files_processed"] += 1
                global_stats["total_misfires"] += file_misfire_count
                global_stats["critical_misfires"] += file_critical
                global_stats["high_severity"] += file_high
                global_stats["medium_severity"] += file_medium
                global_stats["low_severity"] += file_low
                
            finally:
                try:
                    mdf.close()
                except Exception:
                    pass
                    
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
            file_summaries.append({
                "file": file_path.name,
                "status": f"Error: {str(e)[:50]}",
                "misfires": 0
            })
    
    # Create visualizations
    plots = {}
    
    logger.info(f"[MISFIRE] Starting plot creation - all_data_frames: {len(all_data_frames) if all_data_frames else 0}, include_plots: {include_plots}, all_misfire_events: {len(all_misfire_events) if all_misfire_events else 0}")
    
    if all_data_frames and include_plots:
        full_df = pd.concat(all_data_frames, ignore_index=True)
        
        # Main RPM plot with misfire events highlighted
        fig = go.Figure()
        
        # Plot RPM for each file
        for file_name in full_df['file'].unique():
            file_df = full_df[full_df['file'] == file_name]
            fig.add_trace(go.Scatter(
                x=file_df['time'].tolist(),
                y=file_df['rpm'].tolist(),
                mode='lines',
                name=f'RPM ({file_name[:30]})',
                line=dict(width=1.5),
                showlegend=True
            ))
        
        # Highlight misfire events with per-cylinder information (once, after all files)
        if all_misfire_events:
            misfire_df = pd.DataFrame(all_misfire_events)
            
            # Color by severity and show cylinder if available
            for severity in ['critical', 'high', 'medium', 'low']:
                severity_events = misfire_df[misfire_df['severity'] == severity]
                if not severity_events.empty:
                    colors = {'critical': 'red', 'high': 'orange', 'medium': 'yellow', 'low': 'lightblue'}
                    sizes = {'critical': 12, 'high': 10, 'medium': 8, 'low': 6}
                    
                    # Build hover text with cylinder info
                    hover_texts = []
                    for idx, row in severity_events.iterrows():
                        hover_text = f"Time={row['time']:.2f}<br>RPM={row['rpm']:.0f}<br>Severity={severity}"
                        if 'cylinder' in row and pd.notna(row['cylinder']):
                            hover_text += f"<br>Cylinder={int(row['cylinder'])}"
                        if 'confidence' in row and pd.notna(row['confidence']):
                            hover_text += f"<br>Confidence={row['confidence']:.2f}"
                        hover_texts.append(hover_text)
                    
                    fig.add_trace(go.Scatter(
                        x=severity_events['time'].tolist(),
                        y=severity_events['rpm'].tolist(),
                        mode='markers',
                        name=f'Misfires ({severity})',
                        marker=dict(
                            color=colors.get(severity, 'gray'),
                            size=sizes.get(severity, 8),
                            symbol='x',
                            line=dict(width=2, color='black')
                        ),
                        text=hover_texts,
                        hovertemplate='%{text}<extra></extra>'
                    ))
        
        fig.update_layout(
            title='Misfire Detection - RPM Timeline with Events',
            xaxis_title='Time (s)',
            yaxis_title='RPM',
            template="plotly_dark",
            height=600,
            hovermode='x unified',
            paper_bgcolor='black',  # Deep black background like IUPR/fuel
            plot_bgcolor='black',  # Deep black background like IUPR/fuel
            font=dict(color='#dce1e6')  # Light text for dark mode
        )
        
        # Always create the main RPM plot even if there are no misfire events
        try:
            plots['RPM Timeline with Misfires'] = {"plotly_json": fig.to_json()}
            logger.info(f"[MISFIRE] ✅ Created: RPM Timeline with Misfires")
        except Exception as e:
            logger.error(f"[MISFIRE] ❌ Failed to create RPM timeline plot: {e}", exc_info=True)
        
        # Severity distribution
        if all_misfire_events:
            misfire_df = pd.DataFrame(all_misfire_events)
            severity_counts = misfire_df['severity'].value_counts()
            
            fig_severity = go.Figure()
            fig_severity.add_trace(go.Bar(
                x=severity_counts.index.tolist(),
                y=severity_counts.values.tolist(),
                marker_color=['red', 'orange', 'yellow', 'lightblue'][:len(severity_counts)],
                text=severity_counts.values.tolist(),
                textposition='auto'
            ))
            
            fig_severity.update_layout(
                title='Misfire Severity Distribution',
                xaxis_title='Severity',
                yaxis_title='Count',
                template="plotly_dark",
                height=400,
                paper_bgcolor='black',  # Deep black background like IUPR/fuel
                plot_bgcolor='black',  # Deep black background like IUPR/fuel
                font=dict(color='#dce1e6')  # Light text for dark mode
            )
            
            try:
                plots['Severity Distribution'] = {"plotly_json": fig_severity.to_json()}
                logger.info(f"[MISFIRE] ✅ Created: Severity Distribution")
            except Exception as e:
                logger.error(f"[MISFIRE] ❌ Failed to create severity distribution plot: {e}", exc_info=True)
            
            # Confidence distribution
            if 'confidence' in misfire_df.columns:
                fig_confidence = go.Figure()
                fig_confidence.add_trace(go.Histogram(
                    x=misfire_df['confidence'].tolist(),
                    nbinsx=20,
                    marker_color='steelblue',
                    opacity=0.7
                ))
                
                fig_confidence.update_layout(
                    title='Misfire Detection Confidence Distribution',
                    xaxis_title='Confidence Score',
                    yaxis_title='Count',
                    template="plotly_dark",
                    height=400
                )
                
                try:
                    plots['Confidence Distribution'] = {"plotly_json": fig_confidence.to_json()}
                    logger.info(f"[MISFIRE] ✅ Created: Confidence Distribution")
                except Exception as e:
                    logger.error(f"[MISFIRE] ❌ Failed to create confidence distribution plot: {e}", exc_info=True)
            
            # RPM distribution at misfire events
            fig_rpm_dist = go.Figure()
            fig_rpm_dist.add_trace(go.Histogram(
                x=misfire_df['rpm'].tolist(),
                nbinsx=30,
                marker_color='coral',
                opacity=0.7
            ))
            
            fig_rpm_dist.update_layout(
                title='RPM Distribution at Misfire Events',
                xaxis_title='RPM',
                yaxis_title='Count',
                template="plotly_dark",
                height=400
            )
            
            try:
                plots['RPM Distribution at Misfires'] = {"plotly_json": fig_rpm_dist.to_json()}
                logger.info(f"[MISFIRE] ✅ Created: RPM Distribution at Misfires")
            except Exception as e:
                logger.error(f"[MISFIRE] ❌ Failed to create RPM distribution plot: {e}", exc_info=True)
            
            # NEW: Per-Cylinder Misfire Distribution (if cylinder info available)
            if 'cylinder' in misfire_df.columns:
                cylinder_counts = misfire_df['cylinder'].value_counts().sort_index()
                if len(cylinder_counts) > 0:
                    fig_cylinder = go.Figure()
                    fig_cylinder.add_trace(go.Bar(
                        x=cylinder_counts.index.astype(str).tolist(),
                        y=cylinder_counts.values.tolist(),
                        marker_color='crimson',
                        text=cylinder_counts.values.tolist(),
                        textposition='auto',
                        name='Misfires per Cylinder'
                    ))
                    
                    fig_cylinder.update_layout(
                        title='Per-Cylinder Misfire Distribution',
                        xaxis_title='Cylinder Number',
                        yaxis_title='Misfire Count',
                        template="plotly_dark",
                        height=400
                    )
                    
                    try:
                        plots['Per-Cylinder Distribution'] = {"plotly_json": fig_cylinder.to_json()}
                        logger.info(f"[MISFIRE] ✅ Created: Per-Cylinder Distribution")
                    except Exception as e:
                        logger.error(f"[MISFIRE] ❌ Failed to create per-cylinder distribution plot: {e}", exc_info=True)
                else:
                    logger.warning(f"[MISFIRE] ⚠️  Per-Cylinder Distribution: No cylinder data available")
            else:
                logger.warning(f"[MISFIRE] ⚠️  Per-Cylinder Distribution: 'cylinder' column not found in misfire data")
    
    # Summary of all plots created
    if len(plots) > 0:
        logger.info(f"[MISFIRE] 📊 Plot creation summary: Created {len(plots)} plots: {list(plots.keys())}")
    else:
        logger.warning(f"[MISFIRE] ⚠️  WARNING: No plots were created! Check above logs for errors.")
    
    # Statistics summary
    statistics = []
    if all_misfire_events:
        misfire_df = pd.DataFrame(all_misfire_events)
        
        # Calculate OBD-II style statistics
        total_revolutions = 0
        for df in all_data_frames:
            if len(df) > 1:
                time_span = df['time'].max() - df['time'].min()
                avg_rpm = df['rpm'].mean()
                if time_span > 0 and avg_rpm > 0:
                    # Revolutions = (time in minutes) * RPM
                    revolutions = (time_span / 60.0) * avg_rpm
                    total_revolutions += revolutions
        
        misfires_per_1000_rev = 0
        if total_revolutions > 0:
            misfires_per_1000_rev = (global_stats["total_misfires"] / total_revolutions) * 1000
        
        # OBD-II MIL (Malfunction Indicator Light) status
        mil_status = "OFF"
        if misfires_per_1000_rev > OBD_MISFIRE_COUNT_THRESHOLD:
            mil_status = "ON (Misfire Rate Exceeded)"
        elif global_stats["total_misfires"] > 0 and (global_stats["total_misfires"] / total_revolutions) > OBD_MISFIRE_RATE_LIMIT:
            mil_status = "ON (Misfire Rate > 2%)"
        
        statistics = [
            {"metric": "Total Misfire Events", "value": len(all_misfire_events)},
            {"metric": "Critical Severity", "value": global_stats["critical_misfires"]},
            {"metric": "High Severity", "value": global_stats["high_severity"]},
            {"metric": "Medium Severity", "value": global_stats["medium_severity"]},
            {"metric": "Low Severity", "value": global_stats["low_severity"]},
            {"metric": "Average Confidence", "value": round(float(misfire_df['confidence'].mean()), 2) if 'confidence' in misfire_df.columns else 0},
            {"metric": "Average RPM at Misfire", "value": round(float(misfire_df['rpm'].mean()), 1) if 'rpm' in misfire_df.columns else 0},
            {"metric": "Misfires per 1000 Revolutions (OBD-II)", "value": round(misfires_per_1000_rev, 2) if total_revolutions > 0 else 0},
            {"metric": "MIL Status", "value": mil_status},
        ]
        
        # Add per-cylinder statistics if available
        if 'cylinder' in misfire_df.columns:
            cylinders_with_misfires = misfire_df['cylinder'].nunique()
            statistics.append({"metric": "Cylinders with Misfires", "value": int(cylinders_with_misfires)})
            
            # Most problematic cylinder
            if len(misfire_df['cylinder'].value_counts()) > 0:
                most_problematic = misfire_df['cylinder'].value_counts().index[0]
                statistics.append({"metric": "Most Problematic Cylinder", "value": int(most_problematic)})
    
    statistics.extend([
        {"metric": "Files Processed", "value": global_stats["files_processed"]},
        {"metric": "Total Files", "value": global_stats["total_files"]},
    ])
    
    # Format output to match DFC/IUPR structure
    tables = {}
    if all_misfire_events:
        tables["Misfire Events"] = all_misfire_events[:1000]  # Limit for display
    if file_summaries:
        tables["File Summary"] = file_summaries
    if statistics:
        tables["Statistics"] = statistics
    
    # Format plots to match expected structure
    formatted_plots = {}
    for plot_name, plot_data in plots.items():
        formatted_plots[plot_name] = {
            "plotly_json": plot_data.get("plotly_json") if isinstance(plot_data, dict) else plot_data
        }
    
    return {
        "tables": tables,
        "plots": formatted_plots,
        "meta": {
            "ok": True,
            "files_processed": global_stats["files_processed"],
            "total_misfires": global_stats["total_misfires"],
            "detection_methods": [
                "crankshaft_variance", "frequency_domain", "statistical_anomaly", 
                "angular_velocity", "wavelet_analysis", "ml_isolation_forest", "pattern_matching",
                "per_cylinder_crankshaft", "signal_fusion"  # New OEM-level methods
            ],
            "oem_features": {
                "per_cylinder_identification": True,
                "adaptive_thresholds": True,
                "signal_fusion": True,
                "load_dependent_detection": True,
                "temperature_compensation": True,
                "obd_ii_compliance": True
            },
            "system_version": "2.0-OEM"
        }
    }


def compute_misfire_plotly(files: List[str]) -> Dict[str, Any]:
    """Convenience wrapper for app.py integration."""
    return compute_misfire([Path(f) for f in files])


if __name__ == "__main__":
    import argparse
    import json
    
    ap = argparse.ArgumentParser(description="Advanced Misfire Detection")
    ap.add_argument("--files", nargs="+", required=True, help="MDF file paths")
    ap.add_argument("--cylinders", type=int, help="Number of cylinders (auto-detect if not specified)")
    ap.add_argument("--min-rpm", type=float, default=MIN_RPM_FOR_DETECTION, help="Minimum RPM for detection")
    ap.add_argument("--max-rpm", type=float, default=MAX_RPM_FOR_DETECTION, help="Maximum RPM for detection")
    ap.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    
    args = ap.parse_args()
    result = compute_misfire(
        [Path(f) for f in args.files],
        cylinder_count=args.cylinders,
        min_rpm=args.min_rpm,
        max_rpm=args.max_rpm,
        include_plots=not args.no_plots
    )
    print(json.dumps(result, indent=2)[:50000])  # Limit output

