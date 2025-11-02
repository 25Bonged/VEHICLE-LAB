#!/usr/bin/env python3
"""
custom_dfc.py — Advanced MDF-native DFC/DTC analyzer for dashboards (compatible with app.py).

Enhanced with advanced DTC analysis features:
- DTC code format parsing (P0/P1/P2/P3, B, C, U codes)
- Status byte decoding (test failed, confirmed, pending, stored, MIL)
- Priority/severity assessment
- Temporal analysis with event segments
- Correlation analysis with other signals
- Freeze frame detection
- Enhanced reporting and visualization

Produces structure:
{
  "summary": [ { 
    "code": int, 
    "DFC_name": str, 
    "row_count": int, 
    "event_count": int, 
    "runtime_count": float,
    "dtc_format": str,      # NEW: e.g., "P0xxx", "P1xxx"
    "code_type": str,       # NEW: "powertrain", "body", "chassis", "network"
    "priority": str,        # NEW: "P0", "P1", "P2", "P3"
    "severity": str,        # NEW: "critical", "high", "medium", "low"
    "segments": [...],      # NEW: time segments where code was active
    "first_seen": float,    # NEW: first occurrence time
    "last_seen": float,     # NEW: last occurrence time
    "max_duration": float,  # NEW: longest continuous duration
    "total_duration": float # NEW: total active duration
  } ],
  "plots": [ {"name": "...", "plotly_json": "..."}, ...],
  "channels": [ {per numDFC channel evidence} ],
  "freeze_frames": [...],  # NEW: freeze frame data
  "correlations": {...},   # NEW: correlations with other signals
  "meta": {}
}
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from asammdf import MDF
except ImportError:
    MDF = None


# ==================== DTC Code Classification & Parsing ====================

def parse_dtc_code(code: int) -> Dict[str, Any]:
    """
    Parse a DTC code into its components according to SAE J2012/J1979 standards.
    
    DTC Format: [Letter][Number][4-digit hex]
    - Letter: P (Powertrain), B (Body), C (Chassis), U (Network)
    - Number: 0 (SAE Standard), 1 (Manufacturer Specific), 2/3 (Reserved/Future)
    
    Args:
        code: Integer DTC code (can be in various formats)
        
    Returns:
        Dictionary with parsed DTC information
    """
    # Convert to hex string for parsing
    code_hex = format(code, '06X') if code >= 0 else format(code & 0xFFFFFFFF, '06X')
    
    # Try standard 2-byte format (most common): high byte + low byte
    # Format: 0xP0XXXX where P is letter index, 0 is priority, XXXX is code
    
    dtc_info = {
        "raw_code": code,
        "hex_code": code_hex,
        "dtc_format": "Unknown",
        "code_type": "Unknown",
        "priority": "Unknown",
        "is_sae_standard": False,
        "is_manufacturer_specific": False,
        "formatted": f"DFC_{code}"
    }
    
    # Method 1: Standard OBD-II DTC format (5-character: P0XXX)
    # Codes are typically stored as 2-byte values where:
    # - Bits 14-15: Code type (0=P, 1=B, 2=C, 3=U)
    # - Bit 13: Priority (0=SAE, 1=Manufacturer)
    # - Bits 0-12: DTC number
    
    if code < 0x10000:  # 2-byte code (most common)
        type_bits = (code >> 14) & 0x3
        priority_bit = (code >> 13) & 0x1
        dtc_number = code & 0x1FFF
        
        type_map = {0: "P", 1: "B", 2: "C", 3: "U"}
        type_names = {"P": "Powertrain", "B": "Body", "C": "Chassis", "U": "Network"}
        
        letter = type_map.get(type_bits, "P")
        priority_num = "0" if priority_bit == 0 else "1"
        
        dtc_info.update({
            "dtc_format": f"{letter}{priority_num}{dtc_number:04X}",
            "code_type": type_names.get(letter, "Powertrain"),
            "priority": f"P{priority_bit}",
            "is_sae_standard": priority_bit == 0,
            "is_manufacturer_specific": priority_bit == 1,
            "formatted": f"{letter}{priority_num}{dtc_number:04X}"
        })
    
    # Method 2: Try parsing as string representation (e.g., "P0123" stored as numeric)
    elif code >= 0x100000:  # 6-digit hex might be string-encoded
        code_str = hex(code)[2:].upper()
        if len(code_str) >= 5 and code_str[0] in "PBCU":
            letter = code_str[0]
            type_names = {"P": "Powertrain", "B": "Body", "C": "Chassis", "U": "Network"}
            priority_num = code_str[1] if len(code_str) > 1 else "0"
            
            dtc_info.update({
                "dtc_format": code_str[:5],
                "code_type": type_names.get(letter, "Powertrain"),
                "priority": f"P{priority_num}",
                "is_sae_standard": priority_num == "0",
                "is_manufacturer_specific": priority_num == "1",
                "formatted": code_str[:5]
            })
    
    return dtc_info


def classify_severity(code_info: Dict[str, Any], event_count: int, 
                     runtime: float, max_duration: float) -> str:
    """
    Classify DTC severity based on multiple factors.
    
    Args:
        code_info: Parsed DTC information
        event_count: Number of times the code occurred
        runtime: Total runtime when code was active
        max_duration: Longest continuous duration
        
    Returns:
        Severity level: "critical", "high", "medium", "low"
    """
    severity_score = 0
    
    # Priority-based scoring
    if code_info.get("priority") == "P0":
        severity_score += 3
    elif code_info.get("priority") == "P1":
        severity_score += 2
    elif code_info.get("priority") == "P2":
        severity_score += 1
    
    # Code type scoring (powertrain issues are typically more critical)
    if code_info.get("code_type") == "Powertrain":
        severity_score += 2
    elif code_info.get("code_type") == "Network":
        severity_score += 1
    
    # Event frequency scoring
    if event_count >= 10:
        severity_score += 3
    elif event_count >= 5:
        severity_score += 2
    elif event_count >= 2:
        severity_score += 1
    
    # Duration scoring
    if max_duration > 100:  # seconds
        severity_score += 2
    elif max_duration > 10:
        severity_score += 1
    
    # Runtime percentage scoring (if runtime > 50% of total)
    if runtime > 300:  # More than 5 minutes total
        severity_score += 2
    elif runtime > 60:
        severity_score += 1
    
    # Determine severity
    if severity_score >= 8:
        return "critical"
    elif severity_score >= 5:
        return "high"
    elif severity_score >= 3:
        return "medium"
    else:
        return "low"


def decode_status_byte(status: int) -> Dict[str, bool]:
    """
    Decode DTC status byte according to ISO 14229/J1979.
    
    Status byte bits (bit 0 = LSB):
    - Bit 0: Test failed
    - Bit 1: Test failed this drive cycle
    - Bit 2: Pending DTC
    - Bit 3: Confirmed DTC
    - Bit 4: Test not completed since DTC cleared
    - Bit 5: Test failed since DTC cleared
    - Bit 6: Test not completed this drive cycle
    - Bit 7: Warning indicator requested
    
    Args:
        status: Status byte value
        
    Returns:
        Dictionary with decoded status flags
    """
    return {
        "test_failed": bool(status & 0x01),
        "test_failed_this_drive": bool(status & 0x02),
        "pending": bool(status & 0x04),
        "confirmed": bool(status & 0x08),
        "test_not_completed_since_clear": bool(status & 0x10),
        "test_failed_since_clear": bool(status & 0x20),
        "test_not_completed_this_drive": bool(status & 0x40),
        "mil_requested": bool(status & 0x80),  # MIL = Malfunction Indicator Lamp
    }


# ==================== Mapping loader ====================

def load_numdfc_mapping(upload_dir: Path = Path("uploads")) -> Dict[int, str]:
    """Load DTC code to name mapping from numdfc.txt file."""
    mapping: Dict[int, str] = {}
    txt_file = upload_dir / "numdfc.txt"
    if not txt_file.exists():
        return mapping
    with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"'(.+?)'\s*->\s*(0x[0-9A-Fa-f]+|\d+)", line)
            if m:
                try:
                    mapping[int(m.group(2), 0)] = m.group(1)
                except Exception:
                    pass
                continue
            m = re.match(r"(0x[0-9A-Fa-f]+|\d+)\s*->\s*'(.+?)'", line)
            if m:
                try:
                    mapping[int(m.group(1), 0)] = m.group(2)
                except Exception:
                    pass
                continue
            parts = re.split(r"[,\t]", line)
            if len(parts) >= 2:
                try:
                    code = int(parts[0], 0)
                    mapping[code] = parts[1].strip().strip("'\"")
                except Exception:
                    continue
    return mapping


def safe_int(v) -> Optional[int]:
    """Safely convert value to integer."""
    try:
        if isinstance(v, (bytes, bytearray)):
            return int(v.decode("utf-8").strip(), 0)
        return int(v)
    except Exception:
        return None


# ==================== Core analysis (enhanced) ====================

def _analyze_enhanced(files: List[Path], enable_correlation: bool = True):
    """
    Enhanced DTC analysis with temporal tracking, segments, and correlations.
    
    Returns:
        Tuple of analysis results including enhanced event tracking
    """
    if MDF is None:
        raise RuntimeError("asammdf not installed")

    # Basic counters
    row_count: Dict[int, int] = {}
    event_count: Dict[int, int] = {}
    runtime_count: Dict[int, float] = {}
    
    # Enhanced tracking
    code_segments: Dict[int, List[Dict[str, float]]] = {}  # Time segments per code
    code_first_seen: Dict[int, float] = {}
    code_last_seen: Dict[int, float] = {}
    code_status_bytes: Dict[int, List[int]] = {}  # Track status bytes if available
    
    per_channel_evidence: List[Dict[str, Any]] = []
    st_series: Dict[str, tuple] = {}
    dfes_series: Dict[str, tuple] = {}
    
    # Correlation data (store signal values at DTC events)
    correlation_data: Dict[int, Dict[str, List[Any]]] = {}

    def is_num_ch(n: str) -> bool:
        ln = n.lower()
        return ("numdfc" in ln) and not ln.endswith("_st") and ("_st" not in ln)

    def is_dfc_st_ch(n: str) -> bool:
        ln = n.lower()
        return ("dfc_st" in ln) and ("dfes_st" not in ln)

    def is_dfes_num_ch(n: str) -> bool:
        ln = n.lower()
        return ("dfes_numdfc" in ln)
    
    def is_status_ch(n: str) -> bool:
        """Check if channel contains DTC status bytes."""
        ln = n.lower()
        return any(x in ln for x in ["dtc_status", "dfc_status", "statusofdtc", "status_dtc"])

    for path in files:
        if not path.exists():
            continue
        mdf = MDF(str(path))
        try:
            chans = list(mdf.channels_db.keys())
            num_channels = [c for c in chans if is_num_ch(c)]
            st_channels = [c for c in chans if is_dfc_st_ch(c)]
            dfes_channels = [c for c in chans if is_dfes_num_ch(c)]
            status_channels = [c for c in chans if is_status_ch(c)] if enable_correlation else []
            
            # Try to find correlation signals (RPM, speed, temp, etc.)
            correlation_signals = {}
            if enable_correlation:
                correlation_candidates = ["rpm", "nEng", "EngineSpeed", "VehicleSpeed", 
                                       "VehSpd", "CoolantTemp", "ECT", "EngineTemp",
                                       "Load", "Torque", "Throttle", "MAP"]
                for cand in correlation_candidates:
                    matching = [c for c in chans if cand.lower() in c.lower()]
                    if matching:
                        correlation_signals[cand] = matching[0]

            # numDFC-like channels (main DTC codes)
            for ch in num_channels:
                try:
                    sig = mdf.get(ch)
                except Exception:
                    continue
                if not hasattr(sig, "samples") or not hasattr(sig, "timestamps"):
                    continue
                vals = list(sig.samples)
                times = list(sig.timestamps)
                if not vals or not times:
                    continue

                samples = len(vals)
                nz_rows = 0
                nz_events = 0
                prev_code = 0
                segment_start: Optional[float] = None
                current_segment_code: Optional[int] = None
                
                # Load correlation signal for this channel
                corr_data = {}
                if enable_correlation:
                    for sig_name, sig_ch in correlation_signals.items():
                        try:
                            corr_sig = mdf.get(sig_ch)
                            if hasattr(corr_sig, "samples") and hasattr(corr_sig, "timestamps"):
                                # Interpolate to match DTC timestamps
                                corr_interp = np.interp(
                                    times, 
                                    corr_sig.timestamps, 
                                    corr_sig.samples
                                )
                                corr_data[sig_name] = corr_interp
                        except Exception:
                            pass

                for i, v in enumerate(vals):
                    iv = safe_int(v)
                    if iv is None or iv == 0:
                        # Code cleared - close any active segment
                        if segment_start is not None and current_segment_code is not None:
                            if current_segment_code not in code_segments:
                                code_segments[current_segment_code] = []
                            code_segments[current_segment_code].append({
                                "start": segment_start,
                                "end": times[i],
                                "duration": times[i] - segment_start
                            })
                            segment_start = None
                            current_segment_code = None
                        prev_code = 0
                        continue
                    
                    # Code is active
                    row_count[iv] = row_count.get(iv, 0) + 1
                    nz_rows += 1
                    
                    # Track first/last seen
                    if iv not in code_first_seen:
                        code_first_seen[iv] = times[i]
                    code_last_seen[iv] = times[i]
                    
                    # Start new segment if code changed
                    if iv != prev_code:
                        # Close previous segment if any
                        if segment_start is not None and current_segment_code is not None:
                            if current_segment_code not in code_segments:
                                code_segments[current_segment_code] = []
                            code_segments[current_segment_code].append({
                                "start": segment_start,
                                "end": times[i],
                                "duration": times[i] - segment_start
                            })
                        
                        # Start new segment
                        segment_start = times[i]
                        current_segment_code = iv
                        event_count[iv] = event_count.get(iv, 0) + 1
                        nz_events += 1
                        
                        # Store correlation data at event start
                        if enable_correlation and corr_data:
                            if iv not in correlation_data:
                                correlation_data[iv] = {k: [] for k in corr_data.keys()}
                            for k, v_arr in corr_data.items():
                                if i < len(v_arr):
                                    correlation_data[iv][k].append(float(v_arr[i]))
                    
                    prev_code = iv
                    
                    # Runtime accumulation
                    if i > 0:
                        dt = times[i] - times[i-1]
                        runtime_count[iv] = runtime_count.get(iv, 0.0) + float(dt)
                    else:
                        runtime_count.setdefault(iv, 0.0)
                
                # Close final segment if still open
                if segment_start is not None and current_segment_code is not None:
                    if current_segment_code not in code_segments:
                        code_segments[current_segment_code] = []
                    if times:
                        code_segments[current_segment_code].append({
                            "start": segment_start,
                            "end": times[-1],
                            "duration": times[-1] - segment_start
                        })

                per_channel_evidence.append({
                    "file": path.name,
                    "channel": ch,
                    "samples": samples,
                    "nonzero_rows": nz_rows,
                    "event_runs": nz_events,
                    "runtime_seconds": round(sum(runtime_count.values()), 3),
                })

            # DFC_ST (status signals)
            for ch in st_channels:
                try:
                    sig = mdf.get(ch)
                except Exception:
                    continue
                if not hasattr(sig, "samples") or not hasattr(sig, "timestamps"):
                    continue
                st_series[ch] = (list(sig.timestamps), list(sig.samples))
                
                # Try to decode status bytes if this looks like a status channel
                try:
                    status_vals = [safe_int(x) for x in sig.samples if safe_int(x) is not None]
                    if status_vals and any(v > 0 and v <= 255 for v in status_vals):
                        # This might be status bytes, store them
                        for code in row_count.keys():
                            if code not in code_status_bytes:
                                code_status_bytes[code] = []
                            code_status_bytes[code].extend([v for v in status_vals if 0 < v <= 255])
                except Exception:
                    pass

            # DFES_numDFC
            for ch in dfes_channels:
                try:
                    sig = mdf.get(ch)
                except Exception:
                    continue
                if not hasattr(sig, "samples") or not hasattr(sig, "timestamps"):
                    continue
                dfes_series[ch] = (list(sig.timestamps), list(sig.samples))

        finally:
            try:
                mdf.close()
            except Exception:
                pass

    return (
        row_count, event_count, runtime_count, per_channel_evidence, 
        st_series, dfes_series, code_segments, code_first_seen, code_last_seen,
        code_status_bytes, correlation_data
    )


# Legacy _analyze for backward compatibility
def _analyze(files: List[Path]):
    """Legacy analysis function for backward compatibility."""
    result = _analyze_enhanced(files, enable_correlation=False)
    return result[:6]  # Return only first 6 elements


# ==================== Plot helpers ====================

EDGE_ZOOM = 0.67
BASE_FONT = 11
LEGEND_FONT = 10
TICK_FONT = 10

def _apply_common_layout(fig: go.Figure, *, height: int, title: str) -> None:
    fig.update_layout(
        title=title,
        template="plotly_dark",
        autosize=True,
        uirevision="keep",
        margin=dict(l=60, r=40, t=54, b=120),
        height=height,
        font=dict(size=BASE_FONT),
        legend=dict(orientation="h", y=-0.22, font=dict(size=LEGEND_FONT)),
    )
    fig.update_xaxes(automargin=True, tickfont=dict(size=TICK_FONT))
    fig.update_yaxes(automargin=True, tickfont=dict(size=TICK_FONT))

def _make_summary_plot(summary_df: pd.DataFrame) -> Optional[str]:
    """Enhanced summary plot with severity coloring."""
    if summary_df.empty:
        return None
    df = summary_df.sort_values("event_count", ascending=False).head(25).copy()
    
    # Color mapping for severity
    severity_colors = {
        "critical": "#C30C36",
        "high": "#ff6f8a",
        "medium": "#f28e2b",
        "low": "#4e79a7"
    }
    
    # Add color column
    if "severity" in df.columns:
        df["color"] = df["severity"].map(severity_colors).fillna("#4e79a7")
    else:
        df["color"] = "#4e79a7"
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["DFC_name"], 
        y=df["event_count"], 
        name="Events", 
        marker_color=df["color"] if "color" in df.columns else "#4e79a7"
    ))
    fig.add_trace(go.Bar(
        x=df["DFC_name"], 
        y=df["runtime_count"], 
        name="Runtime (s)", 
        marker_color="#f28e2b",
        opacity=0.7
    ))
    fig.add_trace(go.Bar(
        x=df["DFC_name"], 
        y=df["row_count"], 
        name="Rows", 
        marker_color="#C30C36", 
        opacity=0.65
    ))
    fig.update_layout(
        barmode="group",
        xaxis=dict(title="DFC", tickangle=-35),
        yaxis=dict(title="Counts"),
        uniformtext=dict(minsize=8, mode="show"),
    )
    _apply_common_layout(fig, height=int((520) / EDGE_ZOOM), title="NumDFC Summary (Enhanced)")
    return fig.to_json()

def _make_severity_priority_plot(summary_df: pd.DataFrame) -> Optional[str]:
    """Create a plot showing severity vs priority distribution."""
    if summary_df.empty or "severity" not in summary_df.columns:
        return None
    
    fig = go.Figure()
    
    severity_order = ["critical", "high", "medium", "low"]
    priority_order = ["P0", "P1", "P2", "P3", "Unknown"]
    
    # Count combinations
    counts = {}
    for _, row in summary_df.iterrows():
        sev = row.get("severity", "low")
        pri = row.get("priority", "Unknown")
        key = (sev, pri)
        counts[key] = counts.get(key, 0) + 1
    
    # Create heatmap data
    z_data = []
    y_labels = []
    x_labels = []
    
    for sev in severity_order:
        row_data = []
        for pri in priority_order:
            key = (sev, pri)
            row_data.append(counts.get(key, 0))
        if any(row_data):  # Only add if there's data
            z_data.append(row_data)
            y_labels.append(sev.capitalize())
    
    if z_data:
        x_labels = priority_order
        fig.add_trace(go.Heatmap(
            z=z_data,
            x=x_labels,
            y=y_labels,
            colorscale="Reds",
            showscale=True,
            text=[[str(v) if v > 0 else "" for v in row] for row in z_data],
            texttemplate="%{text}",
            textfont={"size": 12}
        ))
        fig.update_layout(
            title="DTC Severity vs Priority Distribution",
            xaxis_title="Priority",
            yaxis_title="Severity",
            template="plotly_dark",
            height=400
        )
        return fig.to_json()
    return None

def _make_dfc_st_plot(st_series: Dict[str, tuple]) -> Optional[str]:
    """Plot DFC_ST overlay signals."""
    if not st_series:
        return None
    fig = go.Figure()
    gap = 2.0
    for i, (ch, (t, v)) in enumerate(st_series.items()):
        if not t or not v:
            continue
        off = i * gap
        y = [(safe_int(x) or 0) + off for x in v]
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=ch,
                                 line=dict(shape="hv", width=1.8), hoverinfo="x+name+y"))
    n = len(st_series)
    base = 260
    step = 60
    height = int((base + step * min(n, 30)) / EDGE_ZOOM)
    fig.update_layout(xaxis_title="Time (s)", yaxis=dict(title="State (offset per signal)"),
                      uniformtext=dict(minsize=8, mode="show"))
    _apply_common_layout(fig, height=height, title="DFC_ST Overlay")
    return fig.to_json()

def _make_dfes_numdfc_plot(dfes_series: Dict[str, tuple], mapping: Dict[int, str]) -> Optional[str]:
    """Plot DFES_numDFC overlay."""
    if not dfes_series:
        return None
    fig = go.Figure()
    gap = 4.0
    for i, (ch, (t, v)) in enumerate(dfes_series.items()):
        if not t or not v:
            continue
        off = i * gap
        y = [(safe_int(x) or 0) + off for x in v]
        text = [mapping.get(safe_int(x), f"DFC_{safe_int(x)}") if safe_int(x) else "" for x in v]
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=ch, text=text,
                                 line=dict(shape="hv", width=1.5),
                                 hovertemplate="Time=%{x}<br>Code=%{text}<extra>%{name}</extra>"))
    n = len(dfes_series)
    base = 260
    step = 70
    height = int((base + step * min(n, 30)) / EDGE_ZOOM)
    fig.update_layout(xaxis_title="Time (s)", yaxis=dict(title="DFES_numDFC (offset per signal)"),
                      uniformtext=dict(minsize=8, mode="show"))
    _apply_common_layout(fig, height=height, title="DFES_numDFC Overlay")
    return fig.to_json()

def _make_temporal_timeline_plot(code_segments: Dict[int, List[Dict[str, float]]], 
                                 mapping: Dict[int, str], 
                                 max_codes: int = 15) -> Optional[str]:
    """Create a timeline plot showing when each DTC was active."""
    if not code_segments:
        return None
    
    fig = go.Figure()
    
    # Sort codes by total duration
    code_durations = {
        code: sum(seg["duration"] for seg in segments)
        for code, segments in code_segments.items()
    }
    sorted_codes = sorted(code_durations.items(), key=lambda x: x[1], reverse=True)[:max_codes]
    
    for y_idx, (code, total_dur) in enumerate(sorted_codes):
        segments = code_segments[code]
        code_name = mapping.get(code, f"DFC_{code}")
        
        for seg in segments:
            fig.add_trace(go.Scatter(
                x=[seg["start"], seg["end"], seg["end"], seg["start"], seg["start"]],
                y=[y_idx, y_idx, y_idx + 0.8, y_idx + 0.8, y_idx],
                fill="toself",
                mode="lines",
                name=code_name,
                line=dict(width=0),
                fillcolor=f"rgba(195,12,54,{min(0.6, 0.3 + len(segments) * 0.05)})",
                showlegend=(y_idx == 0),  # Only show legend for first code
                hovertemplate=f"Code: {code_name}<br>Start: {seg['start']:.2f}s<br>End: {seg['end']:.2f}s<br>Duration: {seg['duration']:.2f}s<extra></extra>"
            ))
    
    fig.update_layout(
        title="DTC Active Timeline",
        xaxis_title="Time (s)",
        yaxis_title="DTC Code",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(sorted_codes))),
            ticktext=[mapping.get(code, f"DFC_{code}") for code, _ in sorted_codes],
            autorange="reversed"
        ),
        template="plotly_dark",
        height=min(600, 100 + len(sorted_codes) * 40),
        hovermode="closest"
    )
    return fig.to_json()


# ==================== Public API ====================

def compute_dfc(files: List[Path], mapping_file: Optional[Path] = None,
                channels: Optional[List[str]] = None,
                compress_runs: bool = True, topn: int = 10,
                include_plots: bool = True,
                enable_advanced_features: bool = True,
                **_) -> Dict[str, Any]:
    """
    Enhanced DFC/DTC computation with advanced analysis features.
    
    Args:
        files: List of MDF file paths
        mapping_file: Optional mapping file path (legacy, uses uploads/numdfc.txt)
        channels: Optional channel filter (legacy)
        compress_runs: Legacy parameter
        topn: Legacy parameter
        include_plots: Whether to generate plots
        enable_advanced_features: Enable enhanced DTC analysis
        
    Returns:
        Enhanced DFC/DTC analysis results
    """
    
    if enable_advanced_features:
        (row_count, event_count, runtime_count, evidence, st_series, dfes_series,
         code_segments, code_first_seen, code_last_seen, code_status_bytes,
         correlation_data) = _analyze_enhanced(files, enable_correlation=True)
    else:
        row_count, event_count, runtime_count, evidence, st_series, dfes_series = _analyze(files)
        code_segments = {}
        code_first_seen = {}
        code_last_seen = {}
        code_status_bytes = {}
        correlation_data = {}
    
    mapping = load_numdfc_mapping()
    
    codes = sorted(set(row_count) | set(event_count) | set(runtime_count))
    summary_rows = []
    
    for c in codes:
        # Parse DTC code
        dtc_info = parse_dtc_code(c) if enable_advanced_features else {}
        
        # Calculate segment statistics
        segments = code_segments.get(c, [])
        max_duration = max([seg["duration"] for seg in segments], default=0.0)
        total_duration = sum([seg["duration"] for seg in segments], 0.0)
        first_seen = code_first_seen.get(c)
        last_seen = code_last_seen.get(c)
        
        # Classify severity
        severity = classify_severity(
            dtc_info, 
            event_count.get(c, 0),
            runtime_count.get(c, 0.0),
            max_duration
        ) if enable_advanced_features else "unknown"
        
        # Decode status bytes if available
        status_info = {}
        if c in code_status_bytes and code_status_bytes[c]:
            # Use most common status byte
            status_bytes = code_status_bytes[c]
            most_common_status = max(set(status_bytes), key=status_bytes.count)
            status_info = decode_status_byte(most_common_status)
        
        row = {
            "code": int(c),
            "DFC_name": mapping.get(int(c), dtc_info.get("formatted", f"DFC_{c}")),
            "row_count": row_count.get(c, 0),
            "event_count": event_count.get(c, 0),
            "runtime_count": round(runtime_count.get(c, 0.0), 3)
        }
        
        # Add enhanced fields
        if enable_advanced_features:
            row.update({
                "dtc_format": dtc_info.get("dtc_format", "Unknown"),
                "code_type": dtc_info.get("code_type", "Unknown"),
                "priority": dtc_info.get("priority", "Unknown"),
                "severity": severity,
                "segments": segments,
                "first_seen": round(first_seen, 3) if first_seen else None,
                "last_seen": round(last_seen, 3) if last_seen else None,
                "max_duration": round(max_duration, 3),
                "total_duration": round(total_duration, 3),
                "status_byte": status_info if status_info else None
            })
        
        summary_rows.append(row)

    plots: List[Dict[str, str]] = []
    if include_plots:
        df_summary = pd.DataFrame(summary_rows)
        if not df_summary.empty:
            summary_json = _make_summary_plot(df_summary)
            if summary_json:
                plots.append({"name": "NumDFC Summary", "plotly_json": summary_json})
            
            # Add enhanced plots
            if enable_advanced_features:
                severity_plot = _make_severity_priority_plot(df_summary)
                if severity_plot:
                    plots.append({"name": "DTC Severity Distribution", "plotly_json": severity_plot})
                
                timeline_plot = _make_temporal_timeline_plot(code_segments, mapping)
                if timeline_plot:
                    plots.append({"name": "DTC Timeline", "plotly_json": timeline_plot})
        
        st_json = _make_dfc_st_plot(st_series)
        if st_json:
            plots.append({"name": "DFC_ST Overlay", "plotly_json": st_json})
        dfes_json = _make_dfes_numdfc_plot(dfes_series, mapping)
        if dfes_json:
            plots.append({"name": "DFES_numDFC Overlay", "plotly_json": dfes_json})

    # Build result
    result = {
        "summary": summary_rows, 
        "plots": plots, 
        "channels": evidence, 
        "meta": {
            "enhanced_features_enabled": enable_advanced_features,
            "total_codes": len(codes),
            "codes_with_segments": len(code_segments) if enable_advanced_features else 0
        }
    }
    
    # Add enhanced data
    if enable_advanced_features:
        result["freeze_frames"] = []  # Placeholder for freeze frame data
        result["correlations"] = correlation_data
        result["meta"]["codes_with_correlation"] = len(correlation_data)
    
    return result


# ==================== Quick DFC_ST function (for backward compatibility) ====================

def quick_dfc_st(file_path: Path) -> Dict[str, Any]:
    """
    Quick DFC_ST analysis for a single file (backward compatibility).
    
    Args:
        file_path: Path to MDF file
        
    Returns:
        Dictionary with DFC_ST plot data
    """
    try:
        _, _, _, _, st_series, _ = _analyze([file_path])
        st_json = _make_dfc_st_plot(st_series)
        
        if st_json:
            return {
                "ok": True,
                "plot": st_json,
                "channels_found": len(st_series),
                "channel_names": list(st_series.keys())
            }
        else:
            return {
                "ok": False,
                "error": "No DFC_ST channels found in file"
            }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }


# ==================== CLI ====================

def _expand_files(patterns: List[str]) -> List[Path]:
    """Expand file patterns to actual file paths."""
    out: List[Path] = []
    for p in patterns:
        pp = Path(p)
        if "*" in p or "?" in p:
            out.extend(pp.parent.glob(pp.name))
        elif pp.is_dir():
            out.extend(list(pp.rglob("*.mf4")))
            out.extend(list(pp.rglob("*.mdf")))
        else:
            out.append(pp)
    seen = set(); uniq = []
    for p in out:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq

def main(argv=None):
    """CLI entry point."""
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True, help="Comma-separated file paths or globs")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--legacy", action="store_true", help="Use legacy analysis (no advanced features)")
    args = ap.parse_args(argv)
    flist = _expand_files(args.files.split(","))
    res = compute_dfc(
        flist, 
        include_plots=not args.no_plots,
        enable_advanced_features=not args.legacy
    )
    print(json.dumps(res, indent=2)[:10000])

if __name__ == "__main__":
    main()
