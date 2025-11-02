#!/usr/bin/env python3
"""
custom_iupr.py — MDF-native IUPR analyzer (app.py compatible, cleaned + professional)

Improvements:
- Removed fleet overlay progression plots
- No individual progression plots
- Only keep:
    * Final Ratios bar chart
    * Histogram
    * Min/Max bar chart
- Fixed label overlaps (rotated / wrapped axis labels, margins)
- Annotations kept inside canvas
- Stable on Edge browser zoom (65%–125%)
"""

from __future__ import annotations
import argparse, json, os, re
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import plotly.graph_objects as go

try:
    from asammdf import MDF
except ImportError:
    MDF = None

# Regex
NUM_RX     = re.compile(r".*DIUMPR_ctNum\[?(\d+)\]?", re.IGNORECASE)
DEN_RX     = re.compile(r".*DIUMPR_ctDenom\[?(\d+)\]?", re.IGNORECASE)
GEN_DEN_RX = re.compile(r".*DIUMPR_ctGenDenom", re.IGNORECASE)

# Monitor names
MONITOR_NAMES = {
    0: "Catalyst_Bank1", 1: "Catalyst_Bank2",
    2: "OxygenSensor_Bank1", 3: "OxygenSensor_Bank2",
    4: "EGR_VVT", 5: "SecAirSys", 6: "EvpSys",
    7: "SecOxySens_Bank1", 8: "SecOxySens_Bank2",
    9: "AFR1_Bank1", 10: "AFR1_Bank2",
    11: "PF_Bank1", 12: "PF_Bank2",
    13: "Private", 14: "Unused"
}

def _read_signal(mdf: MDF, name: str):
    try:
        sig = mdf.get(name)
        return sig.samples
    except Exception:
        return None

def process_file(file_path: Path):
    """Extract numerator/denominator and compute ratios"""
    mdf = MDF(str(file_path), memory="minimum")
    try:
        channels = list(mdf.channels_db.keys())
        num_ch, den_ch, gen_denom = {}, {}, None
        for ch in channels:
            if (m := NUM_RX.match(ch)): num_ch[int(m.group(1))] = ch; continue
            if (m := DEN_RX.match(ch)): den_ch[int(m.group(1))] = ch; continue
            if GEN_DEN_RX.match(ch): gen_denom = ch

        summary = {}
        for idx, n_ch in sorted(num_ch.items()):
            d_ch = den_ch.get(idx) or gen_denom
            if not d_ch: continue
            v_n, v_d = _read_signal(mdf,n_ch), _read_signal(mdf,d_ch)
            if v_n is None or v_d is None: continue
            num, den = np.array(v_n,float), np.array(v_d,float)
            if num.size==0 or den.size==0: continue
            n = min(len(num),len(den)); num,den = num[:n],den[:n]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(den>0,num/den,np.nan)
            valid = ratio[~np.isnan(ratio)]
            if not valid.size: continue
            summary[idx] = {
                "index": idx,
                "monitor": MONITOR_NAMES.get(idx,f"Monitor{idx}"),
                "final_ratio": float(valid[-1]),
                "min_ratio": float(np.nanmin(valid)),
                "max_ratio": float(np.nanmax(valid)),
                "file": file_path.name
            }
        return summary
    finally:
        try: mdf.close()
        except: pass

def compute_iupr(files: List[Path]) -> Dict[str, Any]:
    merged = {}
    for f in files:
        if not f.exists(): continue
        res = process_file(f)
        for idx,row in res.items(): merged[idx]=row
    rows = [merged[idx] for idx in sorted(merged.keys())]

    plots = []
    if rows:
        monitors=[r["monitor"] for r in rows]
        ratios=[r["final_ratio"] for r in rows]
        mins=[r["min_ratio"] for r in rows]
        maxs=[r["max_ratio"] for r in rows]

        # Final Ratios
        bar=go.Figure()
        bar.add_trace(go.Bar(x=monitors,y=ratios,marker_color="steelblue"))
        bar.add_hline(y=0.1,line_dash="dash",line_color="red")
        bar.add_hline(y=0.9,line_dash="dash",line_color="green")
        bar.update_layout(
            title="Final IUPR Ratios",
            template="plotly_dark",height=500,
            margin=dict(l=60,r=40,t=60,b=120),
            xaxis=dict(tickangle=-30, automargin=True),
            paper_bgcolor='black',  # Deep black background
            plot_bgcolor='black',  # Deep black background
            font=dict(color='#dce1e6')  # Light text for dark mode
        )
        plots.append({"name":"Final IUPR Ratios","plotly_json":bar.to_json()})

        # Histogram
        hist=go.Figure()
        hist.add_trace(go.Histogram(x=ratios,nbinsx=20,marker_color="skyblue"))
        hist.add_vline(x=0.1,line_dash="dash",line_color="red")
        hist.add_vline(x=0.9,line_dash="dash",line_color="green")
        hist.update_layout(
            title="Histogram of Final Ratios",
            template="plotly_dark",height=400,
            margin=dict(l=60,r=40,t=60,b=90),
            xaxis=dict(automargin=True),
            paper_bgcolor='black',  # Deep black background
            plot_bgcolor='black',  # Deep black background
            font=dict(color='#dce1e6')  # Light text for dark mode
        )
        plots.append({"name":"Histogram of Final Ratios","plotly_json":hist.to_json()})

        # Min/Max
        mm=go.Figure()
        mm.add_trace(go.Bar(x=monitors,y=maxs,name="Max",marker_color="seagreen"))
        mm.add_trace(go.Bar(x=monitors,y=mins,name="Min",marker_color="indianred"))
        mm.update_layout(
            barmode="group",
            title="Min/Max Ratios per Monitor",
            template="plotly_dark",height=450,
            margin=dict(l=60,r=40,t=60,b=120),
            xaxis=dict(tickangle=-30, automargin=True),
            paper_bgcolor='black',  # Deep black background
            plot_bgcolor='black',  # Deep black background
            font=dict(color='#dce1e6')  # Light text for dark mode
        )
        plots.append({"name":"Min/Max Ratios","plotly_json":mm.to_json()})

    return {
        "tables": {"Final IUPR Summary": rows},
        "plots": {p["name"]:{"plotly_json":p["plotly_json"]} for p in plots},
        "meta": {"monitor_count": len(rows)}
    }

def compute_iupr_plotly(files: List[str]) -> Dict[str, Any]:
    return compute_iupr([Path(f) for f in files])

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--files",nargs="+",required=True)
    args=ap.parse_args()
    print(json.dumps(compute_iupr([Path(f) for f in args.files]),indent=2)[:20000])
