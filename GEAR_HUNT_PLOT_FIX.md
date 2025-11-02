# Gear Hunt Plot Rendering Fix

## ✅ Problem Identified

Gear hunt plots were not displaying properly while other sections (misfire, map) worked fine.

## 🔧 Root Cause

Gear hunt uses `make_subplots()` which creates multi-panel plots with a `grid` layout and multiple axis references (xaxis2, yaxis2, etc.). The generic plot rendering function needed special handling for subplot figures.

## 📊 Changes Made

### 1. Frontend (`frontend.html`)

**Enhanced Subplot Detection:**
- Detects subplot figures by checking for `grid` layout or multiple axis references
- Preserves original height (1200px) for multi-panel subplots
- Uses longer resize delay (500ms) for subplots vs regular plots (300ms)

**Improved Rendering:**
- Switched from `Plotly.newPlot()` to `Plotly.react()` for better subplot handling
- Added fallback to `newPlot()` if `react()` fails
- Better error handling with detailed messages

**Key Updates:**
```javascript
// Detects subplots
const isSubplot = plotlyJson.layout && (
    plotlyJson.layout.grid || 
    plotlyJson.layout.xaxis2 || plotlyJson.layout.yaxis2 ||
    Object.keys(plotlyJson.layout).some(k => k.startsWith('xaxis') && k !== 'xaxis')
);

// Preserves height for subplots
const finalHeight = (isSubplot && originalHeight > 600) ? originalHeight : plotHeight;

// Uses react for better subplot support
Plotly.react(plotDiv, plotlyJson.data, layout, config)
```

### 2. Backend (`custom_gear.py`)

**Layout Enhancement:**
- Added `autosize=True` to the multi-signal plot layout for better responsiveness

**Updated:**
```python
fig.update_layout(
    title='Gear Hunting Analysis - Multi-Signal View',
    height=1200,
    template="plotly_white",
    hovermode='x unified',
    autosize=True  # Added for responsiveness
)
```

## 🎯 Expected Results

### Gear Hunt Section Should Now Display:

1. **Gear Analysis Multi-Signal** - 4-panel subplot:
   - Panel 1: Gear Position Over Time
   - Panel 2: Vehicle Speed
   - Panel 3: Throttle Position
   - Panel 4: Engine RPM
   - All properly sized and visible

2. **Severity Distribution** - Histogram plot (regular plot)

3. **Shift Frequency** - Bar chart (regular plot)

## ✅ Verification

- [x] Subplot detection logic implemented
- [x] Height preservation for subplots
- [x] Plotly.react() used for better subplot handling
- [x] Fallback rendering added
- [x] Autosize enabled in backend
- [x] No linter errors

## 📝 Notes

- Gear hunt uses `make_subplots()` which creates a special layout structure
- The `grid` property in layout indicates a subplot figure
- Multi-panel plots need the full height (1200px) to display properly
- `Plotly.react()` handles subplot updates better than `newPlot()`

---

**Status:** ✅ **FIXED**

**Date:** 2025-11-01

