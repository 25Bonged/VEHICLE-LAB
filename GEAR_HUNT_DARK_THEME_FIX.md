# Gear Hunt Plot - Dark Theme & Combined Speed/RPM Fix

## ✅ Changes Made

### 1. **Dark Background Theme**
- Changed all gear hunt plots from `plotly_white` to `plotly_dark`
- Added transparent backgrounds (`paper_bgcolor` and `plot_bgcolor`) to blend with dashboard
- Applied to:
  - Main multi-signal plot
  - Severity Distribution plot
  - Shift Frequency plot

### 2. **Combined Speed & RPM Panel**
- **Reduced from 4 rows to 3 rows:**
  - Row 1: Gear Position Over Time
  - Row 2: **Speed & RPM (combined)** ← NEW
  - Row 3: Throttle Position

- **Dual Y-Axis Implementation:**
  - **Left Y-axis (y2):** Speed (km/h) - Green line (#4CAF50)
  - **Right Y-axis (y4):** RPM - Orange line (#FF9800)
  - Both axes overlay in the same panel for easy correlation

### 3. **Technical Improvements**
- Proper axis naming for subplots
- Color-coded traces (Green for Speed, Orange for RPM)
- Dynamic subplot titles based on available data
- Legend shows both Speed and RPM when both exist

## 📊 Plot Structure

### Before:
```
Row 1: Gear Position
Row 2: Speed (separate)
Row 3: Throttle
Row 4: RPM (separate)
```

### After:
```
Row 1: Gear Position
Row 2: Speed & RPM (combined with dual y-axes)
Row 3: Throttle
```

## 🎨 Visual Improvements

1. **Dark Theme:**
   - Matches dashboard dark mode
   - Transparent backgrounds for seamless integration
   - Better readability in dark environment

2. **Combined Speed/RPM:**
   - Easier correlation between speed and engine RPM
   - Single panel reduces vertical scrolling
   - Color-coded lines for quick identification

## 🔧 Code Changes

**File:** `custom_gear.py`

**Key Updates:**
- Changed `make_subplots(rows=4)` → `make_subplots(rows=3)`
- Updated subplot titles to include "Speed & RPM"
- Added dual y-axis configuration for row 2
- Changed all `template="plotly_white"` → `template="plotly_dark"`
- Added color coding for Speed (green) and RPM (orange)

---

**Status:** ✅ **COMPLETE**

**Date:** 2025-11-01

