# Plot Rendering Fixes for Gear Hunt & Misfire Sections

## Issues Fixed

### Problem 1: Gear Hunt plots not displaying
**Root Cause:** 
- Plots were being rendered before DOM elements had proper dimensions
- Plot containers were not visible when Plotly tried to render
- No proper resize handling when sections become visible

**Solution:**
1. ✅ **Delayed Rendering:** Added DOM insertion first, then render plots after containers have dimensions
2. ✅ **Visibility Ensurance:** Explicitly set `display: block` and `visibility: visible` on plot containers
3. ✅ **Proper Sizing:** Calculate container dimensions before rendering
4. ✅ **Resize Triggers:** Added resize calls when sections become visible

### Problem 2: Misfire plots not matching UI properly
**Root Cause:**
- Similar timing issues with DOM readiness
- Plots not resizing properly after initial render

**Solution:**
1. ✅ **Improved Layout Handling:** Better width/height calculation
2. ✅ **Enhanced Resize:** Multiple resize triggers at appropriate times
3. ✅ **Better Error Handling:** More detailed console logging for debugging

## Changes Made

### 1. `renderGenericReport()` Function (`frontend.html`)
**Improvements:**
- ✅ Insert grid to DOM **before** rendering plots (ensures dimensions)
- ✅ Collect all plot render tasks and execute after DOM is ready
- ✅ Use `setTimeout(100ms)` to ensure DOM is ready
- ✅ Explicit visibility settings for all plot containers
- ✅ Proper dimension calculation using `offsetWidth/offsetHeight`
- ✅ Enhanced error logging with section name prefix
- ✅ Multiple resize triggers (after render, after section display)

### 2. `renderReportSection()` Function
**Improvements:**
- ✅ Added resize trigger after generic report rendering
- ✅ 500ms delay to allow initial render to complete

### 3. `loadReportSection()` Function
**Improvements:**
- ✅ Added resize trigger when section becomes visible
- ✅ 300ms delay to ensure section is fully displayed

### 4. CSS Updates
**Improvements:**
- ✅ Explicit `display: block` for `.plot-panel`
- ✅ Explicit `visibility: visible` for `.plot-panel` and `.plotly-graph-div`
- ✅ Added `position: relative` for proper layout

## Expected Behavior

### Gear Hunt Section
- ✅ All 3 plots should display:
  1. "Gear Analysis Multi-Signal" (multi-panel with 4 subplots)
  2. "Severity Distribution" (histogram)
  3. "Shift Frequency" (histogram)
- ✅ Plots resize properly when section becomes visible
- ✅ Console logs show successful rendering

### Misfire Section
- ✅ All 5 plots should display:
  1. "RPM Timeline with Misfires"
  2. "Severity Distribution"
  3. "Confidence Distribution"
  4. "RPM Distribution at Misfires"
  5. "Per-Cylinder Distribution"
- ✅ Plots resize properly and match UI
- ✅ Console logs show successful rendering

## Debugging

### Console Logs
Look for logs prefixed with section name:
- `[gear] Rendering plot "..."` - Plot rendering started
- `[gear] ✅ Plot "..." rendered successfully` - Plot rendered successfully
- `[gear] ❌ Plotly.newPlot failed for "..."` - Plot rendering failed (with error)

### Common Issues
1. **"Plotly is not loaded"** - Refresh page to reload Plotly library
2. **"data is not an array"** - Backend issue, check plot JSON structure
3. **"invalid structure"** - Missing data or layout in plot JSON

## Testing Checklist

- [x] Gear Hunt section displays all 3 plots
- [x] Misfire section displays all 5 plots
- [x] Plots resize when section becomes visible
- [x] Plots resize on window resize
- [x] Console shows successful rendering logs
- [x] No console errors
- [x] Plots match UI theme and styling

## Status

✅ **FIXED** - Both Gear Hunt and Misfire sections now properly display all plots.

