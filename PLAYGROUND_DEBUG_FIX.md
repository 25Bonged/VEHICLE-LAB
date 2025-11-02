# Playground Plot Rendering Debug & Fix

## ✅ Problem Identified

The playground section was not rendering any plots, even though it used to work.

## 🔍 Root Causes Found

1. **Data Format Mismatch**: Backend returns `x`/`y` keys, but frontend `arr()` function needed to handle both `x`/`y` and `timestamps`/`values` formats
2. **Missing Error Handling**: No clear feedback when data fetch fails or returns empty
3. **Silent Failures**: Errors were being swallowed without user feedback

## 🔧 Fixes Applied

### 1. Enhanced `fetchSeriesForSignals` Function
- Added comprehensive error checking
- Better logging for debugging
- Graceful handling of empty responses
- Status messages for user feedback

**Changes:**
```javascript
// Enhanced error checking
if (!res) {
  console.error('[Playground] No response received from /analytics');
  return {};
}

if (res.error) {
  console.error('[Playground] Server error:', res.error, res);
  return {};
}

if (!res.series) {
  console.warn('[Playground] Response missing "series" key. Full response:', res);
  return {};
}
```

### 2. Fixed `arr()` Function to Handle Multiple Formats
- Now supports both `x`/`y` and `timestamps`/`values` formats
- Normalizes data to ensure consistent format
- Better fallback handling

**Changes:**
```javascript
// Normalize to x/y format (backend may return x/y or timestamps/values)
return {
  x: found.timestamps || found.x || [],
  y: found.values || found.y || [],
  timestamps: found.timestamps || found.x || [],
  values: found.values || found.y || [],
  name: found.name || id,
  unit: found.unit || ''
};
```

### 3. Backend Response Format Enhancement
- Added `timestamps` and `values` keys alongside `x` and `y` for compatibility
- Maintains backward compatibility

**Changes in `app.py`:**
```python
out[key]={"timestamps":xs,"values":ys,"name":s.get("name",key),"unit":s.get("unit",""),
          "x":xs,"y":ys}  # Keep x/y for backward compatibility
```

### 4. Enhanced `renderPlot` Function
- Added debugging logs to track data flow
- Better error messages for empty data
- Status updates for user feedback

**Changes:**
```javascript
// Debug: Log what we got back
console.debug('[Playground] Fetched series data:', {
  requested: [...idsToFetch],
  received: Object.keys(S),
  missing: [...idsToFetch].filter(id => !S[id])
});

if (Object.keys(S).length === 0) {
  console.warn('[Playground] No series data received. Check signal IDs and file selection.');
  const statusEl = el('pg-status');
  if (statusEl) {
    statusEl.textContent = 'No data available. Please select signals and ensure files are uploaded.';
    statusEl.classList.remove('hidden');
  }
  Plotly.react(plotDiv, [], {title: 'No Data Available', ...}, PLOTLY_BASE_CONFIG);
  return;
}
```

## 🧪 Testing

To test with ETK data:

1. **Upload ETK Data File:**
   - Go to "Analyse" tab
   - Upload your ETK `.mdf` or `.mf4` file
   - Wait for channel discovery to complete

2. **Navigate to Playground:**
   - Click "Playground" tab
   - Select signals from dropdowns (X, Y axes)
   - Click "Render Plot"

3. **Check Console:**
   - Open browser DevTools (F12)
   - Check Console tab for `[Playground]` debug messages
   - Look for:
     - Request details
     - Response received
     - Missing signals (if any)

4. **Expected Behavior:**
   - Plot should render with selected signals
   - Status message shows "Ready" when complete
   - Any errors will be logged to console

## 📊 Debugging Guide

If plots still don't render:

1. **Check Browser Console:**
   - Look for `[Playground]` prefixed messages
   - Check for error messages

2. **Verify Signal Selection:**
   - Ensure signals are selected in X and Y dropdowns
   - Check that signals exist in the uploaded file

3. **Verify File Upload:**
   - Check "Files" tab to confirm file is uploaded
   - Verify file is selected (should show in "Analyse" section)

4. **Check Network Tab:**
   - Open DevTools → Network tab
   - Click "Render Plot"
   - Look for `/analytics` POST request
   - Check response for `series` key and data

## ✅ Verification Checklist

- [x] Enhanced error handling in `fetchSeriesForSignals`
- [x] Fixed `arr()` function to handle multiple formats
- [x] Added debugging logs throughout render pipeline
- [x] Backend returns both `x`/`y` and `timestamps`/`values`
- [x] User-friendly error messages
- [x] Status updates for feedback

---

**Status:** ✅ **FIXED**

**Date:** 2025-11-01

