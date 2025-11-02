# Calibration AI Section Removal

## ✅ Changes Made

Removed the unused "Calibration AI" section from the dashboard.

### Frontend (`frontend.html`)

1. **Navigation Tab:**
   - Removed `<button id="tabbtn-cie">Calibration AI</button>` from the tabs navigation

2. **Section HTML:**
   - Removed entire `<section id="sec-cie">` section including:
     - CIE controls (Sync Channels, Sync Theme buttons)
     - Status indicator
     - iframe for `/cie.html`

3. **CSS Styles:**
   - Removed all `#sec-cie` styles
   - Removed all `.cie-*` class styles (controls, panels, inputs, plots, results)
   - Removed CIE-specific styling

4. **JavaScript:**
   - Removed `initCIEModule()` function and all CIE-related event handlers
   - Removed `'sec-cie'` from tab switching array
   - Removed `initCIEModule()` call from initialization

### Backend (`app.py`)

1. **Route:**
   - Removed `@app.get("/cie.html")` route handler

## 📊 Remaining Sections

The dashboard now has these sections:
1. **Analyse** - Upload & Discover
2. **Report** - Analysis reports (DFC, IUPR, Gear Hunt, Misfire, etc.)
3. **Files** - File management
4. **Playground** - Custom plot visualization

## 🧹 Cleanup Status

✅ Navigation button removed  
✅ Section HTML removed  
✅ CSS styles removed  
✅ JavaScript functions removed  
✅ Backend route removed  
✅ No references remaining

---

**Status:** ✅ **COMPLETE**

**Date:** 2025-11-01

