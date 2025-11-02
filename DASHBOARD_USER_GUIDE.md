# MDF Analytics Dashboard - Advanced User Guide
## Complete Guide for Diagnostic Engineers

**Version:** 2.0  
**Last Updated:** October 2025  
**Target Audience:** Diagnostic Engineers, Data Analysts, Calibration Engineers

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Supported File Formats](#supported-file-formats)
4. [Core Features Walkthrough](#core-features-walkthrough)
5. [Advanced Workflows](#advanced-workflows)
6. [API Reference](#api-reference)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

## Overview

### What is This Dashboard?

The MDF Analytics Dashboard is a comprehensive web-based tool for analyzing vehicle diagnostic data from MDF/MF4 files, CSV, and Excel formats. It provides:

- **Multi-format Support**: MDF, MF4, CSV, Excel (.xlsx, .xls)
- **Real-time Signal Analysis**: Plotting, statistics, FFT, histograms
- **Empirical Map Generation**: 2D heatmaps and 3D surface plots
- **Advanced Analytics**: DFC analysis, IUPR monitoring, CC/SL behavior
- **Gear Hunting Detection**: Automatic event detection and visualization
- **Calibration AI**: Automated map optimization and recommendations

### Architecture

```
┌─────────────────────────────────────────────────┐
│           Frontend (HTML/JavaScript)           │
│  - Real-time visualization with Plotly          │
│  - Interactive signal selection                │
│  - Dynamic report generation                   │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│          Backend (Flask/Python)                 │
│  - File upload & processing                     │
│  - Signal extraction & caching                 │
│  - Analytics computation                       │
│  - Map generation                              │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│         Data Sources                             │
│  - MDF/MF4 (via asammdf)                        │
│  - CSV/Excel (via pandas)                       │
│  - Upload directory: ./uploads/                  │
└─────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Starting the Dashboard

```bash
cd /path/to/backend_mdf
python3 launch_dashboard.py
```

**Expected Output:**
```
🚀 Starting MDF Analytics Dashboard...
📁 Working directory: /path/to/backend_mdf
✅ App imported successfully
🌐 Starting server on http://localhost:8000
📊 Dashboard will be available at: http://localhost:8000
🛑 Press Ctrl+C to stop the server
```

### 2. Access the Dashboard

Open your browser and navigate to: **http://localhost:8000**

### 3. First Steps

1. **Upload Files**: Drag & drop MDF/MF4/CSV/Excel files into the upload area
2. **View Channels**: Click "Files" tab to see uploaded files
3. **Select Signals**: Use "Analyse" tab to select signals for analysis
4. **Generate Plots**: Click "Plot Signals" to visualize data

---

## Supported File Formats

### MDF/MF4 Files
- **Format**: ASAM MDF (Measurement Data Format) v3.x and v4.x
- **Usage**: Primary format for vehicle diagnostic data
- **Features**: 
  - Automatic channel discovery
  - Time-synchronized data
  - Unit information preserved
  - Multi-file merging support

### CSV Files
- **Format**: Comma-separated values with flexible structure
- **Supported Variants**:
  - Standard CSV (header row only)
  - Description + Units rows (automatically detected)
  - Multiple encodings: UTF-8, Latin-1, ISO-8859-1, CP1252
- **Time Column Detection**: Automatic detection of time/timestamp columns

### Excel Files
- **Formats**: `.xlsx` (Excel 2007+), `.xls` (Excel 97-2003)
- **Features**: 
  - Multiple sheet support (reads first sheet by default)
  - Automatic numeric conversion
  - Time column auto-detection

### File Size Limits
- **Maximum upload size**: 500 MB per file (configurable)
- **Recommended**: Files < 100 MB for best performance
- **Large files**: Automatically processed in chunks

---

## Core Features Walkthrough

### Section 1: Upload & Discover

#### Uploading Files

**Method 1: Drag & Drop**
1. Drag files from your file explorer
2. Drop into the upload area
3. Files are automatically processed

**Method 2: Click to Browse**
1. Click the upload area
2. Select files from file picker
3. Supports multiple file selection

**Supported Extensions**: `.mdf`, `.mf4`, `.csv`, `.xlsx`, `.xls`

#### Channel Discovery

After upload:
- **Automatic channel listing**: All signals appear in the channel table
- **Search functionality**: Filter channels by name
- **Channel information**:
  - Channel ID (original name)
  - Clean name (normalized)
  - Presence count (how many files contain this signal)
  - File-specific channels

#### File Management

**View Uploaded Files**:
- Click "Files" tab
- See all uploaded files with:
  - File name
  - Size
  - Format
  - Upload timestamp

**Remove Files**:
- Select files to delete
- Click "Delete Selected"
- Or use "Delete All" to clear all uploads

---

### Section 2: Analyse

#### Signal Selection

1. **Browse Channels**: Scroll through the channel list or use search
2. **Select Signals**: Click checkboxes next to signals
3. **Batch Selection**: 
   - Use "Select All" for all visible channels
   - Use "Clear Selected" to deselect
   - Selection persists across searches

#### Signal Plotting

**Basic Plot**:
1. Select one or more signals
2. Click "Plot Signals"
3. View interactive Plotly chart

**Plot Options**:

**Time Handling**:
- **Include Time**: Use original timestamps (recommended)
- **Exclude Time**: Use sample index as X-axis

**Normalization**:
- **Normalize**: Scale all signals to 0-1 range
- **Raw Values**: Use original signal values

**Downsampling**:
- **Max Points**: Limit points for performance (default: 10,000)
- **Algorithm**: 
  - LTTB (Largest-Triangle-Three-Buckets): Best visual quality
  - Stride: Simple decimation (faster)

#### Advanced Features

**Multi-Signal Plotting**:
- Select multiple signals for comparison
- Each signal gets its own color
- Synchronized time axis

**Subplots Mode**:
- Enable "Subplots" checkbox
- Each signal in separate subplot
- Shared X-axis for easy comparison

**Multi-Y Mode**:
- Enable "Multi Y" checkbox
- Multiple Y-axes for different scales
- Useful for signals with different units

#### Statistics

After plotting, view:
- **Min/Max/Mean**: Basic statistics per signal
- **Sample Count**: Number of data points
- **Copy to Clipboard**: Export statistics

#### FFT Analysis

**Enabling FFT**:
1. Check "FFT" checkbox before plotting
2. Plot signals
3. View frequency domain plot

**FFT Features**:
- Frequency spectrum visualization
- Amplitude vs frequency
- Useful for identifying periodic patterns

#### Histograms

**Enabling Histograms**:
1. Check "Histogram" checkbox before plotting
2. Plot signals
3. View distribution plots

**Histogram Features**:
- Distribution visualization
- Bins: 50 (default, adjustable)
- Useful for understanding signal ranges

---

### Section 3: Playground

**Advanced Interactive Plotting**

The Playground offers full control over plot generation:

#### Plot Configuration

**X-Axis Selection**:
- Choose any signal as X-axis
- Default: Time

**Y-Axis Selection**:
- Select one or more signals
- Multi-Y support for different scales

**Z-Axis (3D Plots)**:
- Select signal for 3D visualization
- Creates scatter3d or surface plots

**Plot Types**:
- `scatter`: Line plot
- `scatter3d`: 3D scatter
- `surface`: 3D surface
- `heatmap`: 2D heatmap
- `histogram`: Distribution
- `violin`: Distribution with density
- `box`: Box plot
- `bar`: Bar chart

#### Time Range Selection

**Filtering by Time**:
- `tmin`: Minimum timestamp
- `tmax`: Maximum timestamp
- Only data within range is plotted

#### Downsampling

- **Algorithm**: LTTB (recommended) or Stride
- **Points**: Maximum points to display
- **Purpose**: Improve performance with large datasets

#### Adding Traces

- **Add Mode**: Add new traces to existing plot
- **Replace Mode**: Replace all traces (default)

---

### Section 4: Report

#### Report Sections

**Available Sections**:
1. **Empirical Map**: Calibration maps (BSFC, efficiency, etc.)
2. **DFC Analysis**: Diagnostic trouble code frequency analysis
3. **CC/SL Analysis**: Cruise control / speed limiter behavior
4. **IUPR Analysis**: In-use performance tracking (if available)
5. **Gear Hunting**: Transmission gear hunting detection
6. **Custom Analysis**: User-defined analysis

#### Generating Reports

1. **Select Section**: Choose report type from tabs
2. **Configure Parameters**: Set analysis-specific options
3. **Generate**: Click generate button
4. **View Results**: Plots and tables appear below

#### Exporting Reports

**Formats Available**:
- **PNG**: Plot images
- **CSV**: Tabular data
- **XLSX**: Excel workbook with multiple sheets
- **JSON**: Raw data export

**Export Options**:
- Export individual plots
- Export full report as PDF (if available)
- Download map data as CSV/Excel

---

### Section 5: Empirical Map

**Most Important for Calibration Engineers**

#### Understanding Empirical Maps

Empirical maps visualize 3D relationships:
- **X-Axis**: Typically Engine RPM
- **Y-Axis**: Typically Engine Torque
- **Z-Axis**: Target parameter (BSFC, efficiency, temperature, etc.)

#### Map Generation Workflow

**Step 1: File Selection**
1. Navigate to "Empirical Map" section
2. Check files to include in map generation
3. Multiple files are automatically merged

**Step 2: Signal Selection**

**Using Presets**:
1. Click "Select Preset" dropdown
2. Choose preset (e.g., "CI Engine — BSFC")
3. Preset auto-configures:
   - X/Y/Z axis signals
   - Bin configuration
   - Interpolation method

**Manual Configuration**:
1. **X-Axis Signal**: Select RPM signal (e.g., "Epm_nEng")
2. **Y-Axis Signal**: Select Torque signal (e.g., "TqSys_tqCkEngReal_RTE")
3. **Z-Axis Signal**: Select target parameter (e.g., "BSFC", "FuelRate")

**Step 3: Bin Configuration**

**Format**: `start:stop:step` or comma-separated values

**Examples**:
- RPM bins: `100:6000:100` (100 to 6000 RPM, 100 RPM steps)
- Torque bins: `0:600:50` (0 to 600 N·m, 50 N·m steps)
- Custom: `0,500,1000,1500,2000` (specific values)

**Best Practices**:
- **RPM bins**: Match your calibration tables (typically 100-500 RPM steps)
- **Torque bins**: 5-50 N·m steps depending on resolution needed
- **Coverage**: Ensure bins cover your operating range

**Step 4: Map Type Selection**

- **Heatmap**: 2D color-coded map (fast, good for overview)
- **Contour**: Contour lines overlay (shows gradients)
- **3D Surface**: Interactive 3D visualization (best detail)

**Step 5: Advanced Options**

**Min Samples Per Bin**:
- **Purpose**: Minimum data points required in each bin
- **Default**: 6 samples
- **Lower values**: More bins filled, but less reliable
- **Higher values**: More reliable, but fewer bins filled

**Interpolation Method**:
- **linear**: Fast, basic interpolation
- **cubic**: Smoother, slower
- **rbf**: Radial basis function (smooth, slowest)

**Smoothing**:
- **Range**: 0.0 to 2.0
- **Purpose**: Reduces noise in maps
- **Higher values**: More smoothing (less detail)
- **Default**: 0.5

**Step 6: Generate Map**

1. Click "Generate Map"
2. Watch progress indicator
3. Results appear in tabs:
   - **Heatmap View**: 2D map
   - **3D View**: Interactive surface
   - **Statistics**: Map statistics table

#### Interpreting Map Results

**Coverage Percentage**:
- Shows what % of map cells have data
- **Good**: >70% coverage
- **Acceptable**: 40-70%
- **Poor**: <40% (need more data or wider bins)

**Map Statistics**:
- **Mean**: Average value across map
- **Min/Max**: Extreme values
- **Std**: Standard deviation (variability)

**Visual Analysis**:
- **Color gradients**: Show parameter distribution
- **Contour lines**: Identify optimal regions
- **Gaps**: Missing data areas

#### Map Presets

**CI Engine — BSFC**:
- **Purpose**: Brake-specific fuel consumption mapping
- **X**: Engine RPM
- **Y**: Engine Torque
- **Z**: BSFC (g/kWh)
- **Use Case**: Fuel efficiency optimization

**SI Engine — Efficiency/AFR**:
- **Purpose**: Spark-ignition engine efficiency
- **X**: Engine RPM
- **Y**: Engine Torque
- **Z**: Efficiency, BSFC, AFR, Lambda
- **Use Case**: Engine performance tuning

**Electric Motor — Efficiency**:
- **Purpose**: Electric motor efficiency mapping
- **X**: Motor Speed
- **Y**: Motor Torque
- **Z**: Motor Efficiency
- **Use Case**: EV/HEV optimization

---

### Section 6: DFC Analysis

**Diagnostic Fault Code Frequency Analysis**

#### Purpose

Analyzes how often diagnostic fault codes (DFCs) occur in your data:
- **Code frequency**: How many times each code appears
- **Event count**: Total occurrences
- **Runtime analysis**: Time-based occurrence patterns
- **Evidence channels**: Signals associated with DFCs

#### Running DFC Analysis

1. Navigate to "Report" → "DFC Analysis"
2. Select files containing DFC data
3. Click "Generate DFC Analysis"
4. View results:
   - **Summary Table**: Top DFCs by frequency
   - **Plots**: Time-series plots of DFC occurrences
   - **Evidence**: Associated signal information

#### DFC Mapping

**Custom Mapping File**:
- Create `numdfc.txt` in uploads directory
- Format: `code -> 'Description'` or `'Description' -> code`
- Example: `0x1234 -> 'Coolant Temperature High'`

**Automatic Mapping**:
- System attempts to match codes to known descriptions
- Uses channel names and patterns

---

### Section 7: CC/SL Analysis

**Cruise Control / Speed Limiter Behavior**

#### Purpose

Analyzes cruise control and speed limiter behavior:
- **Overshoot detection**: When vehicle exceeds set speed
- **Response time**: Time to reach set speed
- **Stability**: Speed regulation quality

#### Configuration

**Environment Variables** (in dashboard):
- **Overshoot Margin**: Allowed speed deviation (km/h)
- **Flag Value**: DFC flag value to check
- **Fallback Mode**: Behavior when flags unavailable

---

### Section 8: Calibration AI

**Automated Map Optimization**

#### Features

- **DOE Generation**: Design of Experiments
- **Map Recommendations**: AI-suggested map improvements
- **Optimization**: Automated parameter tuning
- **Training Data Preparation**: Prepare datasets for ML models

**Note**: Requires optional dependencies (scikit-learn, scipy, skopt, optuna)

---

## Advanced Workflows

### Workflow 1: Full Calibration Map Generation

**Scenario**: Generate comprehensive BSFC map for engine calibration

**Steps**:
1. **Upload Files**:
   ```
   - Upload multiple MDF files covering operating range
   - Include steady-state and transient data
   - Ensure good coverage of RPM/Torque space
   ```

2. **Select Preset**:
   ```
   - Choose "CI Engine — BSFC"
   - Or manually configure:
     * X: Epm_nEng (RPM)
     * Y: TqSys_tqCkEngReal_RTE (Torque)
     * Z: bsfc_gpkwh (BSFC)
   ```

3. **Configure Bins**:
   ```
   RPM: 800:4000:200  (800 to 4000, 200 RPM steps)
   Torque: 50:500:25  (50 to 500 N·m, 25 N·m steps)
   ```

4. **Set Quality**:
   ```
   Min Samples: 8 (higher quality)
   Interpolation: rbf (smoother)
   Smoothing: 0.8 (reduce noise)
   ```

5. **Generate & Export**:
   ```
   - Generate map
   - Review coverage (>70% ideal)
   - Export as CSV/XLSX for ECU import
   ```

### Workflow 2: Signal Correlation Analysis

**Scenario**: Find relationships between signals

**Steps**:
1. **Plot Multiple Signals**:
   ```
   - Select 2-5 related signals
   - Use "Subplots" mode
   - Enable time synchronization
   ```

2. **Statistical Analysis**:
   ```
   - View statistics table
   - Check min/max/mean
   - Identify correlations visually
   ```

3. **FFT Analysis**:
   ```
   - Enable FFT for all signals
   - Compare frequency spectrums
   - Identify common frequencies
   ```

### Workflow 3: Fault Code Investigation

**Scenario**: Investigate recurring DFCs

**Steps**:
1. **Run DFC Analysis**:
   ```
   - Generate DFC report
   - Identify most frequent codes
   ```

2. **Correlate with Signals**:
   ```
   - Find signals associated with DFC
   - Plot signals around DFC events
   - Look for abnormal patterns
   ```

3. **Time Analysis**:
   ```
   - Use time filtering in Playground
   - Focus on DFC event windows
   - Identify triggers
   ```

### Workflow 4: CSV Data Import & Analysis

**Scenario**: Analyze data from external source (Excel export, datalogger, etc.)

**Steps**:
1. **Prepare CSV**:
   ```
   - Format: Header row with signal names
   - Optional: Description row, Units row
   - Ensure Time column exists (or use index)
   ```

2. **Upload**:
   ```
   - Drag & drop CSV file
   - System auto-detects structure
   - Channels appear in table
   ```

3. **Verify Time Column**:
   ```
   - Check if "Time" column detected
   - If not, signal timestamps use index
   - Verify data looks correct in plots
   ```

4. **Analysis**:
   ```
   - Use same analysis features as MDF
   - Generate maps, statistics, etc.
   ```

### Workflow 5: Multi-File Comparison

**Scenario**: Compare different test runs or configurations

**Steps**:
1. **Upload Multiple Files**:
   ```
   - Upload all files to compare
   - Files are merged automatically
   - Channel table shows presence count
   ```

2. **Select Common Signals**:
   ```
   - Choose signals present in all files
   - Use presence filter if needed
   ```

3. **Comparative Analysis**:
   ```
   - Plot all files together
   - Use different colors per file
   - Compare statistics
   ```

4. **Generate Maps**:
   ```
   - Include all files in map generation
   - System merges data automatically
   - Better map coverage
   ```

---

## API Reference

### For Advanced Users / Automation

#### Endpoints

**1. File Upload**
```
POST /smart_merge_upload
Content-Type: multipart/form-data

Parameters:
- files: File(s) to upload
- mode: "append" or "replace" (default: "replace")

Response:
{
  "ok": true,
  "active_files": [...],
  "added": [...],
  "channels": [...],
  "channel_count": N
}
```

**2. Get Channels**
```
GET /api/channels

Response:
{
  "channels": [
    {
      "id": "signal_name",
      "name": "display_name",
      "clean": "clean_name",
      "presence": "X/Y",
      "present_count": N
    }
  ],
  "count": N
}
```

**3. Analytics (Signal Plotting)**
```
POST /analytics
Content-Type: application/x-www-form-urlencoded

Parameters:
- signals: Comma-separated signal IDs
- include_time: "1" or "0"
- normalize: "1" or "0"
- max_points: Integer (default: 100000)
- want_fft: "1" or "0"
- want_hist: "1" or "0"

Response:
{
  "series": {
    "signal_id": {
      "name": "...",
      "timestamps": [...],
      "values": [...],
      "unit": "..."
    }
  },
  "stats": [...],
  "fft": {...},
  "hist": {...}
}
```

**4. Empirical Map Generation**
```
POST /api/compute_map
Content-Type: application/json

Body:
{
  "files": ["path/to/file1.mdf", "path/to/file2.mdf"],
  "rpm_bins": [100, 6000, 500] or "100:6000:500",
  "tq_bins": [0, 600, 50] or "0:600:50",
  "min_samples_per_bin": 3,
  "preset": "ci_engine_default",
  "enable_surface": true,
  "enable_contours": true,
  "interp_method": "linear",
  "smoothing": 0.5
}

Response:
{
  "tables": {
    "Map Summary": [...],
    "Signal Mapping": [...]
  },
  "plots": {
    "engine_bsfc_heatmap": {"plotly_json": "..."},
    "engine_bsfc_surface": {"plotly_json": "..."}
  },
  "meta": {
    "ok": true,
    "progress": [...],
    "missing_signals": {...}
  }
}
```

**5. Playground API (JSON Mode)**
```
POST /analytics
Content-Type: application/json

Body:
{
  "fn": "series",
  "ids": ["signal1", "signal2"],
  "tmin": 0.0,
  "tmax": 100.0,
  "downsample": 20000,
  "algo": "lttb"
}

Response:
{
  "ok": true,
  "series": {...}
}
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Files Not Uploading

**Symptoms**: Upload fails or files don't appear

**Solutions**:
1. **Check file size**: Must be < 500 MB
2. **Check format**: Must be .mdf, .mf4, .csv, .xlsx, .xls
3. **Check permissions**: Ensure uploads directory is writable
4. **Browser console**: Check for JavaScript errors (F12)

#### Issue 2: Signals Not Found

**Symptoms**: Channel table empty or signals missing

**Solutions**:
1. **Verify file format**: Ensure valid MDF/CSV/Excel file
2. **Check file integrity**: File might be corrupted
3. **Try different file**: Test with known good file
4. **Check logs**: Server logs show specific errors

#### Issue 3: Plots Not Rendering

**Symptoms**: Empty plot areas or errors

**Solutions**:
1. **Check signal selection**: Ensure signals are selected
2. **Check data**: Verify signals have valid data (not all NaN)
3. **Browser compatibility**: Use modern browser (Chrome, Firefox, Edge)
4. **Clear cache**: Hard refresh (Ctrl+Shift+R or Cmd+Shift+R)

#### Issue 4: Map Generation Fails

**Symptoms**: No plots in map section

**Solutions**:
1. **Check signals**: Ensure RPM and Torque signals are found
2. **Check bins**: Verify bin configuration is valid
3. **Check data coverage**: Need sufficient data points
4. **Min samples**: Lower min_samples_per_bin if needed
5. **Signal mapping**: Use overrides to map your signal names

#### Issue 5: CSV/Excel Not Reading Correctly

**Symptoms**: Wrong columns or data format issues

**Solutions**:
1. **Check structure**: Ensure header row is first row
2. **Time column**: Name time column "Time" or "timestamp"
3. **Encoding**: Try saving CSV as UTF-8
4. **Clean data**: Remove special characters from headers
5. **Manual mapping**: Use signal overrides if needed

#### Issue 6: Performance Issues

**Symptoms**: Slow loading, timeouts

**Solutions**:
1. **File size**: Split large files (>100 MB)
2. **Downsampling**: Use lower max_points value
3. **Signal selection**: Limit to essential signals
4. **Browser**: Close other tabs/applications
5. **Server resources**: Check server CPU/memory

### Signal Mapping Issues

#### Problem: RPM/Torque Not Found

**Solution**: Use signal overrides in Empirical Map section

**Steps**:
1. Go to Empirical Map
2. Use "Signal Overrides" or "Edit Mapping"
3. Map your signals:
   ```
   rpm: YourRPMSignalName
   torque: YourTorqueSignalName
   ```
4. Save and regenerate map

#### Problem: Wrong Signal Selected

**Solution**: Check signal aliases and naming

**Understanding Signal Names**:
- MDF files often have multiple aliases for same signal
- System tries all aliases automatically
- Check "Signal Mapping" table in results

---

## Best Practices

### File Preparation

#### For MDF Files:
1. **Clean data**: Remove corrupted or incomplete files
2. **Naming**: Use descriptive filenames with timestamps
3. **Coverage**: Include files covering full operating range
4. **Size**: Keep files < 100 MB for best performance

#### For CSV Files:
1. **Header format**: First row = column names
2. **Time column**: Include "Time" or "timestamp" column
3. **Data types**: Ensure numeric columns are numeric (not text)
4. **Encoding**: Use UTF-8 when possible
5. **Units**: Include units row if needed (system auto-detects)

#### For Excel Files:
1. **Sheet structure**: Keep data on first sheet
2. **Headers**: First row = column names
3. **No merged cells**: Flatten merged cells before export
4. **Data consistency**: Same structure across sheets

### Analysis Workflow

#### 1. Initial Upload & Verification
```
✓ Upload files
✓ Verify channels appear
✓ Check file sizes and counts
✓ Verify time columns detected
```

#### 2. Signal Exploration
```
✓ Browse all available signals
✓ Search for key signals (RPM, Torque, etc.)
✓ Check signal presence across files
✓ Note signal names for later use
```

#### 3. Quick Analysis
```
✓ Plot key signals to verify data quality
✓ Check for obvious anomalies
✓ Verify time ranges are correct
✓ Check signal units
```

#### 4. Deep Analysis
```
✓ Select relevant signals for analysis
✓ Use appropriate plot types
✓ Apply time filtering if needed
✓ Generate statistics
```

#### 5. Map Generation
```
✓ Select files with good coverage
✓ Configure bins based on operating range
✓ Start with presets, then customize
✓ Verify coverage percentage
✓ Export results for further use
```

### Performance Optimization

#### For Large Files:
1. **Pre-filter**: Extract only needed channels before upload
2. **Time filtering**: Use tmin/tmax to limit data range
3. **Downsampling**: Use lower max_points (5000-10000)
4. **Signal selection**: Limit to essential signals only

#### For Multiple Files:
1. **Merge strategy**: Use "append" mode for incremental uploads
2. **File grouping**: Upload related files together
3. **Clear cache**: Use "Replace" mode to clear old data

### Data Quality

#### Signal Validation:
1. **Check units**: Verify units match expectations
2. **Range checks**: Look for impossible values
3. **Missing data**: Check for excessive NaN values
4. **Time gaps**: Verify continuous time series

#### Map Quality:
1. **Coverage**: Aim for >70% map coverage
2. **Sample count**: Ensure sufficient samples per bin
3. **Edge cases**: Check boundary conditions
4. **Smoothing**: Use appropriate smoothing for noise reduction

---

## Advanced Tips

### 1. Custom Signal Aliases

If your signals use non-standard names, the system uses fuzzy matching. For best results:
- Include common abbreviations (e.g., "RPM", "rpm", "nEng")
- Use signal override feature in Empirical Map
- Check "Signal Mapping" table to see what was matched

### 2. Optimal Bin Configuration

**Rule of Thumb**:
- **RPM bins**: Match your calibration table resolution
- **Torque bins**: 10-50 N·m steps (depends on torque range)
- **Coverage**: Wider bins = better coverage, less detail
- **Balance**: Find balance between coverage and resolution

**Example Calculations**:
```
Operating Range: 800-4000 RPM, 50-500 N·m
Optimal bins:
  RPM: 800:4000:200 (17 bins)
  Torque: 50:500:25 (19 bins)
  Total cells: 323
  Target coverage: >70% = 226 cells with data
```

### 3. Signal Overrides

When automatic detection fails:

**In Empirical Map**:
1. Use "Edit Mapping" or signal override fields
2. Manually map signals:
   ```
   rpm: YourActualRPMSignal
   torque: YourActualTorqueSignal
   bsfc_gpkwh: YourBSFCSignal
   ```

**Via API**:
```json
{
  "overrides": {
    "rpm": "Epm_nEng",
    "torque": "TqSys_tqCkEngReal_RTE"
  }
}
```

### 4. Working with CSV Time Columns

**Detected Automatically**:
- Columns named: "Time", "timestamp", "t", "datetime"
- First numeric column (if sequential)

**Manual Override**:
- Rename time column to "Time" before upload
- Or use index-based time (set `include_time=false`)

### 5. Multi-File Strategies

**For Map Generation**:
1. **Steady-state files**: Best for map generation
2. **Transient files**: Include for edge cases
3. **Merge strategy**: All files merged automatically
4. **Weighting**: System averages across files

**For Analysis**:
1. **Group by test**: Upload related files together
2. **Compare runs**: Use different file sets for comparison
3. **Time alignment**: System handles time synchronization

### 6. Export Strategies

**Map Export**:
- **CSV**: For ECU calibration import
- **XLSX**: For Excel-based tools
- **PNG**: For documentation/reports

**Signal Export**:
- Use statistics table copy feature
- Or export raw data via API

### 7. Debugging Signal Issues

**Check Signal Availability**:
```javascript
// In browser console (F12):
fetch('/api/channels').then(r => r.json()).then(console.log)
```

**Check File Content**:
```python
# In Python:
from app import list_channels, read_signal
channels = list_channels(Path('uploads/file.mdf'))
t, v, u = read_signal(Path('uploads/file.mdf'), 'Epm_nEng')
```

### 8. Performance Tuning

**Server-side** (app.py):
- Adjust `MAX_UPLOAD_SIZE_MB` for larger files
- Modify `UPLOAD_CHUNK_SIZE` for upload speed
- Adjust cache sizes if memory constrained

**Client-side**:
- Use downsampling for large datasets
- Limit signal selection to essentials
- Close unused browser tabs

---

## Technical Details

### Signal Name Normalization

The system automatically creates aliases for signals:
- Original name: `MG1CS051_H440_2F.Epm_nEng`
- Aliases: `Epm_nEng`, `epm_neng`, `epm_n_eng`, etc.
- URL encoding variants
- Cleaned versions (spaces, special chars removed)

**This means**: You can search/reference signals by any of these names.

### Data Caching

**Channel Cache**: 
- Cached per file path
- Speeds up repeated channel listing
- Cleared on file replacement

**Series Cache**:
- Cached per (file_set, signal_set) combination
- Speeds up repeated plotting
- Cleared on file replacement or mode change

### Time Handling

**MDF Files**:
- Uses native MDF timestamps
- Preserves original time units
- Handles time gaps automatically

**CSV/Excel Files**:
- Automatic time column detection
- Falls back to index if no time column
- Supports numeric and datetime formats

### Memory Management

**Large Files**:
- Processed in chunks (CHUNK_SIZE = 100,000)
- Memory-mapped MDF reading for >500 MB files
- Downsampling applied automatically

---

## Quick Reference

### Keyboard Shortcuts

- **Upload area**: Press Enter to open file picker
- **Search**: Type in channel search box to filter
- **Select All**: Use checkbox in table header
- **Plot**: Press Enter in signal selection (if available)

### File Formats Summary

| Format | Extension | Time Support | Unit Support | Best For |
|--------|-----------|--------------|--------------|----------|
| MDF v4 | .mf4 | ✅ Native | ✅ Yes | Vehicle diagnostics |
| MDF v3 | .mdf | ✅ Native | ✅ Yes | Legacy data |
| CSV | .csv | ✅ Auto-detect | ⚠️ Partial | External data |
| Excel | .xlsx, .xls | ✅ Auto-detect | ⚠️ Partial | Office workflows |

### Signal Naming Patterns

**Common RPM Signals**:
- `Epm_nEng`, `Ext_nEng_RTE`, `inRpm`, `EngSpeed`, `rpm`

**Common Torque Signals**:
- `TqSys_tqCkEngReal`, `TqSys_tqCkEngReal_RTE`, `inTorque`, `Torque`

**Common Fuel Signals**:
- `FuelRate`, `FuCns_volFuCnsTot`, `FuelCons`, `fuel_flow`

---

## Support & Resources

### Log Files

**Server Logs**: Check terminal output when running dashboard
**Diagnostic Files**: Check `uploads/` for `.error.txt` files if issues occur

### Common Signal Mappings

**RPM**:
- `Epm_nEng` ✅ (common)
- `Ext_nEng_RTE` ✅ (common)
- `inRpm` ✅ (some systems)
- `EngSpeed`, `rpm` ✅ (generic)

**Torque**:
- `TqSys_tqCkEngReal_RTE` ✅ (common)
- `TqSys_tqCkEngReal` ✅ (common)
- `inTorque` ✅ (some systems)
- `EngineTorque`, `torque` ✅ (generic)

### Getting Help

1. **Check logs**: Server terminal shows detailed errors
2. **Browser console**: F12 → Console tab for frontend errors
3. **File structure**: Verify file format matches expectations
4. **Test with sample**: Try with known good file first

---

## Version History

**v2.0 (Current)**:
- ✅ CSV and Excel support
- ✅ Improved signal detection
- ✅ Fixed empirical map plots
- ✅ Enhanced time column detection

**v1.0**:
- ✅ MDF/MF4 support
- ✅ Basic analytics
- ✅ Empirical maps
- ✅ Report generation

---

## Appendix

### A. Required Dependencies

**Core** (required):
- Flask 2.3.3
- pandas
- numpy
- plotly
- asammdf (for MDF files)
- openpyxl (for Excel files)

**Optional** (for advanced features):
- scipy (for map smoothing/interpolation)
- scikit-learn (for Calibration AI)
- scikit-optimize (for optimization)
- optuna (for hyperparameter optimization)

### B. File Structure

```
backend_mdf/
├── app.py                 # Main Flask application
├── frontend.html          # Web interface
├── custom_map.py          # Map generation module
├── custom_dfc.py          # DFC analysis
├── custom_cc_sl.py        # CC/SL analysis
├── custom_iupr.py         # IUPR analysis
├── cie.py                 # Calibration AI
├── launch_dashboard.py    # Dashboard launcher
├── uploads/               # Uploaded files directory
├── maps_outputs/          # Generated map outputs
├── tmp_plots/            # Temporary plot files
└── models/               # Saved ML models
```

### C. Configuration

**Environment Variables** (can be set):
- `MAX_UPLOAD_SIZE_MB`: Maximum file size (default: 500)
- `UPLOAD_CHUNK_SIZE`: Upload chunk size
- Various analysis thresholds

---

**End of Guide**

For questions or issues, refer to the Troubleshooting section or check server logs for detailed error messages.

