# Dashboard Quick Reference Card

## Startup
```bash
cd /path/to/backend_mdf
python3 launch_dashboard.py
# Open: http://localhost:8000
```

## File Upload
- **Drag & drop** or **click to browse**
- **Supported**: `.mdf`, `.mf4`, `.csv`, `.xlsx`, `.xls`
- **Max size**: 500 MB per file

## Signal Selection
1. **Browse** channels or **search** by name
2. **Check** signals to analyze
3. **Click** "Plot Signals"

## Plot Options
- **Include Time**: Use timestamps (default: ON)
- **Normalize**: Scale to 0-1 range
- **Max Points**: Limit for performance (default: 10,000)
- **FFT**: Frequency analysis
- **Histogram**: Distribution plots

## Empirical Map Generation

### Quick Preset
1. Go to **Report** → **Empirical Map**
2. Select **preset** (e.g., "CI Engine — BSFC")
3. Configure **bins** (e.g., RPM: `100:6000:500`)
4. Click **Generate Map**

### Manual Configuration
```
X-Axis: RPM signal (e.g., Epm_nEng)
Y-Axis: Torque signal (e.g., TqSys_tqCkEngReal_RTE)
Z-Axis: Target parameter (e.g., BSFC)

RPM Bins: 100:6000:500  (start:stop:step)
Torque Bins: 0:600:50

Min Samples: 3-8 (higher = more reliable)
```

### Bin Format
- **Range**: `start:stop:step` (e.g., `100:6000:500`)
- **Custom**: `0,500,1000,1500` (comma-separated)

## Common Signal Names

| Signal Type | Common Names |
|-------------|--------------|
| **RPM** | `Epm_nEng`, `Ext_nEng_RTE`, `inRpm`, `EngSpeed` |
| **Torque** | `TqSys_tqCkEngReal_RTE`, `inTorque`, `EngineTorque` |
| **Fuel** | `FuelRate`, `FuCns_volFuCnsTot`, `FuelCons` |
| **Lambda** | `lambda`, `afr`, `AirFuelRatio`, `Lambda` |
| **Temp** | `ECT_C`, `IAT_C`, `TExh`, `CoolantTemp` |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Files not uploading** | Check size (<500 MB), format, permissions |
| **Signals not found** | Check file format, verify integrity |
| **Plots not showing** | Select signals, check browser console (F12) |
| **Map generation fails** | Verify RPM/Torque signals found, check bins |
| **CSV/Excel issues** | Check header row, time column name, encoding |

## Signal Override (If Auto-Detection Fails)
```
In Empirical Map:
- Use "Edit Mapping" or signal override fields
- Map: rpm → YourRPMSignalName
- Map: torque → YourTorqueSignalName
```

## Performance Tips
- **Large files**: Use downsampling (10,000-20,000 points)
- **Multiple files**: Upload related files together
- **Time filtering**: Use tmin/tmax to limit range

## Export Formats
- **CSV**: Tabular data
- **XLSX**: Excel workbook
- **PNG**: Plot images
- **JSON**: Raw data export

## Keyboard Shortcuts
- **Upload**: Enter in upload area
- **Search**: Type in channel search box
- **Select All**: Checkbox in table header

## Best Practices
1. ✅ **File preparation**: Clean data, descriptive names
2. ✅ **Signal verification**: Plot key signals first
3. ✅ **Map bins**: Match calibration table resolution
4. ✅ **Coverage**: Aim for >70% map coverage
5. ✅ **Export**: Save results for further analysis

## API Endpoints (Advanced)

```
POST /smart_merge_upload          # Upload files
GET  /api/channels                # List channels
POST /analytics                   # Plot signals
POST /api/compute_map             # Generate maps
```

## Quick Workflows

### 1. Basic Analysis
```
Upload → Select Signals → Plot → View Statistics
```

### 2. Map Generation
```
Upload Files → Select Preset → Configure Bins → Generate → Export
```

### 3. Fault Investigation
```
Upload → DFC Analysis → Correlate Signals → Time Filter → Analyze
```

### 4. Multi-File Comparison
```
Upload Multiple Files → Select Common Signals → Compare Plots
```

---

**For detailed guide, see**: `DASHBOARD_USER_GUIDE.md`

