# Fuel Consumption Analysis Module

## Overview

A comprehensive fuel consumption analysis tool for vehicle calibration engineers, providing professional-grade analysis of fuel consumption from MDF/MF4 data loggers. This module follows the same architecture as DFC and IUPR modules, integrating seamlessly into the dashboard.

## Features

### Core Analysis Capabilities

1. **BSFC (Brake Specific Fuel Consumption) Analysis**
   - Calculates BSFC in g/kWh from fuel mass flow and mechanical power
   - Operating point analysis (RPM vs Torque bins)
   - Statistical distribution analysis

2. **Fuel Flow Rate Analysis**
   - Volume consumption (L/h)
   - Mass flow rate (kg/s, g/s)
   - Automatic unit detection and conversion

3. **Distance-Based Consumption**
   - L/100km calculation (when distance and speed data available)
   - Fuel consumption correlation with vehicle speed

4. **Statistical Summaries**
   - Mean, median, standard deviation
   - Percentiles (P25, P75, P95)
   - Min/Max values
   - Sample counts for all metrics

5. **Operating Point Analysis**
   - RPM-Torque binning
   - BSFC distribution across operating points
   - Sample count per bin
   - Average fuel flow per operating region

### Visualizations

1. **BSFC vs Operating Points**
   - Scatter plot with color-coded BSFC values
   - RPM vs Torque visualization
   - Interactive hover information

2. **Fuel Flow Time Series**
   - Volume flow (L/h) over time
   - Mass flow (g/s) over time (dual y-axis)
   - Temporal analysis of fuel consumption

3. **BSFC Distribution Histogram**
   - Distribution of BSFC values
   - Percentile markers (Median, P25, P75)
   - Statistical insights

4. **Speed vs Fuel Consumption**
   - Correlation analysis
   - Trend line overlay
   - Distance-based consumption metrics

5. **Fuel Efficiency Map**
   - 2D heatmap of BSFC across RPM-Torque space
   - Optimal operating region identification
   - Calibration insights

## Signal Detection

The module automatically detects fuel-related signals using common OEM naming patterns:

- **Fuel Consumption**: `FuelRate`, `FuCns_volFuCnsTot`, `FuelCons`, `fuel_flow`
- **RPM**: `Epm_nEng`, `EngineSpeed`, `rpm`, `nEng`
- **Torque**: `TqSys_tqCkEngReal`, `EngineTorque`, `torque`
- **Speed**: `VehSpd`, `VehicleSpeed`, `speed`
- **Distance**: `Distance`, `Odometer`, `TotalDistance`
- **Air Flow**: `MAF`, `air_mass_flow`, `InM_mfAirCanPurgEstim`
- **Lambda/AFR**: `lambda`, `AFR`, `AirFuelRatio`

## Integration

### Backend (`app.py`)
- `_build_fuel_payload()` function builds the response
- Integrated into `/api/report_section` endpoint
- Handles errors gracefully with diagnostic messages

### Frontend (`frontend.html`)
- New "Fuel Consumption" tab in Report section
- Uses generic report renderer for tables and plots
- Fully integrated with existing dashboard infrastructure

## Usage

1. Upload MDF/MF4 files through the dashboard
2. Navigate to the "Report" tab
3. Click on "Fuel Consumption" subtab
4. The system will automatically:
   - Detect fuel-related channels
   - Extract and align signals
   - Calculate derived metrics (BSFC, power, etc.)
   - Generate statistical summaries
   - Create interactive visualizations

## Output Structure

```json
{
  "tables": {
    "Fuel Summary": [statistical metrics],
    "Operating Point Analysis": [binned RPM-Torque data],
    "Fuel Channels Found": [channel mapping]
  },
  "plots": {
    "BSFC vs Operating Points": {...},
    "Fuel Flow Time Series": {...},
    "BSFC Distribution": {...},
    "Speed vs Fuel Consumption": {...},
    "Fuel Efficiency Map": {...}
  },
  "meta": {
    "total_samples": count,
    "files_processed": count,
    "channels_found": count,
    "evidence": [channel detection logs]
  }
}
```

## Technical Details

### BSFC Calculation
```
BSFC (g/kWh) = (fuel_mass_flow_kgps × 3,600,000) / mech_power_kw
```

Where:
- `fuel_mass_flow_kgps` = fuel mass flow in kg/s
- `mech_power_kw` = torque (Nm) × angular_velocity (rad/s) / 1000

### Fuel Density
- Default: 0.745 kg/L (typical gasoline)
- Automatically converts volume to mass flow

### Signal Alignment
- All signals are interpolated to a common 1 Hz time grid
- Handles missing data gracefully
- Filters invalid/NaN values

## Calibration Engineering Insights

This tool provides valuable insights for:

1. **Engine Calibration**
   - Identify optimal operating regions
   - BSFC optimization opportunities
   - Fuel efficiency map generation

2. **Emission Analysis**
   - Correlation with air-fuel ratio
   - Impact of operating conditions on consumption

3. **Driving Cycle Analysis**
   - Fuel consumption patterns
   - Speed-efficiency relationships
   - Distance-based metrics

4. **Diagnostics**
   - Channel detection and mapping
   - Data quality assessment
   - Missing signal identification

## Module Architecture

```
custom_fuel.py
├── Signal Detection
│   ├── find_signal() - Pattern matching
│   └── FUEL_SIGNAL_PATTERNS - OEM naming conventions
├── Data Extraction
│   ├── extract_fuel_data() - Main extraction function
│   └── safe_read_signal() - Robust signal reading
├── Analysis
│   ├── compute_fuel_statistics() - Statistical summaries
│   └── compute_operating_point_analysis() - RPM-Torque binning
├── Visualization
│   ├── plot_bsfc_operating_points()
│   ├── plot_fuel_flow_timeseries()
│   ├── plot_bsfc_distribution()
│   ├── plot_speed_vs_fuel()
│   └── plot_fuel_efficiency_map()
└── Public API
    └── compute_fuel() - Main entry point
```

## Testing

Tested with:
- Real MDF files from vehicle data loggers
- Multiple file formats (MDF, MF4)
- Various signal naming conventions
- Missing data scenarios

## Future Enhancements

Potential improvements:
- Integration with emission data
- Multi-file comparison
- Fuel economy prediction models
- Custom calibration parameter inputs
- Export to calibration tools (ETAS INCA, Vector CANoe)

## Credits

Developed following the architecture of:
- `custom_dfc.py` - DFC analysis module
- `custom_iupr.py` - IUPR analysis module

Maintains consistency with existing dashboard patterns and provides professional-grade analysis capabilities for vehicle calibration engineers.

