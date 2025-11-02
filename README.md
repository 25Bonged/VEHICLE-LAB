# VEHICLELAB - Vehicle Diagnostic Lab

A comprehensive fleet diagnostics tool for analyzing vehicle MDF/MF4 files with advanced features for gear hunting detection, misfire analysis, fuel consumption monitoring, and more.

## Features

- **Gear Hunting Analysis**: Advanced multi-signal correlation with Speed, RPM, and Misfire integration
- **Misfire Detection**: OEM-level detection using multiple algorithms (crankshaft variance, FFT, statistical, angular velocity, wavelet, ML-based)
- **Fuel Consumption Analysis**: BSFC calculations, operating point mapping, efficiency analysis
- **Empirical Mapping**: Dynamic signal mapping across multiple OEM formats
- **Dashboard Interface**: Modern web-based dashboard with interactive Plotly visualizations

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd backend_mdf
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the dashboard:
```bash
python3 launch_dashboard.py
```

4. Access the dashboard at `http://localhost:8000`

## Usage

1. Upload your MDF/MF4 files through the web interface
2. Select analysis sections (Gear Hunt, Misfire, Fuel Consumption, etc.)
3. View interactive plots and diagnostic reports
4. Export data as CSV/Excel if needed

## Project Structure

- `app.py` - Main Flask application
- `custom_gear.py` - Gear hunting detection and analysis
- `custom_misfire.py` - Misfire detection algorithms
- `custom_fuel.py` - Fuel consumption analysis
- `custom_map.py` - Empirical mapping system
- `signal_mapping.py` - Signal name mapping utilities
- `frontend.html` - Web dashboard interface
- `collected_dbc_files/` - Processed DBC file collection

## Documentation

See the various `.md` files for detailed documentation:
- `DASHBOARD_USER_GUIDE.md` - User guide
- `MISFIRE_SYSTEM_VERIFICATION.md` - Misfire detection details
- `FUEL_CONSUMPTION_ANALYSIS.md` - Fuel analysis documentation

## Requirements

- Python 3.8+
- asammdf (for MDF/MF4 file reading)
- Plotly (for interactive visualizations)
- See `requirements.txt` for complete list

## License

[Add your license here]

## Contributing

[Add contribution guidelines if needed]

