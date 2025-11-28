# 🚗 VEHICLE-LAB

**A comprehensive vehicle diagnostic and analysis platform for MDF/MF4 files**

🚀 Live Demo • 📖 Documentation • 💻 Features • 🔄 Workflow • 📚 Repository Info

> ⚠️ **Note:** This is a **documentation-only repository**. All source code (`.py` files) is **private and proprietary**. Only documentation, guides, and examples are included. See CODE_PRIVACY.md for details.

---

## 📖 Overview

**VEHICLE-LAB** is a professional-grade web-based diagnostic platform designed for automotive engineers, calibration specialists, and fleet managers. It combines traditional MATLAB-level analytical capabilities with cutting-edge AI-powered diagnostics through **DiagAI** - an intelligent LLM assistant that transforms how engineers interact with vehicle data.

### 🎯 What Makes It Special?

**VEHICLE-LAB** provides a complete ecosystem for vehicle data analysis:

* **🔬 OEM-Level Analysis**: 9 misfire detection algorithms, advanced gear hunting detection, comprehensive diagnostic agents
* **📊 Maps**: Empirical map generation with Kriging, RBF, and cubic spline interpolation - production-ready accuracy
* **🌐 Web-Based Dashboard**: No installation required, runs in any modern browser with real-time processing
* **📁 Multi-Format Support**: MDF, MF4, CSV, Excel (.xlsx, .xls) - handles all major automotive data formats
* **🚀 Real-Time Processing**: Interactive Plotly visualizations with LTTB downsampling for large datasets
* **🔍 Auto Signal Mapping**: Intelligent signal detection across 330+ DBC files from multiple OEMs
* **🤖 DiagAI**: Advanced LLM-powered vehicle data analysis with natural language queries - ask questions, get engineering-grade insights instantly

### 🍒 DiagAI: The Game Changer

**DiagAI** is not just a feature - it's the **crown jewel** that elevates VEHICLE-LAB from a powerful analysis tool to an intelligent diagnostic assistant. With DiagAI, engineers can:

* **Ask questions in plain English**: "What's the average RPM during acceleration?" or "Show me fuel efficiency patterns"
* **Get instant engineering insights**: Automatic signal detection, correlation analysis, anomaly identification
* **Access specialized knowledge**: 9 diagnostic agents working together for comprehensive analysis
* **Deploy privately with OEM-specific RAGs**: Complete data isolation for enterprise security

**DiagAI transforms vehicle diagnostics from manual analysis to conversational intelligence.**

---

## 🍒 DiagAI - AI-Powered Vehicle Diagnostics

**DiagAI** is the crown jewel of VEHICLE-LAB - an advanced AI assistant that transforms vehicle data analysis through natural language interaction.

### 🧠 What is DiagAI?

DiagAI is an intelligent diagnostic assistant powered by Large Language Models (LLM) that enables engineers to:

* **Ask questions in natural language**: "What's the average RPM during acceleration?" or "Show me fuel efficiency patterns"
* **Get instant insights**: Automatic signal detection, correlation analysis, and anomaly identification
* **Receive engineering-grade analysis**: MATLAB-level insights with professional formatting
* **Access specialized knowledge**: 9 diagnostic agents for misfire, gear hunting, fuel analysis, and more

### 🔐 Enterprise-Grade Security for OEMs

**Private LLM Deployment with OEM-Specific RAGs**

* **🔒 Private LLM Infrastructure**: Deploy your own private LLM instance (DeepSeek, LM Studio, or custom models) - your data never leaves your infrastructure
* **📚 OEM-Specific RAG (Retrieval Augmented Generation)**: Each OEM gets isolated knowledge bases with:
  - Proprietary calibration data
  - Brand-specific diagnostic procedures
  - Custom signal mappings
  - Internal engineering documentation
* **🛡️ Complete Data Isolation**: OEM data is completely isolated - no cross-contamination between different manufacturers
* **🔐 Secure Multi-Tenancy**: Enterprise-ready architecture supporting multiple OEMs with strict data boundaries

### 🚀 DiagAI Capabilities

#### Intelligent Signal Discovery
* **1,817+ signals** automatically indexed from uploaded files
* **155 alias patterns** for intelligent signal matching
* **330+ DBC files** integrated for cross-OEM compatibility
* **Smart pattern recognition**: Handles case variations, partial matches, and fuzzy matching

#### Natural Language Queries
```
"List all available signals"
"Show me RPM statistics"
"Analyze fuel efficiency patterns"
"Compare torque vs throttle correlation"
"Detect misfire events in this data"
"Generate comprehensive diagnostic report"
```

#### Multi-Agent Diagnostic System
1. **MisfireAgent** - Engine misfire detection (CSVA, FFT, ML-based)
2. **DFCAgent** - Diagnostic Trouble Code analysis
3. **IUPRAgent** - In-Use Performance Ratio (OBD-II compliance)
4. **GearAgent** - Gear hunting & transmission analysis
5. **DrivabilityAgent** - Anti-jerk, torque filter, pedal map
6. **StartWarmupAgent** - Engine start & warmup analysis
7. **WLTPAgent** - WLTP cycle & emissions testing
8. **FuelAgent** - Fuel consumption & BSFC calculation
9. **CANBusOffAgent** - CAN bus diagnostics

#### Advanced Analysis Functions
* **Statistical Analysis**: Mean, median, percentiles, distributions
* **Correlation Analysis**: Pearson, Spearman, correlation matrices
* **Anomaly Detection**: Automatic outlier identification
* **Time Series Analysis**: Trend detection, pattern recognition
* **Comparative Analysis**: Multi-signal comparisons
* **Empirical Map Generation**: BSFC, efficiency islands, operating point analysis

### 🔧 LLM Architecture

**Multi-LLM Support with Smart Routing:**
* **DeepSeek** (Primary) - Cloud or local deployment, 8B-67B models
* **LM Studio** (Secondary) - Local deployment, 7B-13B models
* **Ollama** (Fallback) - Open-source local models
* **Trained Model** (Specialized) - Automotive-specific fine-tuned models

**Smart Query Routing:**
* Complex queries → DeepSeek (high-quality reasoning)
* Simple queries → LM Studio (fast responses)
* Automotive-specific → Trained model (specialized knowledge)
* Automatic failover between models

---

## 🚀 Live Demo

> **🌐 Try it online:** Coming Soon - Deploy to Render/Railway/Heroku

### 📸 Dashboard Preview

#### **Core Dashboard Features**

| **Main Interface**                                                                    | **Signal Analysis**                                                                     | **Empirical Maps**                                                                  |
| ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| ![Main Dashboard](docs/screenshots/dashboard-main.png)                               | ![Signal Analysis](docs/screenshots/signal-analysis.png)                                | ![Empirical Map](docs/screenshots/empirical-map.png)                                |
| Upload MDF/MF4 files, browse signals, real-time processing                           | Interactive Plotly charts, signal correlation, statistical analysis                     | 2D heatmaps, 3D surface plots, efficiency islands, operating point visualization    |

#### **Advanced Analysis Features**

| **Misfire Detection**                                                                 | **Gear Hunting Analysis**                                                                | **Fuel Analysis**                                                                     |
| ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| ![Misfire Detection](docs/screenshots/misfire.png)                                    | ![Gear Hunt](docs/screenshots/gear-hunt.png)                                            | ![Fuel Analysis](docs/screenshots/fuel-analysis.png)                                 |
| 9 detection algorithms, severity analysis, event timeline                            | Automatic detection, speed-RPM correlation, hunting event visualization                | BSFC maps, fuel consumption metrics, efficiency analysis                            |

#### **3D Visualization**

| **3D Empirical Maps**                                                                 |
| ------------------------------------------------------------------------------------- |
| ![3D Empirical Map](docs/screenshots/empirical-map-3d.png)                            |
| Interactive 3D surface plots with rotation, zoom, and export capabilities           |

---

## 🎯 Key Features

### 🤖 DiagAI - AI-Powered Diagnostics 

**Natural Language Interface**
* Ask questions in plain English: "What's the average RPM during acceleration?"
* Get instant engineering-grade insights with MATLAB-level accuracy
* Automatic signal detection from natural language queries
* Context-aware responses that understand automotive terminology

**Intelligent Analysis**
* **1,817+ signals** automatically indexed and searchable
* **155 alias patterns** for intelligent signal matching
* **9 specialized diagnostic agents** working in coordination:
  - MisfireAgent, DFCAgent, IUPRAgent
  - GearAgent, DrivabilityAgent, StartWarmupAgent
  - WLTPAgent, FuelAgent, CANBusOffAgent
* **Multi-LLM support** with smart routing (DeepSeek, LM Studio, Ollama)

**Enterprise Security**
* **Private LLM deployment** - your data never leaves your infrastructure
* **OEM-specific RAG systems** - isolated knowledge bases per manufacturer
* **Complete data isolation** - zero cross-contamination between OEMs
* **Secure multi-tenancy** - enterprise-ready with strict access controls

### 📊 Data Processing & Management

**Multi-Format Support**
* MDF, MF4, CSV, Excel (.xlsx, .xls) - all major automotive formats
* Automatic signal extraction and indexing on upload
* Real-time processing with fast signal extraction
* Large file handling optimized for files up to several GB

**Intelligent Signal Mapping**
* **330+ DBC files** integrated from multiple OEMs
* **1,817+ signals** automatically cataloged with metadata
* Smart pattern recognition: case-insensitive, fuzzy matching, partial matches
* Cross-OEM compatibility with intelligent signal aliasing

### 🔬 Advanced Analytics

**Empirical Map Generation**
* BSFC (Brake Specific Fuel Consumption) maps with Kriging, RBF, cubic spline interpolation
* Efficiency islands visualization
* Operating point analysis
* 2D heatmaps and 3D surface plots

**Specialized Diagnostics**
* **Misfire Detection**: 9 algorithms (CSVA, FFT, ML-based, statistical)
* **Gear Hunting Analysis**: Automatic detection and visualization
* **Fuel Consumption Analysis**: Comprehensive fuel metrics and BSFC calculation
* **WLTP Cycle Analysis**: Complete WLTP testing and emissions analysis
* **Correlation Analysis**: Multi-signal correlation matrices and heatmaps

**Engine Analysis**
* CI (Compression Ignition/Diesel) engine support
* SI (Spark Ignition/Gasoline) engine support
* Power calculation from RPM and torque
* Efficiency metrics and optimization recommendations

### 📈 Visualization & Reporting

**Interactive Charts**
* Plotly-powered visualizations with zoom, pan, export capabilities
* Real-time interaction with large datasets
* LTTB downsampling for efficient rendering

**Comprehensive Plot Types**
* **2D Heatmaps**: Engine maps, efficiency islands, operating regions
* **3D Surface Plots**: BSFC surfaces, power maps, torque curves
* **Time Series Plots**: Multi-signal overlays, dual-axis support
* **Statistical Plots**: Distributions, histograms, box plots
* **Correlation Matrices**: Multi-signal relationship visualization

**Professional Reports**
* PDF export with comprehensive diagnostics
* Customizable report sections
* Engineering-grade formatting
* Export buttons for all analysis sections

---

## 🔄 Complete Workflow

### Step 1: File Upload & Automatic Indexing

**Upload Your Data**
1. Upload MDF/MF4/CSV/Excel files via web interface
2. Files are automatically saved to secure storage
3. **Automatic signal extraction** happens immediately:
   - All signals extracted using `asammdf` library
   - Signals stored in DuckDB database with complete metadata:
     - Signal name (canonical)
     - Time series data
     - Units of measurement
     - ECU source (ECM, TCM, ABS, etc.)
     - CAN bus information (CAN, LIN, FlexRay)
   - **1,817+ signals** automatically cataloged
   - **155 alias patterns** mapped for intelligent matching

**Result**: Signals are immediately available for analysis - no manual ingestion needed!

### Step 2: Choose Your Analysis Method

**Option A: Traditional Dashboard Analysis**
- Navigate to specific analysis sections (Misfire, Gear Hunt, Fuel, etc.)
- Select signals from dropdown menus
- Configure analysis parameters
- Generate visualizations and reports

**Option B: DiagAI - Natural Language Analysis (Recommended)**
- Open DiagAI chat interface
- Ask questions in plain English
- Get instant insights with engineering-grade analysis

### Step 3: DiagAI Workflow (The Cherry on Top)

**Natural Language Query**
```
User: "What's the average RPM during acceleration?"
```

**Behind the Scenes:**
1. **Signal Detection**: DiagAI identifies "RPM" and "acceleration" keywords
2. **Signal Matching**: Finds matching signals using intelligent aliasing (1,817+ signals, 155 patterns)
3. **Data Fetching**: Automatically retrieves time series data for relevant signals
4. **Context Enhancement**: Adds domain knowledge from RAG system (OEM-specific if configured)
5. **LLM Analysis**: Multi-LLM system analyzes data with smart routing:
   - Complex queries → DeepSeek (high-quality reasoning)
   - Simple queries → LM Studio (fast responses)
   - Automotive-specific → Trained model (specialized knowledge)
6. **Agent Coordination**: Relevant diagnostic agents provide specialized insights
7. **Response Generation**: Engineering-grade analysis with:
   - Statistical summaries
   - Correlation analysis
   - Anomaly detection
   - Optimization recommendations
   - Professional formatting

**Result**: Instant engineering insights without manual data manipulation!

### Step 4: Advanced Analysis Examples

**Complex Conditional Queries**
```
User: "Show me data where rpm<2000 & torque>150"
→ DiagAI automatically:
   - Filters data with multi-signal conditions
   - Generates instant visualization
   - Provides statistical summary
   - Identifies operating patterns
```

**Comprehensive Diagnostics**
```
User: "Analyze engine performance and detect any issues"
→ DiagAI coordinates multiple agents:
   - MisfireAgent: Checks for misfire events
   - FuelAgent: Analyzes fuel efficiency
   - GearAgent: Detects gear hunting
   - DrivabilityAgent: Evaluates drivability metrics
   - Generates comprehensive diagnostic report
```

**Fleet Comparison**
```
User: "Compare fuel efficiency across my fleet"
→ DiagAI:
   - Identifies all vehicles in database
   - Compares fuel metrics across trips
   - Identifies patterns and differences
   - Provides optimization recommendations
```

### Step 5: Export & Share Results

**Export Options**
- **PDF Reports**: Professional diagnostic reports with all analysis
- **CSV/Excel**: Raw data export for further analysis
- **Interactive Charts**: Export Plotly visualizations
- **DiagAI Conversations**: Save chat history for documentation

### 🔄 Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   1. Upload Files                            │
│              (MDF/MF4/CSV/Excel)                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           2. Automatic Signal Indexing                       │
│   • Extract all signals                                     │
│   • Store in DuckDB with metadata                           │
│   • Map 155 alias patterns                                  │
│   • Update signal catalog (1,817+ signals)                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│        3. Choose Analysis Method                             │
│   ┌──────────────────┐      ┌──────────────────┐           │
│   │  Dashboard UI    │  OR  │   DiagAI Chat    │           │
│   │  (Traditional)   │      │  (AI-Powered)    │           │
│   └──────────────────┘      └──────────────────┘           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│        4. DiagAI Processing (If Using AI)                    │
│   • Natural language understanding                           │
│   • Signal detection & matching                              │
│   • Data fetching from database                             │
│   • RAG context enhancement (OEM-specific)                    │
│   • Multi-LLM analysis with smart routing                    │
│   • Multi-agent coordination                                 │
│   • Engineering insight generation                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│        5. Results & Insights                                 │
│   • Statistical summaries                                    │
│   • Visualizations (Plotly charts)                          │
│   • Diagnostic recommendations                               │
│   • Anomaly detection                                        │
│   • Optimization suggestions                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│        6. Export & Share                                     │
│   • PDF reports                                             │
│   • CSV/Excel data export                                   │
│   • Interactive chart export                                 │
│   • Conversation history                                     │
└─────────────────────────────────────────────────────────────┘
```

### 🚀 Quick Start Workflow

**For First-Time Users:**
1. **Upload** your first MDF/MF4 file
2. **Wait** for automatic signal indexing (happens in background)
3. **Open DiagAI** chat interface
4. **Ask**: "List all available signals" to see what's available
5. **Ask**: "Show me RPM statistics" to get instant insights
6. **Explore** traditional dashboard sections for specific analyses
7. **Export** results as PDF or CSV for documentation

**For Power Users:**
1. Upload multiple files for fleet analysis
2. Use DiagAI for complex queries: "Compare fuel efficiency across all vehicles"
3. Generate comprehensive reports with multiple diagnostic agents
4. Export everything for further analysis in MATLAB or other tools

---

## 🏗️ Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Client (Web Browser)                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐             │
│  │   HTML5    │  │ JavaScript │  │   Plotly  │             │
│  │   CSS3     │  │  (ES6+)    │  │ Charts    │             │
│  └────────────┘  └────────────┘  └────────────┘             │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/REST API
┌────────────────────────▼────────────────────────────────────┐
│              Flask Backend Server (Python)                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Core Application (app.py)                          │   │
│  │  - File Upload & Processing                         │   │
│  │  - Signal Extraction & Caching                     │   │
│  │  - RESTful API Endpoints                            │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  DiagAI - LLM-Powered Analysis                      │   │
│  │  ├─ Multi-LLM Support (DeepSeek, LM Studio, etc.)   │   │
│  │  ├─ Private LLM Deployment (OEM-specific)          │   │
│  │  ├─ RAG System (OEM-isolated knowledge bases)      │   │
│  │  ├─ Natural Language Processing                     │   │
│  │  └─ Smart Query Routing                             │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Analysis Modules                                    │   │
│  │  ├─ custom_map.py (Empirical Maps)                  │   │
│  │  ├─ custom_misfire.py (Misfire Detection)          │   │
│  │  ├─ custom_gear.py (Gear Hunt)                     │   │
│  │  ├─ custom_fuel.py (Fuel Analysis)                 │   │
│  │  └─ custom_iupr.py, custom_dfc.py, etc.              │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Multi-Agent Diagnostic System                      │   │
│  │  ├─ MisfireAgent, DFCAgent, IUPRAgent              │   │
│  │  ├─ GearAgent, DrivabilityAgent                    │   │
│  │  ├─ StartWarmupAgent, WLTPAgent                    │   │
│  │  └─ FuelAgent, CANBusOffAgent                      │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Data Processing Layer                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │  asammdf   │  │  pandas    │  │  numpy     │            │
│  │ (MDF/MF4)  │  │ (DataFrames│  │ (Arrays)   │            │
│  └────────────┘  └────────────┘  └────────────┘            │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │  scipy     │  │ scikit-learn│  │ signal_mapping│        │
│  │ (Interpolation)│ (ML)      │  │ (DBC files)│            │
│  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│              Private LLM Infrastructure (OEM-Specific)      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │  DeepSeek  │  │ LM Studio  │  │  Custom    │            │
│  │  (Private) │  │  (Local)   │  │  Models    │            │
│  └────────────┘  └────────────┘  └────────────┘            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  RAG System (OEM-Isolated Knowledge Bases)          │   │
│  │  ├─ OEM A: Proprietary calibration data            │   │
│  │  ├─ OEM B: Brand-specific diagnostics              │   │
│  │  └─ OEM C: Custom signal mappings                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔐 Security & Privacy

### Enterprise-Grade Data Protection

* **🔒 Private LLM Deployment**: Your data never leaves your infrastructure
* **🛡️ OEM Data Isolation**: Complete separation between different manufacturers
* **📚 Isolated RAG Systems**: Each OEM has dedicated knowledge bases
* **🔐 Secure Multi-Tenancy**: Enterprise-ready with strict access controls
* **🚫 No Data Sharing**: Zero cross-contamination between OEMs
* **✅ Compliance Ready**: Meets enterprise security and privacy requirements

---

## 🚀 Deployment

### Deploy to Render

1. **Create `render.yaml`:**

```yaml
services:
  - type: web
    name: vehicle-lab
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python3 app.py
    envVars:
      - key: PORT
        value: 8000
```

1. **Deploy:**  
   * Connect GitHub repository to Render  
   * Render will auto-deploy on push

### Deploy to Railway

1. **Create `Procfile`:**

```
web: python3 app.py
```

1. **Deploy:**  
   * Connect GitHub repository to Railway  
   * Railway auto-detects Python and deploys

### Deploy to Heroku

1. **Create `Procfile`:**

```
web: python3 app.py
```

1. **Create `runtime.txt`:**

```
python-3.10.0
```

1. **Deploy:**

```bash
heroku create vehicle-lab
git push heroku main
```

### Update app.py for Production

Modify `app.py` to use environment variable for port:

```python
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

---

## 📚 Documentation

Comprehensive documentation is available:

* **📖 Complete User Guide** - Full walkthrough for all features
* **🔬 Empirical Map Guide** - Map generation tutorial
* **⚡ Misfire Detection** - Misfire algorithms explained
* **⛽ Fuel Analysis** - Fuel consumption metrics
* **🗺️ Signal Mapping** - DBC file integration
* **🤖 DiagAI Guide** - LLM-powered analysis documentation
* **📊 Project Summary** - Technical overview

---

## 🎬 Usage Examples

### Example 1: Generate BSFC Map

```python
# 1. Upload MDF file via web interface
# 2. Navigate to "Empirical Map" section
# 3. Select:
#    - Template: BSFC (Brake Specific Fuel Consumption)
#    - X-axis: Engine RPM
#    - Y-axis: Engine Load
#    - Interpolation: Kriging
# 4. Click "Generate Map"
# 5. View 2D heatmap and 3D surface plot
```

### Example 2: Detect Misfire Events

```python
# 1. Upload MDF file with crankshaft data
# 2. Navigate to "Misfire Detection" section
# 3. System automatically:
#    - Runs 9 detection algorithms
#    - Identifies misfire events
#    - Generates severity plots
# 4. Review events with timestamps
```

### Example 3: Analyze with DiagAI

```python
# 1. Upload MDF file
# 2. Open DiagAI chat interface
# 3. Ask natural language questions:
#    - "What's the average RPM during acceleration?"
#    - "Show me fuel efficiency patterns"
#    - "Detect any anomalies in the engine data"
#    - "Compare torque vs throttle correlation"
# 4. Get instant engineering-grade insights
```

### Example 4: Analyze Gear Hunting

```python
# 1. Upload MDF file with transmission data
# 2. Navigate to "Gear Hunt" section
# 3. System automatically:
#    - Correlates Speed and RPM signals
#    - Detects hunting events
#    - Visualizes with time-series plots
```

---

## 🛠️ API Reference

### Core Endpoints

#### Upload File

```
POST /api/upload
Content-Type: multipart/form-data

Response: {
  "success": true,
  "filename": "example.mdf",
  "size": 1024000
}
```

#### Get Channels

```
GET /api/files/{filename}/channels

Response: {
  "channels": ["EngineRPM", "VehicleSpeed", ...],
  "count": 150
}
```

#### Extract Signals

```
POST /api/analytics
Content-Type: application/json

{
  "filename": "example.mdf",
  "ids": ["EngineRPM", "VehicleSpeed"],
  "tmin": 0,
  "tmax": 1000,
  "downsample": 1000,
  "algo": "lttb"
}
```

#### Generate Empirical Map

```
POST /api/empirical_map
Content-Type: application/json

{
  "filename": "example.mdf",
  "template": "bsfc",
  "x_signal": "EngineRPM",
  "y_signal": "EngineLoad",
  "method": "kriging"
}
```

#### DiagAI Query

```
POST /api/diagai/query
Content-Type: application/json

{
  "query": "What's the average RPM during acceleration?",
  "session_id": "session_123",
  "filename": "example.mdf"
}

Response: {
  "response": "The average RPM during acceleration is 2,450 RPM...",
  "signals_used": ["EngineRPM", "VehicleSpeed"],
  "analysis_type": "statistical"
}
```

For complete API documentation, see DASHBOARD_USER_GUIDE.md.

---

## 🧪 Testing

```bash
# Run all tests
python3 -m pytest test_*.py

# Test specific modules
python3 test_misfire_full.py
python3 test_map_comprehensive.py
python3 test_gear_plots.py
python3 test_diagai_20_questions.py
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Contribution Guidelines

* Follow PEP 8 style guide
* Add tests for new features
* Update documentation
* Ensure all tests pass

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📋 Changelog

See CHANGELOG.md for a detailed list of changes and version history.

---

## 🙏 Acknowledgments

* **DBC Files**: Thanks to comma.ai OpenDBC, cantools
* **Libraries**: Flask, Plotly, asammdf, pandas, numpy, scipy
* **LLM Providers**: DeepSeek, LM Studio, Ollama
* **Inspiration**: MATLAB-based calibration tools

---

## 📞 Contact & Support

### 🚀 Enterprise Solutions & SaaS Integration

**VEHICLE-LAB** is available for enterprise deployment, SaaS integration, and custom development. We offer professional services for automotive OEMs, fleet management companies, and calibration specialists.

#### 💼 Enterprise Services

**Private LLM Deployment with OEM-Specific RAGs**
* On-premise or cloud deployment with complete data isolation
* Custom knowledge bases tailored to your brand and proprietary data
* Secure multi-tenancy architecture for multiple OEMs
* Enterprise-grade security and compliance

**SaaS Integration & White-Label Solutions**
* White-label deployment with custom branding
* API integration for existing platforms
* Custom feature development
* Scalable cloud infrastructure

**Professional Services**
* Implementation and deployment assistance
* Custom diagnostic agent development
* Training and onboarding for engineering teams
* Ongoing support and maintenance

#### 🤝 Contributing

We welcome contributions from the automotive engineering community! Whether you're improving algorithms, adding new diagnostic capabilities, or enhancing documentation, your contributions help make VEHICLE-LAB better for everyone.

**How to Contribute:**
1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

**Contribution Guidelines:**
* Follow PEP 8 style guide
* Add tests for new features
* Update documentation
* Ensure all tests pass

#### 📧 Business Inquiries

**For Enterprise Solutions, SaaS Integration, or Custom Development:**

* **GitHub Issues**: Create an issue with the `[Business Inquiry]` tag
* **GitHub Discussions**: Use the "Business Inquiries" category
* **Repository**: https://github.com/25Bonged/VEHICLE-LAB

**We're here to help you:**
* Deploy VEHICLE-LAB in your organization
* Integrate DiagAI with your existing systems
* Develop custom diagnostic capabilities
* Scale your vehicle data analysis operations

**Let's discuss how VEHICLE-LAB can transform your vehicle diagnostics workflow.**

---

**Made with ❤️ for the automotive engineering community**

⭐ **Star this repo if you find it useful!**

