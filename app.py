#!/usr/bin/env python3
"""
app.py – MDF Analytics Backend (Enhanced + Playground Unified)

Version: 1.0.36-robust-alias   (prev: 1.0.35-fallback-reader)

Key Capability Areas:
  - File management & secure upload (append / replace)
  - Channel discovery (union / intersection)
  - Multi-signal extraction (/analytics) with:
        * Legacy form-data mode (stats / FFT / hist / normalization)
        * New JSON "Playground" mode {fn:'series', ids:[], tmin, tmax, downsample, algo:lttb|stride}
  - High-fidelity downsampling (LTTB) for interactive plotting
  - Single-signal histogram endpoint (/api/histogram)
  - DFC analysis with evidence, code frequency, per-code plots, optional demo
  - IUPR analysis (optional module)
  - CC/SL overshoot detection (per-file + aggregated cruise vs limiter)
  - Integrated Gear Hunting analysis with event detection and plots
  - **Integrated Advanced Misfire Detection with multiple algorithms (crankshaft variance, FFT, statistical anomaly, angular velocity)**
  - **Integrated Empirical Map generation via custom_map.py module**
  - Report endpoints (/api/report_section, /api/report)
  - Quick DFC_ST plot endpoint (/api/dfc_analysis)
  - Debug endpoints for CC/SL, DFC, and signal map
  - Hardened JSON error handling & security headers

Additions in 1.0.36-robust-alias:
  - Feat: Added `_aliases_for_channel` helper, a powerful new function that generates a comprehensive set of alias keys for any channel name, handling normalization, case, URL encoding, compact forms, and array notations.
  - Patch: The `ch_map` construction in `extract_series` now uses this new helper to register all possible aliases for every canonical channel, ensuring that signal requests from the frontend are reliably mapped to the correct data source.
  - Patch: The legacy analytics endpoint's name resolution logic has been completely replaced with a robust system that uses the same aliasing logic and adds `difflib`-based fuzzy matching as a fallback, drastically reducing lookup failures.
  - Patch: `build_signal_lookup` is now aligned with the new aliasing strategy, providing consistent behavior.
  - Feat: Added more detailed debug logging to trace signal resolution failures.
"""

from __future__ import annotations
import os, re, time, shutil, traceback, json, math, glob, uuid, subprocess, sys, decimal, datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from functools import lru_cache
from contextlib import contextmanager
from pathlib import Path
from flask import Flask, request, jsonify
import urllib.parse

# ---------- CIE integration (robust import + aliases) ----------
try:
    import cie as cie_module
except Exception as _e:
    cie_module = None
    import logging
    logging.getLogger(__name__).warning("Failed to import cie.py: %s", _e)

if cie_module is not None:
    # --- Direct class imports from the CIE module ---
    ModelManager = getattr(cie_module, "ModelManager", None)
    OptimizationEngine = getattr(cie_module, "OptimizationEngine", None)
    MapGenerator = getattr(cie_module, "MapGenerator", None)
    InterpolationEngine = getattr(cie_module, "InterpolationEngine", None)
    RecommendationEngine = getattr(cie_module, "RecommendationEngine", None)
    DOEEngine = getattr(cie_module, "DOEEngine", None)
    
    # --- Helper function imports ---
    prepare_training_data = getattr(cie_module, "prepare_training_data", None)
    extract_series_for_training = getattr(cie_module, "extract_series_for_training", None)
    build_and_recommend_map = getattr(cie_module, "build_and_recommend_map", None)
    standardize_response = getattr(cie_module, "standardize_response", None)
else:
    # Define _missing and set all imports to it if cie_module is not available
    def _missing(*a, **k):
        raise RuntimeError("CIE module not available. Check cie.py and optional dependencies (scikit-learn, scipy, skopt).")
    ModelManager = OptimizationEngine = MapGenerator = InterpolationEngine = RecommendationEngine = DOEEngine = _missing
    prepare_training_data = extract_series_for_training = build_and_recommend_map = standardize_response = _missing


from pathlib import Path
Path("models").mkdir(exist_ok=True)

from flask import Flask, request, jsonify, send_from_directory, Response, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException

# Custom DFC analyzer
from custom_dfc import compute_dfc  # quick_dfc_st imported lazily

try:
    from custom_iupr import compute_iupr_plotly
except Exception:
    compute_iupr_plotly = None
try:
    from custom_cc_sl import analyze_cc_sl_behavior_mdf
except Exception:
    analyze_cc_sl_behavior_mdf = None
try:
    from custom_misfire import compute_misfire_plotly
except Exception:
    compute_misfire_plotly = None
try:
    from custom_fuel import compute_fuel_plotly
except Exception:
    compute_fuel_plotly = None

# try import external custom map module (prefer to use custom_map.py)
try:
    import custom_map as custom_map_module
    compute_map_plotly = getattr(custom_map_module, "compute_map_plotly", None)
    compute_map = getattr(custom_map_module, "compute_map", None)
except Exception:
    custom_map_module = None
    compute_map_plotly = None
    compute_map = None

APP_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(APP_DIR / "static"))

try:
    import asammdf
    from asammdf import MDF
    app.logger.info("asammdf import OK, version=%s", getattr(asammdf, "__version__", "unknown"))
except Exception as e:
    MDF = None
    app.logger.exception("asammdf import failed: %s", e)

try:
    import mdfreader
    _HAVE_MDFREADER = True
    app.logger.info("mdfreader import OK, version=%s", getattr(mdfreader, "__version__", "unknown"))
except Exception as e:
    _HAVE_MDFREADER = False
    app.logger.exception("mdfreader import failed: %s", e)

# Optional libs
try:
    import pandas as pd
except Exception:
    pd = None
try:
    import numpy as np
except Exception:
    np = None
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    go = None
    make_subplots = None
try:
    from scipy.interpolate import griddata
except ImportError:
    griddata = None


APP_DIR    = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
TMP_PLOTS  = APP_DIR / "tmp_plots"
MAPS_DIR   = APP_DIR / "maps_outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
TMP_PLOTS.mkdir(exist_ok=True)
MAPS_DIR.mkdir(parents=True, exist_ok=True)


MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))
MAX_FILES          = int(os.getenv("MAX_FILES", "400"))
UPLOAD_CHUNK_SIZE  = 1024 * 1024
DEBUG_MODE = os.getenv("BACKEND_DEBUG","0").lower() in ("1","true","yes","on")

# CC/SL thresholds
ENV_OVERSHOOT_MARGIN = float(os.getenv("OVERSHOOT_MARGIN_KMH", "2.0") or 2.0)
ENV_OVERSHOOT_FLAG_VALUE = int(os.getenv("OVERSHOOT_FLAG_VALUE", "5") or 5)
ENV_OVERSHOOT_FALLBACK_NO_FLAGS = os.getenv("OVERSHOOT_FALLBACK_NO_FLAGS","0").lower() in ("1","true","yes","on")

# DFC per-code plot limit
DFC_CODE_PLOT_MAX = int(os.getenv("DFC_CODE_PLOT_MAX","25") or 25)

CORS(app)

# --- CIE: lazy init singletons & model store ---
Path("models").mkdir(exist_ok=True)

_MODEL_MGR = None
_OPTIMIZER = None
_MAPS = None
_INTERPOLATION_ENGINE = None
_RECOMMENDATION_ENGINE = None
_DOE_ENGINE = None

def _ensure_cie_singletons():
    """Lazy initializer for CIE managers."""
    global _MODEL_MGR, _OPTIMIZER, _MAPS, _INTERPOLATION_ENGINE, _RECOMMENDATION_ENGINE, _DOE_ENGINE
    if _MODEL_MGR is not None:
        return
    if cie_module is None or ModelManager is None:
        raise RuntimeError("CIE module or its ModelManager class is not available.")
    
    # Instantiate classes directly
    _MODEL_MGR = ModelManager("models")
    if OptimizationEngine:
        _OPTIMIZER = OptimizationEngine("models")
    if MapGenerator:
        _MAPS = MapGenerator("models")
    if InterpolationEngine:
        _INTERPOLATION_ENGINE = InterpolationEngine()
    if RecommendationEngine:
        _RECOMMENDATION_ENGINE = RecommendationEngine()
    if DOEEngine:
        _DOE_ENGINE = DOEEngine()


ACTIVE_FILES: List[str] = []
CHANNELS_CACHE: Dict[str, List[str]] = {}
SERIES_CACHE: Dict[Tuple[Tuple[str,...], Tuple[str,...]], Dict] = {}


# ============================================================
# Advanced Frontend Integration Endpoints
# ============================================================

@app.post("/api/validate_files")
def api_validate_files():
    """Validate uploaded files before processing with detailed quality assessment"""
    try:
        data = request.get_json(force=True)
        file_paths = data.get("files", [])
        
        if not file_paths:
            return json_error("No files provided for validation", 400)
        
        validation_results = {
            'files': {},
            'overall_score': 0,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        for file_path in file_paths:
            file_validation = _validate_single_file(file_path)
            validation_results['files'][file_path] = file_validation
            
            if file_validation['status'] == 'error':
                validation_results['errors'].append(file_validation['message'])
            elif file_validation['status'] == 'warning':
                validation_results['warnings'].append(file_validation['message'])
        
        validation_results['overall_score'] = _calculate_overall_score(validation_results)
        return safe_jsonify({"ok": True, "validation": validation_results})
        
    except Exception as e:
        return json_error(f"File validation failed: {str(e)}", 500, exc=e)

def _validate_single_file(file_path: str) -> dict:
    """Validate individual file with comprehensive checks"""
    try:
        path = Path(file_path)
        if not path.exists():
            return {
                'name': path.name,
                'status': 'error',
                'message': 'File not found',
                'size_mb': 0,
                'format': path.suffix.lower(),
                'issues': ['File does not exist']
            }
        
        file_info = {
            'name': path.name,
            'size_mb': path.stat().st_size / (1024 * 1024),
            'format': path.suffix.lower(),
            'status': 'pending',
            'issues': []
        }
        
        # Check file size
        if file_info['size_mb'] > 500:
            file_info['issues'].append('File size exceeds 500MB limit')
            file_info['status'] = 'warning'
        
        # Check format compatibility
        if file_info['format'] not in ['.mdf', '.mf4', '.csv', '.xlsx', '.xls']:
            file_info['issues'].append(f'Unsupported format: {file_info["format"]}')
            file_info['status'] = 'error'
        
        # Quick content check for CSV
        if file_info['format'] == '.csv' and pd is not None:
            try:
                sample_data = pd.read_csv(path, nrows=5, encoding='utf-8', low_memory=False)
                if len(sample_data.columns) < 3:
                    file_info['issues'].append('File has too few columns')
                    file_info['status'] = 'warning'
            except Exception as e:
                file_info['issues'].append(f'CSV read error: {str(e)}')
                file_info['status'] = 'warning'
        
        # Quick content check for Excel files
        if file_info['format'] in ['.xlsx', '.xls'] and pd is not None:
            try:
                sample_data = pd.read_excel(path, nrows=5, engine='openpyxl' if file_info['format'] == '.xlsx' else None)
                if len(sample_data.columns) < 3:
                    file_info['issues'].append('File has too few columns')
                    file_info['status'] = 'warning'
            except Exception as e:
                file_info['issues'].append(f'Excel read error: {str(e)}')
                file_info['status'] = 'warning'
        
        # MDF/MF4 specific checks
        if file_info['format'] in ['.mdf', '.mf4']:
            try:
                if MDF is not None:
                    with MDF(path) as mdf:
                        channels = list(mdf.channels_db.keys())
                        if len(channels) < 5:
                            file_info['issues'].append('MDF file has very few channels')
                            file_info['status'] = 'warning'
            except Exception as e:
                file_info['issues'].append(f'MDF read error: {str(e)}')
                file_info['status'] = 'warning'
        
        if file_info['status'] == 'pending':
            file_info['status'] = 'valid'
            
        return file_info
        
    except Exception as e:
        return {
            'name': Path(file_path).name,
            'status': 'error',
            'message': f'Validation error: {str(e)}',
            'size_mb': 0,
            'format': Path(file_path).suffix.lower(),
            'issues': [f'Validation failed: {str(e)}']
        }

def _calculate_overall_score(validation_results: dict) -> float:
    """Calculate overall validation score (0-100)"""
    if not validation_results['files']:
        return 0.0
    
    total_files = len(validation_results['files'])
    valid_files = sum(1 for f in validation_results['files'].values() if f['status'] == 'valid')
    warning_files = sum(1 for f in validation_results['files'].values() if f['status'] == 'warning')
    
    # Score calculation: valid=100, warning=70, error=0
    score = (valid_files * 100 + warning_files * 70) / total_files
    return round(score, 1)

@app.post("/api/signal_suggestions")
def api_signal_suggestions():
    """Get intelligent signal mapping suggestions using UltimateSignalMapper"""
    try:
        data = request.get_json(force=True)
        column_names = data.get("columns", [])
        target_categories = data.get("categories", None)
        
        if not column_names:
            return json_error("No column names provided", 400)
        
        # Use the UltimateSignalMapper from custom_map.py
        if custom_map_module is None:
            return json_error("Signal mapping module not available", 503)
        
        # Initialize the signal mapper
        from custom_map import UltimateSignalMapper, UltimateEngineeringConstants
        constants = UltimateEngineeringConstants()
        signal_mapper = UltimateSignalMapper(constants)
        
        # Get intelligent mapping suggestions
        mapping_results = signal_mapper.intelligent_signal_mapping(column_names, target_categories)
        
        # Format results for frontend
        suggestions = {}
        for column, result in mapping_results.items():
            if result['signal']:
                suggestions[column] = {
                    'suggested_signal': result['signal'],
                    'confidence': result['confidence'],
                    'category': result['category'],
                    'units': result['units'],
                    'method': result['method'],
                    'details': result.get('details', {})
                }
        
        return safe_jsonify({
            "ok": True, 
            "suggestions": suggestions,
            "unmapped_columns": [col for col in column_names if col not in suggestions]
        })
        
    except Exception as e:
        return json_error(f"Signal suggestions failed: {str(e)}", 500, exc=e)

@app.post("/api/map_generation_progress")
def api_map_generation_progress():
    """Get progress status for map generation (placeholder for future async implementation)"""
    try:
        data = request.get_json(force=True)
        job_id = data.get("job_id", "default")
        
        # For now, return a simple status since we're using synchronous processing
        # In the future, this could check a background job queue
        return safe_jsonify({
            "ok": True,
            "job_id": job_id,
            "status": "completed",  # or "processing", "failed"
            "progress": 100,
            "stage": "map_generation_complete",
            "message": "Map generation completed successfully"
        })
        
    except Exception as e:
        return json_error(f"Progress check failed: {str(e)}", 500, exc=e)

@app.get("/api/test_map_module")
def api_test_map_module():
    """Test if the map module is working correctly"""
    try:
        if compute_map_plotly is None:
            return json_error("Map module not available", 503)
        
        # Test with a simple dummy call
        test_result = {
            "module_available": True,
            "custom_map_module": custom_map_module is not None,
            "compute_map_plotly": compute_map_plotly is not None,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return safe_jsonify({"ok": True, "test_result": test_result})
        
    except Exception as e:
        return json_error(f"Module test failed: {str(e)}", 500, exc=e)

# ============================================================
# Calibration Intelligence Engine (CIE) Endpoints
# ============================================================

# --- PATCH B: Hardened api_train_model with numeric coercion and clear errors ---
@app.post("/api/train_model")
def api_train_model():
    """
    Train a calibration surrogate model (features -> targets).
    Defensive behaviour:
      - parse repeated form fields for 'features' and 'targets'
      - coerce extracted data to numeric using pandas (if available)
      - return 400 with lists of offending columns if non-numeric
      - preserve existing model manager training if data valid
    """
    try:
        _ensure_cie_singletons()
        model_id = request.form.get("model_id", "default_model")
        # features and targets may be repeated fields: features=ch1&features=ch2
        feature_signals = request.form.getlist("features") or (request.form.get("features") or "").split(",")
        target_signals  = request.form.getlist("targets")  or (request.form.get("targets")  or "").split(",")
        model_type = request.form.get("model_type", "random_forest")

        # Clean-up: remove empty strings
        feature_signals = [f.strip() for f in feature_signals if isinstance(f, str) and f.strip()]
        target_signals  = [t.strip() for t in target_signals  if isinstance(t, str) and t.strip()]

        if not feature_signals:
            return jsonify({"ok": False, "error": "no_features", "message": "No feature signals provided"}), 400
        if not target_signals:
            return jsonify({"ok": False, "error": "no_targets", "message": "No target signals provided"}), 400

        # Determine files to use for training
        files = [Path(p) for p in (ACTIVE_FILES if ACTIVE_FILES else []) if Path(p).exists()]
        if not files:
            upload_dir = Path(UPLOAD_DIR) if 'UPLOAD_DIR' in globals() else Path('./uploads')
            if upload_dir.exists():
                files = [p for p in sorted(upload_dir.iterdir()) if p.is_file() and p.suffix.lower() in {".mf4", ".mdf", ".csv", ".xlsx", ".xls"}]
        if not files:
            return jsonify({"ok": False, "error": "no_training_files", "message": "No files available for training"}), 400

        # Use the CIE helper to extract aligned X,y — adapt if your helper name differs
        if 'extract_series_for_training' in globals() and callable(extract_series_for_training):
            X, y = extract_series_for_training(files, feature_signals, target_signals, extract_function=extract_series)
        else:
            # fallback to cie.prepare_training_data if direct extraction is implemented elsewhere
            if 'prepare_training_data' in globals() and callable(prepare_training_data):
                features_df = extract_series(files, feature_signals)
                targets_df  = extract_series(files, target_signals)
                X, y = prepare_training_data(features_df, targets_df, dropna=True)
            else:
                return jsonify({"ok": False, "error": "no_extraction_fn", "message": "No series extraction function available on server"}), 500

        if X is None or y is None:
            return jsonify({"ok": False, "error": "extraction_empty", "message": "Extraction returned empty X or y"}), 400

        # Coerce to numeric using pandas if available; collect offending columns
        non_numeric_features = []
        non_numeric_targets = []
        try:
            import pandas as _pd
            # Ensure X is a DataFrame
            if not isinstance(X, _pd.DataFrame):
                X = _pd.DataFrame(X)
            if not isinstance(y, _pd.DataFrame) and not isinstance(y, _pd.Series):
                y = _pd.DataFrame(y)

            # Coerce feature columns
            for col in list(X.columns):
                X[col] = _pd.to_numeric(X[col], errors='coerce')
                if X[col].isnull().all():
                    non_numeric_features.append(col)
            X = X.dropna(how='any')
            # Coerce targets
            if isinstance(y, _pd.DataFrame):
                for col in list(y.columns):
                    y[col] = _pd.to_numeric(y[col], errors='coerce')
                    if y[col].isnull().all():
                        non_numeric_targets.append(col)
                y = y.dropna(how='any')
            else:
                # y is Series
                y = _pd.to_numeric(y, errors='coerce')
                if y.isnull().all():
                    non_numeric_targets.append('target_series')
                y = y.dropna()
        except Exception:
            # pandas may not be available; let model mgr handle type errors and return a 500 if needed
            pass

        if non_numeric_features or non_numeric_targets:
            return jsonify({
                "ok": False,
                "error": "non_numeric_columns",
                "message": "Some extracted columns are non-numeric after coercion",
                "non_numeric_features": non_numeric_features,
                "non_numeric_targets": non_numeric_targets
            }), 400

        # Final emptiness check
        try:
            if hasattr(X, "empty") and X.empty:
                return jsonify({"ok": False, "error": "empty_after_coercion", "message": "No usable numeric feature rows remain after coercion"}), 400
            if hasattr(y, "empty") and y.empty:
                return jsonify({"ok": False, "error": "empty_after_coercion", "message": "No usable numeric target rows remain after coercion"}), 400
            if len(X) == 0 or len(y) == 0:
                return jsonify({"ok": False, "error": "empty_after_coercion", "message": "No usable numeric rows remain after coercion"}), 400
        except Exception:
            pass

        # Call the model manager. Uses _MODEL_MGR as implemented elsewhere in app.py
        if not _MODEL_MGR:
            # lazy-init for safety
            _ensure_cie_singletons()
            if not _MODEL_MGR:
                return jsonify({"ok": False, "error": "model_manager_missing", "message": "Model manager not available"}), 500

        # Use the model manager's API (train_model) — adapt if your manager uses a different name
        try:
            metrics = _MODEL_MGR.train_model(model_id, X, y, model_type)
            return safe_jsonify({
                "ok": True,
                "success": True,
                "model_id": model_id,
                "metrics": metrics,
                "feature_importance": metrics.get("feature_importances", {})
            })
        except Exception as train_exc:
            tb = traceback.format_exc()
            return jsonify({"ok": False, "error": "training_failed", "message": str(train_exc), "trace": tb}), 500

    except RuntimeError as e:
        return json_error(f"CIE Error: {e}", 503)
    except Exception as e:
        return json_error(f"Training failed: {str(e)}", 500, exc=e)


@app.post("/api/optimize")
def api_optimize():
    """Optimize feature values to hit target outputs."""
    try:
        _ensure_cie_singletons()
        data = request.get_json(force=True)
        model_id = data.get("model_id", "default_model")
        targets = data.get("targets", {})
        constraints = data.get("constraints", {})

        if not targets:
            return json_error("No targets provided", 400)

        result = _OPTIMIZER.optimize(model_id, targets, constraints)
        return safe_jsonify({"ok": True, **result})

    except RuntimeError as e:
        return json_error(f"CIE Error: {e}", 503)
    except Exception as e:
        return json_error(f"Optimization error: {str(e)}", 500, exc=e)


@app.post("/api/predict")
def api_predict():
    """Predict target outputs from given feature inputs."""
    try:
        _ensure_cie_singletons()
        data = request.get_json(force=True)
        model_id = data.get("model_id", "default_model")
        inputs = data.get("inputs", {})

        model = _MODEL_MGR.get_model(model_id)
        missing = [f for f in model.features if f not in inputs]
        if missing:
            return json_error(f"Missing features: {missing}", 400)

        df = pd.DataFrame({k: [v] for k, v in inputs.items()})
        preds = model.predict(df).iloc[0].to_dict()

        return safe_jsonify({"ok": True, "success": True, "predicted_outputs": preds, "model_info": model.get_info()})

    except RuntimeError as e:
        return json_error(f"CIE Error: {e}", 503)
    except Exception as e:
        return json_error(f"Prediction failed: {str(e)}", 500, exc=e)


@app.post("/api/train_map_model")
def api_train_map_model():
    """Train a surrogate model specifically for calibration map generation."""
    return api_train_model()


@app.post("/api/generate_map")
def api_generate_map():
    """Generate a high-resolution calibration map from a trained model."""
    try:
        _ensure_cie_singletons()
        data = request.get_json(force=True)
        model_id = data.get("model_id", "default_model")
        grid_definitions = data.get("grid_definitions", {})

        if not grid_definitions:
            return json_error("No grid definitions provided", 400)

        map_data = _MAPS.generate_map(model_id, grid_definitions)
        return safe_jsonify({"ok": True, "success": True, "map_data": map_data})

    except RuntimeError as e:
        return json_error(f"CIE Error: {e}", 503)
    except Exception as e:
        return json_error(f"Map generation failed: {str(e)}", 500, exc=e)


@app.get("/api/export_map/<model_id>")
def api_export_map(model_id):
    """Export a generated calibration map (CSV, JSON, INCA-like)."""
    try:
        _ensure_cie_singletons()
        fmt = request.args.get("format", "csv")
        dummy_grid = {"EngineSpeed": list(range(800, 5000, 500)), "Load": [0.2, 0.4, 0.6, 0.8, 1.0]}
        map_data = _MAPS.generate_map(model_id, dummy_grid)
        exported = _MAPS.export_map(map_data, fmt)
        return Response(exported, mimetype="text/plain")
    except RuntimeError as e:
        return json_error(f"CIE Error: {e}", 503)
    except Exception as e:
        return json_error(f"Export failed: {str(e)}", 500, exc=e)


@app.get("/api/list_models")
def api_list_models():
    """List all available trained models."""
    try:
        _ensure_cie_singletons()
        models = _MODEL_MGR.list_models()
        return safe_jsonify({"ok": True, "success": True, "models": models})
    except RuntimeError as e:
        return json_error(f"CIE Error: {e}", 503)
    except Exception as e:
        return json_error(f"List models failed: {str(e)}", 500, exc=e)


@app.post("/api/delete_model/<model_id>")
def api_delete_model(model_id):
    """Delete a trained model by ID."""
    try:
        _ensure_cie_singletons()
        success = _MODEL_MGR.delete_model(model_id)
        return safe_jsonify({"ok": success, "success": success, "deleted": model_id})
    except RuntimeError as e:
        return json_error(f"CIE Error: {e}", 503)
    except Exception as e:
        return json_error(f"Delete failed: {str(e)}", 500, exc=e)


@app.post("/api/bayesian_optimize")
def api_bayesian_optimize():
    try:
        _ensure_cie_singletons()
        data = request.get_json(force=True)
        model_id = data.get("model_id", "default_model")
        targets = data.get("targets", {})
        n_calls = int(data.get("n_calls", 50))
        res = _OPTIMIZER.bayesian_optimize(model_id, targets, n_calls=n_calls)
        return safe_jsonify({"ok": True, **res})
    except RuntimeError as e:
        return json_error(f"CIE Error: {e}", 503)
    except Exception as e:
        return json_error(f"Bayesian optimization failed: {e}", 500, exc=e)

@app.post("/api/predict_with_uncertainty")
def api_predict_with_uncertainty():
    try:
        _ensure_cie_singletons()
        data = request.get_json(force=True)
        model_id = data.get("model_id", "default_model")
        inputs = data.get("inputs", {})
        model = _MODEL_MGR.get_model(model_id)
        df = pd.DataFrame({k: [v] for k, v in inputs.items()})
        preds, uncertainties = model.predict_with_uncertainty(df)
        return safe_jsonify({"ok": True, "success": True, "predictions": preds.iloc[0].to_dict(), "uncertainties": uncertainties.iloc[0].to_dict()})
    except RuntimeError as e:
        return json_error(f"CIE Error: {e}", 503)
    except Exception as e:
        return json_error(f"Uncertainty prediction failed: {e}", 500, exc=e)

@app.post("/api/build_interpolation_map")
def api_build_interpolation_map():
    try:
        _ensure_cie_singletons()
        if not _INTERPOLATION_ENGINE:
            raise RuntimeError("InterpolationEngine not available")
        data = request.get_json(force=True)
        axes = data.get("axes", [])
        target = data.get("target")
        method = data.get("method", "auto")
        grid_pts = int(data.get("grid_points_per_axis", 30))
        if "measurements" in data:
            df = pd.DataFrame(data["measurements"])
        else:
            files = [Path(p) for p in ACTIVE_FILES if Path(p).exists()]
            series_map = extract_series(files, axes + [target], include_time=False, normalize=False)
            if not series_map:
                return json_error("No measurements available", 400)
            min_len = min(len(s["values"]) for s in series_map.values())
            rows = [{k: series_map[k]["values"][i] for k in series_map} for i in range(min_len)]
            df = pd.DataFrame(rows)

        map_res = _INTERPOLATION_ENGINE.build_map(df=df, axes=axes, target=target, method=method, resolution=grid_pts)
        return safe_jsonify({"ok": True, "success": True, "map_data": map_res})
    except RuntimeError as e:
        return json_error(f"CIE Error: {e}", 503)
    except Exception as e:
        return json_error(f"Interpolation map failed: {e}", 500, exc=e)

@app.post("/api/get_recommendations")
def api_get_recommendations():
    try:
        _ensure_cie_singletons()
        if not _RECOMMENDATION_ENGINE or not _INTERPOLATION_ENGINE:
            raise RuntimeError("Recommendation or Interpolation Engine not available")
        data = request.get_json(force=True)
        axes = data.get("axes", [])
        target = data.get("target")
        clip_pct = float(data.get("clip_pct", 0.15))
        topk = int(data.get("topk_points", 10))

        if "measurements" in data:
            df = pd.DataFrame(data["measurements"])
        else:
            files = [Path(p) for p in ACTIVE_FILES if Path(p).exists()]
            series_map = extract_series(files, axes + [target], include_time=False, normalize=False)
            if not series_map:
                return json_error("No measurements available", 400)
            min_len = min(len(s["values"]) for s in series_map.values())
            rows = [{k: series_map[k]["values"][i] for k in series_map} for i in range(min_len)]
            df = pd.DataFrame(rows)

        if "map" in data:
            map_data = data["map"]
        else:
            map_data = _INTERPOLATION_ENGINE.build_map(df=df, axes=axes, target=target)

        rec = _RECOMMENDATION_ENGINE.recommend(df, map_data, axes, target, clip_pct=clip_pct, topk_points=topk)
        return safe_jsonify({"ok": True, "success": True, "recommendation": rec, "map": map_data})
    except RuntimeError as e:
        return json_error(f"CIE Error: {e}", 503)
    except Exception as e:
        return json_error(f"Recommendation failed: {e}", 500, exc=e)

@app.post("/api/generate_doe")
def api_generate_doe():
    try:
        _ensure_cie_singletons()
        if not _DOE_ENGINE:
            raise RuntimeError("DOEEngine not available")
        data = request.get_json(force=True)
        param_ranges = data.get("param_ranges") or data.get("params")
        n_samples = int(data.get("n_samples", 20))
        method = data.get("method", "lhs")
        if not param_ranges:
            return json_error("No parameter ranges provided", 400)
        df = _DOE_ENGINE.generate_doe(param_ranges, n_samples=n_samples, method=method)
        return safe_jsonify({"ok": True, "success": True, "doe_samples": df.to_dict(orient="records")})
    except RuntimeError as e:
        return json_error(f"CIE Error: {e}", 503)
    except Exception as e:
        return json_error(f"DOE generation failed: {e}", 500, exc=e)


# ------------------------------------------------------------------ Security Headers
@app.after_request
def secure_headers(resp):
    resp.headers['X-Content-Type-Options'] = 'nosniff'
    resp.headers['Referrer-Policy']       = 'same-origin'
    # Allow embedding within the same site
    resp.headers['X-Frame-Options']       = 'SAMEORIGIN'
    resp.headers['X-XSS-Protection']      = '0'
    resp.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.plot.ly 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "font-src 'self'; connect-src 'self'; "
        "object-src 'none'; base-uri 'self'; frame-ancestors 'self'"
    )
    return resp

def make_json_serializable(obj):
    """
    Recursively convert numpy / pandas scalar/arrays and other non-json types
    into JSON-serializable native Python types (int, float, bool, list, dict, str).
    Converts NaN, Inf, and -Inf to None for proper JSON serialization.
    """
    # handle None, basic python types
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    
    # Check for NaN, Inf, -Inf (both numpy and Python float)
    if isinstance(obj, float):
        if np and (np.isnan(obj) or np.isinf(obj)):
            return None
        import math
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    
    if isinstance(obj, int):
        return obj

    # numpy scalar types
    if np and isinstance(obj, (np.integer, )):
        return int(obj)
    if np and isinstance(obj, (np.floating, )):
        val = float(obj)
        # Convert NaN/Inf to None
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if np and isinstance(obj, (np.bool_, )):
        return bool(obj)
    if np and isinstance(obj, (np.str_, )):
        return str(obj)
    if np and isinstance(obj, (np.ndarray, )):
        return [make_json_serializable(x) for x in obj.tolist()]

    # pandas scalars/indices/series/dataframes
    if pd and isinstance(obj, (pd.Series, )):
        # Handle NaN in Series - convert to None
        result = []
        for x in obj.tolist():
            if pd.isna(x):
                result.append(None)
            else:
                result.append(make_json_serializable(x))
        return result
    if pd and isinstance(obj, (pd.Index, )):
        # Handle NaN in Index
        result = []
        for x in obj.tolist():
            if pd.isna(x):
                result.append(None)
            else:
                result.append(make_json_serializable(x))
        return result
    if pd and isinstance(obj, (pd.DataFrame, )):
        # convert to list of dicts (rows) - handle NaN values
        records = obj.to_dict(orient='records')
        # Replace NaN values with None in each record
        for record in records:
            for k, v in record.items():
                if pd.isna(v):
                    record[k] = None
                else:
                    record[k] = make_json_serializable(v)
        return records

    # decimal
    if isinstance(obj, decimal.Decimal):
        try:
            # prefer int if exact
            if obj == obj.to_integral_value():
                return int(obj)
        except Exception:
            pass
        return float(obj)

    # dict: convert each key/value
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            # ensure keys are strings
            if not isinstance(k, str):
                try:
                    k2 = str(k)
                except Exception:
                    k2 = repr(k)
            else:
                k2 = k
            new[k2] = make_json_serializable(v)
        return new

    # iterable (list/tuple/set) - handle NaN values explicitly
    if isinstance(obj, (list, tuple, set)):
        result = []
        for x in obj:
            # Check for NaN/Inf before recursive conversion
            if isinstance(x, float):
                if np and (np.isnan(x) or np.isinf(x)):
                    result.append(None)
                    continue
                import math
                if math.isnan(x) or math.isinf(x):
                    result.append(None)
                    continue
            # Check numpy scalars in iterables
            if np and isinstance(x, (np.floating, np.integer)):
                val = float(x) if isinstance(x, np.floating) else int(x)
                if np.isnan(val) or np.isinf(val):
                    result.append(None)
                    continue
            result.append(make_json_serializable(x))
        return result

    # Fallback: numeric-like objects with item() method (numpy scalars)
    try:
        if hasattr(obj, 'item'):
            val = obj.item()
            # item might be numpy scalar or python scalar
            return make_json_serializable(val)
    except Exception:
        pass

    # As a final fallback, convert to string (safe)
    try:
        return str(obj)
    except Exception:
        return repr(obj)

# ------------------------------------------------------------------ Helpers / JSON
def _bool(v, default=False):
    if v is None: return default
    return str(v).lower() in ("1","true","yes","on","y")

def _clean_name(name: str) -> str:
    """Sanitize signal name for use as a clean ID."""
    if not isinstance(name, str):
        name = str(name)
    # Basic: remove special chars, replace with underscore
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

def _extract_display_name(signal_name: str) -> str:
    """
    Extract clean display name from full signal path.
    
    Examples:
        '96D7124080_8128328U_FM77_nc_CAN_VITESSE_VEHICULE_ROUES' -> 'CAN_VITESSE_VEHICULE_ROUES'
        'MG1CS051_H440_2F_EngM_facTranCorSlop_RTE' -> 'EngM_facTranCorSlop_RTE'
        '96D7124080_8128328U_FM77_nc_SG_.PENTE_STATIQUE' -> 'PENTE_STATIQUE'
    
    Rules:
        1. If contains '.', take part after last dot (but skip if result is empty/short)
        2. Remove known OEM/module prefixes (alphanumeric identifiers)
        3. Find first segment that looks like actual signal (starts with letter, all caps, or meaningful)
    """
    if not signal_name:
        return signal_name
    
    name = str(signal_name).strip()
    
    # Rule 1: Check for dot separator (e.g., '96D7124080_8128328U_FM77_nc_SG_.PENTE_STATIQUE')
    if '.' in name:
        parts = name.split('.')
        # Take the last non-empty part
        for part in reversed(parts):
            part = part.strip()
            if part and len(part) > 2:  # Must be meaningful length
                return part
    
    # Rule 2: Remove known OEM/module prefixes
    # Patterns: '96D7124080_8128328U_FM77_nc_', 'MG1CS051_H440_2F_', etc.
    # Module prefixes are: alphanumeric sequences (often with numbers), short segments (nc, SG)
    # Actual signals start with: meaningful words (CAN, EngM, etc.)
    parts = name.split('_')
    
    # Look for the first segment that looks like an actual signal name
    # Signal indicators:
    # - All uppercase word (3+ chars) like CAN, VITESSE, PENTE
    # - Starts with uppercase letter followed by lowercase (like EngM, TqSys)
    # - Not part of module identifier pattern (alphanumeric codes)
    meaningful_start = -1
    for i, part in enumerate(parts):
        if not part:
            continue
        
        # Check if part looks like module identifier (skip these):
        # - Starts with number
        # - Short (1-2 chars) like 'nc', 'SG' (case-insensitive)
        # - Mixed case with numbers at start (like '96D7124080', 'MG1CS051')
        is_module_id = (
            part[0].isdigit() or  # Starts with number
            len(part) <= 2 or  # Short segments (1-2 chars) like 'nc', 'SG'
            (part.isalnum() and part[0].isupper() and any(c.isdigit() for c in part[:3]))  # Mixed alphanumeric at start
        )
        
        if not is_module_id:
            # This could be the signal start
            # Check: is it a meaningful signal name?
            if (part.isupper() and len(part) >= 3) or (part[0].isupper() and part[1:].islower()):
                # Looks like actual signal - check if previous parts were all module identifiers
                if i == 0 or all(
                    not p or 
                    p[0].isdigit() or 
                    len(p) <= 2 or  # Short segments
                    (p.isalnum() and p[0].isupper() and any(c.isdigit() for c in p[:3]))
                    for p in parts[:i]
                ):
                    meaningful_start = i
                    break
    
    if meaningful_start >= 0 and meaningful_start < len(parts):
        # Return from the meaningful part onwards
        result = '_'.join(parts[meaningful_start:])
        if result:
            return result
    
    # Rule 3: Fallback - try regex pattern matching for module prefixes
    # Pattern: One or more sequences of uppercase+digits separated by underscores, ending with underscore
    match = re.match(r'^([A-Z0-9]+_[A-Z0-9]+(_[A-Z0-9]+)*_)', name)
    if match:
        remaining = name[match.end():]
        if remaining and len(remaining) > 3:
            return remaining
    
    # Final fallback: return as-is if no pattern matched
    return name

# --- add this helper (paste near other helper functions) ---
import urllib.parse
import unicodedata
import difflib

def _normalize_text(s: str) -> str:
    """Normalize whitespace, unicode, collapse multiple spaces and strip."""
    if s is None: return ""
    # normalize unicode (NFKC), remove non-breaking spaces, collapse whitespace
    t = unicodedata.normalize("NFKC", str(s))
    t = t.replace("\u00A0", " ")
    t = " ".join(t.split())
    return t.strip()

def _aliases_for_channel(name: str):
    """
    Return a set of alias keys for a channel name:
      - original
      - normalized (_clean_name)
      - lowercase variants
      - URL-decoded/encoded variants
      - compact form (last token after '.' or after '#...') when applicable
      - array-base (strip [i] from end)
    This helps matching names that come from different tools / frontends.
    """
    out = set()
    if not name:
        return out
    # canonical forms
    name = str(name)
    out.add(name)
    clean = _clean_name(name)
    out.add(clean)
    out.add(name.lower())
    out.add(clean.lower())

    # normalized whitespace/unicode
    nname = _normalize_text(name)
    out.add(nname)
    out.add(nname.lower())
    nclean = _normalize_text(clean)
    out.add(nclean)
    out.add(nclean.lower())

    # URL-decode and percent-encode forms
    try:
        dec = urllib.parse.unquote(name)
        if dec != name:
            out.add(dec); out.add(dec.lower()); out.add(_clean_name(dec)); out.add(_clean_name(dec).lower())
    except Exception:
        pass

    # compact/short name heuristics:
    #  - if name contains '.', take last token (often short signal name + qualifier)
    #  - if name contains '/', '#', take substring after last delimiter
    delim_split = [".", "/", "#"]
    for d in delim_split:
        if d in name:
            compact = name.split(d)[-1]
            compact = _normalize_text(compact)
            out.add(compact); out.add(compact.lower()); out.add(_clean_name(compact)); out.add(_clean_name(compact).lower())

    # array index support: map "SignalName[0]" -> "SignalName"
    if "[" in name and "]" in name:
        base = name.split("[")[0]
        base = _normalize_text(base)
        out.add(base); out.add(base.lower()); out.add(_clean_name(base)); out.add(_clean_name(base).lower())

    # also add variant with periods replaced by spaces and vice versa
    out.add(name.replace(".", " "))
    out.add(name.replace(" ", "."))
    # finally strip stray punctuation at both ends
    out.add(name.strip(" '\""))
    return {o for o in out if o}


def _sanitize_filename(name: str) -> str:
    base = Path(name).name
    base = secure_filename(base) or "upload"
    base = re.sub(r'[<>:"/\\|?*]+', "_", base).rstrip(" .")
    if not Path(base).suffix:
        base += ".mf4"
    reserved = {"CON","PRN","AUX","NUL","COM1","COM2","COM3","COM4","COM5","COM6",
                "COM7","COM8","COM9","LPT1","LPT2","LPT3","LPT4","LPT5","LPT6",
                "LPT7","LPT8","LPT9"}
    if Path(base).stem.upper() in reserved:
        base = f"_{base}"
    return base

def list_mdf_files(folder: Path) -> List[Path]:
    return sorted(list(folder.glob("*.mf4")) + list(folder.glob("*.mdf")) + 
                  list(folder.glob("*.csv")) + list(folder.glob("*.xlsx")) + list(folder.glob("*.xls")))

def list_uploaded_files() -> List[str]:
    exts={".mf4",".mdf",".csv",".xlsx",".xls"}
    return [str(p.resolve()) for p in sorted(UPLOAD_DIR.iterdir()) if p.suffix.lower() in exts]

def purge_mdf_files_only():
    """Delete only MDF/MF4 files from UPLOAD_DIR (legacy function name, now handles all supported formats)."""
    if not UPLOAD_DIR.exists():
        return 0
    deleted = 0
    for p in UPLOAD_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in {".mdf", ".mf4", ".csv", ".xlsx", ".xls"}:
            try:
                p.unlink()
                deleted += 1
            except Exception as e:
                app.logger.warning("Failed to delete file %s: %s", p, e)
    return deleted

def purge_all():
    """Deletes all files and subdirectories from UPLOAD_DIR and TMP_PLOTS."""
    for d in [UPLOAD_DIR, TMP_PLOTS]:
        if d.exists():
            for p in d.iterdir():
                try:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
                except Exception as e:
                    app.logger.warning("Failed to purge path %s: %s", p, e)
    ACTIVE_FILES.clear()
    CHANNELS_CACHE.clear()
    SERIES_CACHE.clear()

# --- PATCH A: backwards-compatible alias route for older frontend ---
@app.get("/api/list_uploaded_files")
def api_list_uploaded_files():
    """
    Backwards-compatible alias used by older frontend versions.
    Returns same structure as /api/files: {"files": [{name, path, size}, ...], "count": N}
    """
    try:
        upload_dir = Path(UPLOAD_DIR) if 'UPLOAD_DIR' in globals() else Path('./uploads')
        files = []
        # If list_mdf_files helper exists, prefer it (keeps same filtering)
        if 'list_mdf_files' in globals() and callable(list_mdf_files):
            raw_files = list_mdf_files(upload_dir)
            for p in raw_files:
                try:
                    pp = Path(p)
                    files.append({"name": pp.name, "path": str(pp.resolve()), "size": pp.stat().st_size})
                except Exception:
                    files.append({"name": str(p), "path": str(p), "size": 0})
        else:
            # fallback: enumerate files in upload_dir
            if upload_dir.exists():
                for p in sorted(upload_dir.iterdir()):
                    if p.is_file() and p.suffix.lower() in {".mf4", ".mdf", ".csv", ".xlsx", ".xls"}:
                        try:
                            files.append({"name": p.name, "path": str(p.resolve()), "size": p.stat().st_size})
                        except Exception:
                            files.append({"name": p.name, "path": str(p), "size": 0})
        return safe_jsonify({"files": files, "count": len(files)})
    except Exception as e:
        try:
            return json_error("list_uploaded_files_failed", 500, exc=e)
        except Exception:
            return jsonify({"ok": False, "error": "list_uploaded_files_failed", "message": str(e)}), 500

def safe_jsonify(payload: Any, status: int = 200):
    """
    Return a Flask Response (JSON) with the requested HTTP status.
    Ensures payload is JSON serializable and sets response status properly.
    """
    try:
        resp = jsonify(make_json_serializable(payload))
        resp.status_code = int(status or 200)
        return resp
    except Exception:
        # Fallback: return minimal error
        resp = jsonify({"ok": False, "error": "jsonify_failed", "payload_repr": str(payload)})
        resp.status_code = int(status or 500)
        return resp

def json_error(message: str, status: int = 500, exc: Exception|None=None, extra: Dict|None=None):
    data={"ok":False,"error":message}
    if extra: data.update(extra)
    if exc and DEBUG_MODE:
        data["exception_type"]=exc.__class__.__name__
        data["trace_tail"]=traceback.format_exc().splitlines()[-12:]
    return safe_jsonify(data,status=status)

def _get_payload_param(key: str, default: str | None = None):
    if key in request.form:
        return request.form.get(key, default)
    if request.is_json:
        data = request.get_json(silent=True) or {}
        if key in data:
            return data.get(key, default)
    if key in request.args:
        return request.args.get(key, default)
    return default

def _safe_mkdir(p: Path):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def _load_primary_table_from_map_json(data):
    """
    Accepts parsed JSON from a map run and returns a normalized 'table' object,
    where 'table' is either:
      - a list of row dicts, or
      - a dict-of-columns {col: [..]}
    The function prefers:
      data['table'] -> data['tables']['Map Summary'] -> first table in data['tables'] -> samples -> data
    Returns (table, source_key_name)
    """
    if not isinstance(data, dict):
        return None, None

    if 'table' in data:
        return data['table'], 'table'

    # prefer the specific 'Map Summary' name inside 'tables'
    if 'tables' in data and isinstance(data['tables'], dict):
        tbls = data['tables']
        if 'Map Summary' in tbls:
            return tbls['Map Summary'], 'tables["Map Summary"]'
        # fallback to the first table present
        for k, v in tbls.items():
            return v, f'tables["{k}"]'

    # fallbacks
    if 'samples' in data:
        return data['samples'], 'samples'
    if 'data' in data:
        return data['data'], 'data'
    if 'df' in data:
        return data['df'], 'df'

    # No table-like object found
    return None, None

# ------------------------------------------------------------------ Error handler
@app.errorhandler(Exception)
def handle_any_exception(e: Exception):
    if isinstance(e, HTTPException):
        return json_error(e.description,e.code,exc=e)
    return json_error("internal_server_error",500,exc=e)

# ------------------------------------------------------------------
# MDF open helpers (Hardened with detailed logging & consistent wrapper)
from contextlib import contextmanager

@contextmanager
def open_mdf(path: Path):
    """
    Robust MDF open helper.
    Yields a wrapper object with:
      - channels() -> list[str]
      - get(ch)     -> (timestamps_list, samples_list, unit_str)
      - close()
    This will:
      - Try Excel/CSV first if file extension matches
      - Try asammdf first (and log the full exception if it fails)
      - Fall back to mdfreader (and log its exception)
      - Yield None if neither can open the file
    """
    # Check for Excel/CSV files first
    path_str = str(path)
    suffix_lower = Path(path).suffix.lower()
    if suffix_lower in {'.csv', '.xlsx', '.xls'}:
        if pd is None:
            app.logger.warning("pandas not available, cannot read Excel/CSV file: %s", path)
            yield None
            return
        
        try:
            # Load the file with pandas
            if suffix_lower == '.csv':
                # Try multiple encodings for CSV
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                df = None
                for enc in encodings:
                    try:
                        # Read first few rows to detect structure (without header first)
                        sample_no_header = pd.read_csv(path_str, encoding=enc, nrows=4, header=None, low_memory=False)
                        
                        # Detect pattern: headers in row 0, descriptions in row 1, units in row 2, data from row 3
                        skip_rows = 0
                        if len(sample_no_header) >= 3:
                            first_col = sample_no_header.iloc[:, 0]
                            
                            # Pattern: Row 0=headers, Row 1=descriptions (text), Row 2=units (short text), Row 3=data (numeric)
                            # Check if row 1 and 2 are text descriptions/units
                            row1_first = str(first_col.iloc[1] if len(first_col) > 1 else '').strip()
                            row2_first = str(first_col.iloc[2] if len(first_col) > 2 else '').strip()
                            row3_first = str(first_col.iloc[3] if len(first_col) > 3 else '').strip()
                            
                            # Try to convert row 3 to numeric to see if it's data
                            is_row3_numeric = False
                            try:
                                float(str(row3_first).replace(',', '').strip())
                                is_row3_numeric = True
                            except (ValueError, TypeError):
                                pass
                            
                            # If row 3 is numeric and rows 1-2 are text (likely descriptions/units), skip 2 rows
                            if (is_row3_numeric and 
                                len(row1_first) > 3 and  # Row 1 is descriptive text (not just a number)
                                len(row2_first) < 50):   # Row 2 is likely units (short)
                                skip_rows = 2  # Skip description and units rows
                            # Pattern: Row 0=headers, Row 1=descriptions, Row 2=data
                            elif (is_row3_numeric and len(row1_first) > 3):
                                # Check if row 2 could be units (very short)
                                if len(row2_first) < 20:
                                    skip_rows = 2
                                else:
                                    skip_rows = 1
                        
                        # Read full file - header=0 means use first row as headers
                        # If skip_rows=2, we want: row 0 as header, skip rows 1-2, start data at row 3
                        # skiprows with list: skip specific row indices (1-indexed when header is used)
                        if skip_rows > 0:
                            # When header=0, skiprows list should contain 1-based row indices to skip AFTER the header
                            # So if we want to skip rows 1 and 2 (0-indexed), we pass [1, 2] as 1-based indices
                            skip_list = list(range(1, skip_rows + 1))
                            df = pd.read_csv(path_str, encoding=enc, header=0, skiprows=skip_list, low_memory=False)
                        else:
                            df = pd.read_csv(path_str, encoding=enc, header=0, low_memory=False)
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                    except Exception as e:
                        # If skiprows detection fails, try without skipping
                        try:
                            df = pd.read_csv(path_str, encoding=enc, low_memory=False)
                            break
                        except Exception:
                            continue
                if df is None:
                    raise ValueError(f"Could not decode CSV file {path} with any supported encoding")
                
                # Clean up: try to convert columns to numeric where possible
                for col in df.columns:
                    try:
                        # Try converting to numeric, coercing errors
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception:
                        pass
            else:
                # Excel files
                try:
                    # Try openpyxl for .xlsx, xlrd for .xls (if available)
                    if suffix_lower == '.xlsx':
                        df = pd.read_excel(path_str, engine='openpyxl')
                    else:
                        # .xls files - try openpyxl first (newer versions support it), then xlrd
                        try:
                            df = pd.read_excel(path_str, engine='openpyxl')
                        except Exception:
                            # Fallback to default engine (usually xlrd if available)
                            df = pd.read_excel(path_str)
                except Exception as e:
                    raise ValueError(f"Could not read Excel file {path}: {e}")
            
            # Create a wrapper for DataFrame
            class DataFrameWrap:
                def __init__(self, dataframe, path_obj):
                    self._df = dataframe
                    self._path = path_obj
                    self._time_column = None
                    self._detect_time_column()
                
                def _detect_time_column(self):
                    """Detect time/index column automatically"""
                    # Common time column names
                    time_names = ['time', 'timestamp', 't', 'Time', 'Timestamp', 'T',
                                'datetime', 'DateTime', 'date', 'Date', 'index', 'Index']
                    
                    # Check if any column matches time names
                    for col in self._df.columns:
                        col_lower = str(col).lower().strip()
                        if col_lower in time_names:
                            self._time_column = col
                            return
                    
                    # If no explicit time column, try to detect numeric index-like column
                    # Use first numeric column as time if it looks sequential
                    if np is not None:
                        numeric_cols = self._df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            first_num = numeric_cols[0]
                            col_data = self._df[first_num].dropna()
                            if len(col_data) > 1:
                                # Check if it's roughly sequential (could be time/sample index)
                                diffs = col_data.diff().dropna()
                                if len(diffs) > 0:
                                    # If differences are mostly consistent, use as time
                                    cv = diffs.std() / (diffs.mean() + 1e-10)
                                    if cv < 0.5:  # Low coefficient of variation
                                        self._time_column = first_num
                                        return
                    
                    # Default: use index as time
                    self._time_column = "__index__"
                
                def channels(self):
                    """Return list of column names as channels"""
                    cols = list(self._df.columns)
                    # Don't include the time column in the channels list if it's a named column
                    if self._time_column and self._time_column != "__index__":
                        cols = [c for c in cols if c != self._time_column]
                    return cols
                
                def get(self, ch):
                    """Return (timestamps, values, unit) for a channel"""
                    if ch not in self._df.columns:
                        # Try case-insensitive match
                        ch_lower = ch.lower()
                        for col in self._df.columns:
                            if str(col).lower() == ch_lower:
                                ch = col
                                break
                        else:
                            return ([], [], "")
                    
                    # Get values
                    values = self._df[ch].dropna().tolist()
                    
                    # Get timestamps
                    if self._time_column == "__index__":
                        # Use index as time
                        timestamps = self._df.index.tolist()
                        if len(timestamps) != len(values):
                            # Align with values (handle NaN rows)
                            timestamps = self._df.loc[self._df[ch].notna()].index.tolist()
                    else:
                        # Use time column
                        time_values = self._df[self._time_column].tolist()
                        # Align timestamps with non-null values
                        mask = self._df[ch].notna()
                        timestamps = self._df.loc[mask, self._time_column].tolist()
                    
                    # Ensure timestamps are numeric
                    try:
                        timestamps = [float(t) for t in timestamps]
                    except (ValueError, TypeError):
                        # Try converting datetime to numeric (seconds since start)
                        try:
                            import pandas as pd
                            if isinstance(self._df[self._time_column].iloc[0] if len(self._df) > 0 else None, pd.Timestamp):
                                t0 = self._df[self._time_column].iloc[0]
                                timestamps = [(pd.Timestamp(t) - t0).total_seconds() for t in timestamps]
                            else:
                                # Fallback: use index
                                timestamps = list(range(len(values)))
                        except Exception:
                            timestamps = list(range(len(values)))
                    
                    # Ensure values are numeric
                    try:
                        values = [float(v) for v in values]
                    except (ValueError, TypeError):
                        # Keep original values if not all numeric
                        pass
                    
                    return (timestamps, values, "")
                
                def close(self):
                    """Cleanup - nothing needed for DataFrame"""
                    pass
            
            w = DataFrameWrap(df, path)
            try:
                yield w
            finally:
                try:
                    w.close()
                except Exception:
                    pass
            return
        except Exception as e:
            app.logger.exception("Failed to read Excel/CSV file %s: %s", path, e)
            yield None
            return
    
    class Wrap:
        def __init__(self, obj, flavor):
            self._o = obj
            self._f = flavor

        def channels(self):
            if self._f == "asam":
                try:
                    # modern asammdf exposes channels_db (dict) in many versions
                    if hasattr(self._o, "channels_db") and isinstance(self._o.channels_db, dict):
                        return list(self._o.channels_db.keys())
                    # fallback: some versions expose channels as iterable
                    try:
                        return [c for c in getattr(self._o, "channels", [])]
                    except Exception:
                        return []
                except Exception:
                    return []
            else:
                # mdfreader: attempt keys() then channels attr
                try:
                    if isinstance(self._o, dict):
                        return list(self._o.keys())
                except Exception:
                    pass
                try:
                    if hasattr(self._o, "channels"):
                        chs = getattr(self._o, "channels")
                        if isinstance(chs, dict):
                            return list(chs.keys())
                        try:
                            return [k for k in chs]
                        except Exception:
                            return []
                except Exception:
                    pass
                return []

        def get(self, ch):
            # Return (timestamps_list, samples_list, unit)
            if self._f == "asam":
                try:
                    s = self._o.get(ch)
                except Exception:
                    # sometimes asammdf uses different key names; try find matching name
                    try:
                        # attempt direct attribute access
                        s = getattr(self._o, ch)
                    except Exception:
                        return ([], [], "")
                if s is None:
                    return ([], [], "")
                unit_raw = getattr(s, "unit", "") or ""
                if isinstance(unit_raw, bytes):
                    try:
                        unit_raw = unit_raw.decode("utf-8", "ignore")
                    except Exception:
                        unit_raw = ""
                # Many asammdf Signal objects expose timestamps and samples as numpy arrays
                try:
                    t = getattr(s, "timestamps", None)
                    v = getattr(s, "samples", None)
                    ts = t.tolist() if hasattr(t, "tolist") else list(t or [])
                    vals = v.tolist() if hasattr(v, "tolist") else list(v or [])
                    return (ts, vals, unit_raw)
                except Exception:
                    # fallback: maybe s itself is a tuple-like
                    try:
                        return (list(s[0] or []), list(s[1] or []), unit_raw)
                    except Exception:
                        return ([], [], unit_raw)
            else:
                # mdfreader style channel object
                try:
                    # try get_channel or channels dict
                    if hasattr(self._o, "get_channel"):
                        c = self._o.get_channel(ch)
                    elif hasattr(self._o, "channels") and ch in getattr(self._o, "channels"):
                        c = getattr(self._o, "channels")[ch]
                    else:
                        # try attribute access fallback
                        c = getattr(self._o, ch, None)
                    if c is None:
                        return ([], [], "")
                    unit_raw = getattr(c, "unit", "") or ""
                    if isinstance(unit_raw, bytes):
                        try:
                            unit_raw = unit_raw.decode("utf-8", "ignore")
                        except Exception:
                            unit_raw = ""
                    t = getattr(c, "timestamps", None)
                    v = getattr(c, "samples", None)
                    ts = t.tolist() if hasattr(t, "tolist") else list(t or [])
                    vals = v.tolist() if hasattr(v, "tolist") else list(v or [])
                    return (ts, vals, unit_raw)
                except Exception:
                    # last fallback: try get_channel_data if available
                    try:
                        if hasattr(self._o, "get_channel_data"):
                            arr = self._o.get_channel_data(ch)
                            return ([], list(arr or []), "")
                    except Exception:
                        pass
                    return ([], [], "")

        def close(self):
            try:
                if hasattr(self._o, "close"):
                    self._o.close()
            except Exception:
                pass

    # Try asammdf first — log the exception if it fails
    if MDF is not None:
        try:
            m = MDF(str(path))
            w = Wrap(m, "asam")
            try:
                yield w
            finally:
                try: w.close()
                except Exception: pass
            return
        except Exception as e_as:
            # log traceback to app logger and also write a short diagnostics file so the frontend can request it
            try:
                app.logger.exception("asammdf open failed for %s: %s", path, e_as)
            except Exception:
                print(f"[ERROR] asammdf open failed for {path}: {e_as}")
            # store error trace to a diagnostics file
            try:
                diag_file = (UPLOAD_DIR / (Path(path).name + ".open_asam_error.txt"))
                with open(diag_file, "w", encoding="utf-8") as fh:
                    import traceback
                    fh.write("asammdf open failed:\n")
                    fh.write(traceback.format_exc())
            except Exception:
                pass

    # Fallback: try mdfreader (log exceptions)
    if _HAVE_MDFREADER:
        try:
            import mdfreader
            try:
                m = mdfreader.Mdf(str(path))
                w = Wrap(m, "mdfr")
                try:
                    yield w
                finally:
                    try: w.close()
                    except Exception: pass
                return
            except Exception as e_mdfr_ctor:
                try:
                    app.logger.exception("mdfreader open failed for %s: %s", path, e_mdfr_ctor)
                except Exception:
                    print(f"[ERROR] mdfreader open failed for {path}: {e_mdfr_ctor}")
                try:
                    diag_file = (UPLOAD_DIR / (Path(path).name + ".open_mdfr_error.txt"))
                    import traceback
                    with open(diag_file, "w", encoding="utf-8") as fh:
                        fh.write("mdfreader open failed:\n")
                        fh.write(traceback.format_exc())
                except Exception:
                    pass
        except Exception as e_mdfr_imp:
            try:
                app.logger.exception("mdfreader import failed while opening %s: %s", path, e_mdfr_imp)
            except Exception:
                print(f"[ERROR] mdfreader import failed: {e_mdfr_imp}")

    # If neither library could open it, yield None (caller must handle)
    yield None


class _MDFCtxSimple:
    def __init__(self,path): self._p=str(path); self._m=None
    def __enter__(self):
        from asammdf import MDF as _M
        self._m=_M(self._p); return self._m
    def __exit__(self,exc_type,exc,tb):
        try: self._m.close()
        except: pass

def open_mdf_simple(path: Path):
    return _MDFCtxSimple(path)

# ------------------------------------------------------------------
# Generate many normalized aliases for a given channel name
def normalize_channel_aliases(ch: str) -> List[str]:
    """
    Return a list of alias strings for channel name `ch`.
    Aliases include:
      - original channel name
      - URL-decoded/encoded forms
      - progressive suffix-stripped forms (remove trailing .EA .E8 .EE .E8.001 etc.)
      - remove 'OBD Mode ... #' prefix if present
      - array base (Signal[0] -> Signal)
      - lowercase and _clean_name lowercased
    """
    aliases = []
    if not ch:
        return aliases

    # original
    aliases.append(ch)

    # URL decoded / encoded variants
    try:
        dec = urllib.parse.unquote(ch)
        if dec and dec != ch:
            aliases.append(dec)
    except Exception:
        pass

    try:
        enc = urllib.parse.quote(ch)
        if enc and enc != ch:
            aliases.append(enc)
    except Exception:
        pass

    # Remove common 'wrapper' prefixes like "OBD Mode 1 (Can) #1." or other "prefix.SIGNAL"
    # Pattern: anything up to first '.' that contains '#\d+' or 'OBD' or 'Mode' etc.
    # We will attempt to remove common verbose prefixes heuristically.
    try:
        # Remove leading tags like "OBD Mode 1 (Can) #1." by searching for the last occurrence of
        # a recognizable token before the "signal" part. If the name contains '.' and left side is long, keep rightmost part.
        if '.' in ch:
            # prefer the RHS after the last '.' as candidate short name
            last = ch.split('.')[-1]
            if last and last != ch:
                aliases.append(last)
            # also add the part after the last occurrence of '#\d+.' if present
            m = re.search(r'#\d+\.(.+)', ch)
            if m:
                short = m.group(1)
                if short and short not in aliases:
                    aliases.append(short)
            # also add everything after the first '.' if that looks meaningful
            first_after = ch.split('.', 1)[1] if '.' in ch else None
            if first_after and first_after not in aliases:
                aliases.append(first_after)
    except Exception:
        pass

    # Iteratively strip trailing dot-suffices that look like MDF tags, e.g. ".EA", ".E8", ".E8.001", ".01.91.8D"
    # Add aliases by removing one trailing segment at a time while the trailing segment matches [A-Za-z0-9]+ or hex-like sequences.
    try:
        parts = ch.split('.')
        # if there are at least 2 parts, iteratively drop the last part
        if len(parts) > 1:
            for i in range(len(parts)-1, 0, -1):
                candidate = '.'.join(parts[:i])
                # avoid identical repeats
                if candidate and candidate not in aliases:
                    aliases.append(candidate)
    except Exception:
        pass

    # Array notation: Signal[0] -> Signal
    try:
        if '[' in ch and ']' in ch:
            base = ch.split('[')[0]
            if base and base not in aliases:
                aliases.append(base)
    except Exception:
        pass

    # Lowercase and cleaned versions
    try:
        aliases_lower = []
        for a in aliases:
            al = a.lower()
            aliases_lower.append(al)
            # also cleaned (remove non-alnum to underscore)
            cleaned = re.sub(r'[^0-9a-zA-Z]+', '_', a).strip('_').lower()
            aliases_lower.append(cleaned)
        # append unique lower/cleaned aliases
        for al in aliases_lower:
            if al and al not in aliases:
                aliases.append(al)
    except Exception:
        pass

    # Deduplicate preserving order
    seen = set()
    out = []
    for a in aliases:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out

# ------------------------------------------------------------------
def expanded_aliases_for_file(path: Path) -> Dict[str, List[str]]:
    """
    For a given file return a dict canonical -> [aliases...]
    Uses normalize_channel_aliases to generate aliases for each canonical channel.
    This is the authoritative alias expansion used by build_signal_lookup and the UI.
    """
    out: Dict[str, List[str]] = {}
    try:
        canonals = list_channels(path)
        for can in canonals:
            try:
                aliases = normalize_channel_aliases(can)
                # always include the canonical itself as first alias
                if can not in aliases:
                    aliases.insert(0, can)
                # also ensure cleaned/lowercase forms included
                cleaned = re.sub(r'[^0-9a-zA-Z]+', '_', can).strip('_').lower()
                if cleaned and cleaned not in aliases:
                    aliases.append(cleaned)
                lowered = can.lower()
                if lowered not in aliases:
                    aliases.append(lowered)
            except Exception:
                aliases = [can]
            # dedupe preserving order
            seen = set(); uniq = []
            for a in aliases:
                if a not in seen:
                    seen.add(a); uniq.append(a)
            out[can] = uniq
    except Exception:
        pass
    return out

# ------------------------------------------------------------------
def list_channels(path: Path) -> List[str]:
    """
    Return canonical channel list for `path`.
    Supports MDF/MF4 files (via asammdf) and CSV/Excel files (via pandas).
    Cache is keyed by path string and stores canonical names only.
    """
    k = str(path)
    if k in CHANNELS_CACHE:
        # CHANNELS_CACHE will now store canonical list for each path
        return CHANNELS_CACHE[k]

    raw_chs: List[str] = []
    suffix_lower = path.suffix.lower()
    
    # Handle CSV/Excel files
    if suffix_lower in {'.csv', '.xlsx', '.xls'}:
        try:
            import pandas as pd
            # Read just headers to get column names
            if suffix_lower == '.csv':
                # Try multiple encodings
                for enc in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        df_header = pd.read_csv(path, nrows=0, encoding=enc, low_memory=False)
                        raw_chs = list(df_header.columns)
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
            else:
                # Excel files
                try:
                    df_header = pd.read_excel(path, nrows=0, engine='openpyxl' if suffix_lower == '.xlsx' else None)
                    raw_chs = list(df_header.columns)
                except Exception:
                    # Fallback to default engine
                    try:
                        df_header = pd.read_excel(path, nrows=0)
                        raw_chs = list(df_header.columns)
                    except Exception:
                        pass
        except Exception:
            pass  # Return empty list if pandas not available or file read fails
    else:
        # Handle MDF/MF4 files
        with open_mdf(path) as m:
            if m:
                try:
                    # Prefer canonical list from asammdf/mdfreader
                    raw_chs = m.channels()
                except Exception:
                    try:
                        raw_chs = list(getattr(m._o, "channels_db", {}).keys())
                    except Exception:
                        raw_chs = []

    # Ensure we keep canonical names only (dedup, preserve order)
    seen = set()
    canon = []
    for ch in raw_chs:
        if ch and ch not in seen:
            seen.add(ch)
            canon.append(ch)

    CHANNELS_CACHE[k] = canon
    return canon

# ------------------------------------------------------------------
# Robust read_signal: try asammdf first, then mdfreader fallback
def read_signal(path: Path, requested_ch: str):
    """
    Read requested channel from path. Returns (timestamps_list, samples_list, unit_str).
    Tries asammdf first, and if that yields empty samples, falls back to mdfreader.
    """
    import re, urllib.parse

    # -------------------------------
    # 1) Try with asammdf
    with open_mdf(path) as m:
        if m:
            candidates = [
                requested_ch,
                requested_ch.lower(),
                re.sub(r'[^0-9a-zA-Z]+', '_', requested_ch).strip('_').lower(),
            ]
            try:
                candidates += normalize_channel_aliases(requested_ch)
            except Exception:
                pass

            seen = set()
            cand_list = []
            for c in candidates:
                if c and c not in seen:
                    seen.add(c)
                    cand_list.append(c)

            for cand in cand_list:
                try:
                    sig = m.get(cand)
                    if sig is None:
                        continue

                    # Try extracting
                    try:
                        tlist = sig.timestamps.tolist()
                        vlist = sig.samples.tolist()
                        unit = getattr(sig, "unit", "") or ""
                    except Exception:
                        try:
                            # Handle tuple returns (timestamps, values, unit)
                            if isinstance(sig, (tuple, list)) and len(sig) >= 2:
                                tlist = list(sig[0])
                                vlist = list(sig[1])
                                unit = sig[2] if len(sig) > 2 else ""
                            else:
                                tlist = list(sig[0])
                                vlist = list(sig[1])
                                unit = ""
                        except Exception:
                            tlist, vlist, unit = [], [], ""

                    if vlist:
                        if isinstance(unit, bytes):
                            try:
                                unit = unit.decode("utf-8", "ignore")
                            except:
                                unit = str(unit)
                        unit = unit or ""
                        app.logger.debug(
                            "read_signal: asammdf returned %d samples for %s (requested %s)",
                            len(vlist), cand, requested_ch
                        )
                        return (tlist, vlist, unit)
                    else:
                        app.logger.debug(
                            "read_signal: asammdf returned 0 samples for %s (requested %s)",
                            cand, requested_ch
                        )
                except Exception as e:
                    app.logger.debug("asammdf attempt failed for %s: %s", cand, repr(e))

    # -------------------------------
    # 2) Fallback with mdfreader
    try:
        import mdfreader
        mr = mdfreader.Mdf(str(path))
        try:
            ch_keys = []
            try:
                ch_keys = list(getattr(mr, "channels").keys())
            except Exception:
                pass

            # best match finder
            def find_best_key(cand_list, keys):
                for c in cand_list:
                    if c in keys:
                        return c
                    for k in keys:
                        if c.lower() == k.lower():
                            return k
                for c in cand_list:
                    for k in keys:
                        if c.lower() in k.lower():
                            return k
                return None

            cand_keys = cand_list if 'cand_list' in locals() else [requested_ch]
            best = find_best_key(cand_keys, ch_keys)
            if not best:
                if requested_ch in ch_keys:
                    best = requested_ch
                else:
                    for k in ch_keys:
                        if requested_ch.lower() in k.lower():
                            best = k
                            break
            if not best:
                app.logger.debug("mdfreader found no key for %s", requested_ch)
                return ([], [], "")

            # try extracting data
            arr = None
            try:
                arr = mr.get_channel_data(best)
            except Exception:
                chobj = mr.channels.get(best)
                if isinstance(chobj, dict) and 'data' in chobj:
                    arr = chobj['data']

            if arr is None:
                app.logger.debug("mdfreader returned no data for %s", requested_ch)
                return ([], [], "")

            import numpy as np
            a = np.asarray(arr)
            if a.ndim == 2 and a.shape[1] >= 2:
                tlist = a[:, 0].tolist()
                vlist = a[:, 1].tolist()
            else:
                vlist = a.flatten().tolist()
                tlist = []

            app.logger.debug(
                "read_signal: mdfreader returned %d samples for key %s (requested %s)",
                len(vlist), best, requested_ch
            )
            return (tlist, vlist, "")
        finally:
            try:
                mr.close()
            except:
                pass
    except Exception as e:
        app.logger.debug("mdfreader fallback failed for %s: %s", requested_ch, repr(e))

    # -------------------------------
    # nothing worked
    app.logger.debug("read_signal: no data for %s", requested_ch)
    return ([], [], "")

# ------------------------------------------------------------------ Channel discovery & series extraction
def discover_channels(files: List[Path])->List[Dict]:
    """
    Discover channels from files (MDF/MF4/CSV/Excel).
    Uses signal mapping system to identify signals by role in CSV/Excel files.
    """
    try:
        from signal_mapping import find_signal_in_dataframe, SIGNAL_MAP
    except ImportError:
        find_signal_in_dataframe = None
    
    universe=set(); per=[]
    # Track signal role mappings for CSV/Excel files
    csv_excel_role_mappings = {}  # {file_path: {role: column_name}}
    
    for f in files:
        try:
            chs=list_channels(f)
            s=set(chs)
            per.append(s)
            universe.update(chs)
            
            # For CSV/Excel files, also try to map columns to signal roles
            if find_signal_in_dataframe and f.suffix.lower() in {'.csv', '.xlsx', '.xls'}:
                try:
                    import pandas as pd
                    # Read a small sample to get columns
                    if f.suffix.lower() == '.csv':
                        df_sample = pd.read_csv(f, nrows=1, low_memory=False)
                    else:
                        df_sample = pd.read_excel(f, nrows=1)
                    
                    columns = list(df_sample.columns)
                    role_mapping = {}
                    
                    # Try to find signals for key roles
                    for role in ['rpm', 'torque', 'vehicle_speed', 'fuel_rate', 'lambda', 
                                'coolant_temp', 'throttle', 'map_sensor', 'gear']:
                        found_col = find_signal_in_dataframe(columns, role)
                        if found_col:
                            role_mapping[role] = found_col
                            # Add to universe if not already there
                            if found_col not in universe:
                                universe.add(found_col)
                    
                    if role_mapping:
                        csv_excel_role_mappings[str(f)] = role_mapping
                except Exception:
                    pass  # Skip if CSV/Excel parsing fails
        except:
            per.append(set())
    
    out=[]
    for name in sorted(universe):
        c=sum(1 for s in per if name in s)
        display_name = _extract_display_name(name)
        
        # Check if this column maps to a signal role in any CSV/Excel file
        signal_role = None
        for file_path, mappings in csv_excel_role_mappings.items():
            for role, col_name in mappings.items():
                if col_name == name:
                    signal_role = role
                    break
            if signal_role:
                break
        
        out.append({
            "id": name,
            "name": display_name,  # Clean display name for UI
            "label": display_name,  # Use display name for labels
            "clean": _clean_name(name),
            "full_name": name,  # Keep full name for reference
            "presence": f"{c}/{len(files)}",
            "present_count": c,
            "occurrence": c,
            "signal_role": signal_role,  # Add detected signal role if found
        })
    return out

def _downsample_xy(xs,ys,max_pts:int):
    if not xs or not ys or len(xs)!=len(ys) or max_pts<=0: return xs,ys
    n=len(xs)
    if n<=max_pts: return xs,ys
    step=max(1,n//max_pts)
    return xs[::step], ys[::step]

# --------------- LTTB Downsampling (Largest-Triangle-Three-Buckets) -----------------
def lttb_downsample(xs: List[float], ys: List[float], threshold: int) -> Tuple[List[float], List[float]]:
    """
    LTTB algorithm: returns visually representative subset.
    """
    try:
        import math
    except ImportError:
        return xs, ys
    n = len(xs)
    if threshold >= n or threshold < 3:
        return xs, ys
    bucket_size = (n - 2) / (threshold - 2)
    sampled_x = [xs[0]]
    sampled_y = [ys[0]]
    a = 0
    for i in range(1, threshold - 1):
        start = int((i - 1) * bucket_size) + 1
        end   = int(i * bucket_size) + 1
        if end >= n: end = n - 1
        bucket_range = range(start, end)
        next_start = int(i * bucket_size) + 1
        next_end   = int((i + 1) * bucket_size) + 1
        if next_end >= n: next_end = n
        avg_x = avg_y = 0.0
        avg_count = next_end - next_start
        if avg_count <= 0:
            avg_count = 1
            avg_x = xs[a]; avg_y = ys[a]
        else:
            for idx in range(next_start, next_end):
                avg_x += xs[idx]; avg_y += ys[idx]
            avg_x /= avg_count; avg_y /= avg_count
        ax = xs[a]; ay = ys[a]
        max_area=-1.0; chosen_index=None
        for idx in bucket_range:
            area=abs((ax - avg_x)*(ys[idx]-ay) - (ax - xs[idx])*(avg_y - ay))*0.5
            if area>max_area:
                max_area=area; chosen_index=idx
        if chosen_index is None: chosen_index=start
        sampled_x.append(xs[chosen_index]); sampled_y.append(ys[chosen_index])
        a=chosen_index
    sampled_x.append(xs[-1]); sampled_y.append(ys[-1])
    return sampled_x, sampled_y
# -------------------------------------------------------------------------------

def extract_series(files: List[Path], ids: List[str], include_time=True, normalize=False, max_points=100_000):
    cache_key=None
    if not normalize and include_time and max_points>=100_000:
        key_files=tuple(sorted(str(f) for f in files))
        key_ids=tuple(sorted(ids)); cache_key=(key_files,key_ids)
        if cache_key in SERIES_CACHE: return SERIES_CACHE[cache_key]
    
    ch_map = {}
    for f in files:
        for ch in list_channels(f):
            canonical = ch  # keep canonical stored value
            # register ALL aliases for this canonical channel
            for alias in _aliases_for_channel(ch):
                # do not overwrite an existing mapping to a canonical value
                if alias not in ch_map:
                    ch_map[alias] = canonical

    series={}
    for req in ids:
        raw=ch_map.get(req)
        if not raw: continue
        unit=None; all_t=[]; all_v=[]; last_t=None; last_dt=None
        for f in files:
            t,v,u=read_signal(f,raw)
            if not t or not v: continue
            unit=u or unit
            if include_time:
                if len(t)>=2: dt_local=(t[-1]-t[0])/(len(t)-1)
                else: dt_local=None
                step=dt_local if dt_local is not None else (last_dt if last_dt is not None else 0.0)
                tol=(step*2) if step and step>0 else 0.0
                if last_t is None:
                    shifted=t
                else:
                    delta=t[0]-last_t
                    if delta>=-tol:
                        shifted=t
                    else:
                        use_step=step if step and step>0 else 0.0
                        offset=(last_t+use_step)-t[0]
                        shifted=[ti+offset for ti in t]
                all_t.extend(shifted); last_t=shifted[-1]
                if len(shifted)>=2:
                    last_dt=(shifted[-1]-shifted[0])/(len(shifted)-1)
            else:
                base=0 if not all_t else (all_t[-1]+1)
                all_t.extend([base+i for i in range(len(v))])
            all_v.extend(v)
        if not all_t or not all_v: continue
        if normalize:
            mn,mx=min(all_v),max(all_v)
            if mx>mn: all_v=[(x-mn)/(mx-mn) for x in all_v]
        all_t,all_v=_downsample_xy(all_t,all_v,max_points)
        series[req]={"name":_clean_name(raw),"timestamps":all_t,"values":all_v,"unit":unit}
    if cache_key is not None: SERIES_CACHE[cache_key]=series
    return series

def basic_stats(series_map: Dict[str,Dict])->List[Dict]:
    out=[]
    for cid,data in series_map.items():
        vals=[v for v in data['values'] if isinstance(v,(int,float))]
        if not vals: continue
        out.append({"signal":data['name'],
                    "min":round(min(vals),6),
                    "mean":round(sum(vals)/len(vals),6),
                    "max":round(max(vals),6)})
    return out

def compute_fft(series_map: Dict[str,Dict], n_fft=2048):
    try: import numpy as np
    except ImportError: return {}
    out={}
    for k,d in series_map.items():
        vals=[v for v in d['values'] if isinstance(v,(int,float))]
        if len(vals)<2: continue
        import numpy as np
        arr=np.array(vals)
        fft=np.fft.fft(arr,n=n_fft); freq=np.fft.fftfreq(n_fft)
        out[k]={"freq":freq.tolist(),"amplitude":abs(fft).tolist()}
    return out

def compute_hist(series_map: Dict[str,Dict], bins=50):
    try: import numpy as np
    except ImportError: return {}
    out={}
    import numpy as np
    for k,d in series_map.items():
        vals=[v for v in d['values'] if isinstance(v,(int,float))]
        if not vals: continue
        h,edges=np.histogram(vals,bins=bins)
        out[k]={"hist":h.tolist(),"bins":edges.tolist()}
    return out

# ------------------------------------------------------------------
def build_signal_lookup(files: List[Path])->Dict[str,str]:
    lookup = {}
    for f in files:
        for ch in list_channels(f):
            for alias in _aliases_for_channel(ch):
                if alias not in lookup:
                    lookup[alias] = ch
    return lookup

def get_single_series(files: List[Path],
                      sig_id: str,
                      tmin: Optional[float]=None,
                      tmax: Optional[float]=None,
                      max_points: int = 50_000) -> Optional[Dict[str,Any]]:
    if not files or not sig_id:
        return None
    candidate_ids=[sig_id]
    cleaned=_clean_name(sig_id)
    if cleaned != sig_id:
        candidate_ids.append(cleaned)
    series_map=extract_series(files,candidate_ids,include_time=True,normalize=False,max_points=max_points*2)
    chosen=None
    for cid in candidate_ids:
        if cid in series_map:
            chosen=cid; break
    if not chosen and series_map:
        chosen=next(iter(series_map.keys()))
    if not chosen:
        return None
    data=series_map[chosen]
    xs=data["timestamps"]; ys=data["values"]
    if (tmin is not None) or (tmax is not None):
        sel=[]
        for i,t in enumerate(xs):
            if tmin is not None and t < tmin: continue
            if tmax is not None and t > tmax: continue
            sel.append(i)
        if sel:
            xs=[xs[i] for i in sel]; ys=[ys[i] for i in sel]
    if max_points and len(xs)>max_points:
        step=max(1,len(xs)//max_points)
        xs=xs[::step]; ys=ys[::step]
    return {
        "id": sig_id,
        "resolved_id": chosen,
        "name": data["name"],
        "unit": data.get("unit",""),
        "timestamps": xs,
        "values": ys,
        "n": len(xs)
    }

# ------------------------------------------------------------------ CC/SL detection (legacy + overshoot)
_CC_PATTERNS={
    "actual":["vitesse_vehicule_roues","vehicule_roues","vehicle_speed","veh_speed","vehspd","vitesse_vehicule","spd","veh"],
    "cruise_set":["ext_spdvehvsregreq","vsregreq","vsreg","cruise_set","vsregset"],
    "limiter_set":["ext_spdvehvslimreq","vslimreq","vslim","limiter_set","vslimset","speed_limit","spdlimit"],
    "cruise_flag":["vsctl_stvsregextd","vsregextd","cruise_flag","vsctl_vsreg"],
    "limiter_flag":["vsctl_stvslimextd","vslimextd","limiter_flag","vsctl_vslim"]
}

def _match_channel_any(channels: List[str], patterns: List[str])->str|None:
    lowers=[(c,c.lower()) for c in channels]
    for p in patterns:
        for orig,low in lowers:
            if p in low: return orig
    return None

def analyze_ccsl_overshoot(files: List[Path],
                           overshoot_margin=ENV_OVERSHOOT_MARGIN,
                           flag_value=ENV_OVERSHOOT_FLAG_VALUE,
                           fallback_no_flags=ENV_OVERSHOOT_FALLBACK_NO_FLAGS)->Dict:
    summary=[]; plots=[]; diag={"files_total":len(files),"missing_required":0,"no_actual_data":0,"ok":0,"fallback_no_flags_used":False}
    for f in files:
        chs=list_channels(f)
        if not chs:
            summary.append({"file":f.name,"status":"no_channels"}); continue
        actual_ch=_match_channel_any(chs,_CC_PATTERNS["actual"])
        cruise_set_ch=_match_channel_any(chs,_CC_PATTERNS["cruise_set"])
        limiter_set_ch=_match_channel_any(chs,_CC_PATTERNS["limiter_set"])
        cruise_flag_ch=_match_channel_any(chs,_CC_PATTERNS["cruise_flag"])
        limiter_flag_ch=_match_channel_any(chs,_CC_PATTERNS["limiter_flag"])
        if not actual_ch or (not cruise_set_ch and not limiter_set_ch):
            summary.append({"file":f.name,"status":"missing_required","actual":bool(actual_ch),
                            "has_set":bool(cruise_set_ch or limiter_set_ch)})
            diag["missing_required"]+=1
            continue
        t_act,act_vals,_=read_signal(f,actual_ch)
        if not t_act or not act_vals:
            summary.append({"file":f.name,"status":"no_actual_data"})
            diag["no_actual_data"]+=1
            continue
        _, cruise_set_vals,_=read_signal(f,cruise_set_ch) if cruise_set_ch else ([],[], "")
        _, limiter_set_vals,_=read_signal(f,limiter_set_ch) if limiter_set_ch else ([],[], "")
        _, cruise_flag_vals,_=read_signal(f,cruise_flag_ch) if cruise_flag_ch else ([],[], "")
        _, limiter_flag_vals,_=read_signal(f,limiter_flag_ch) if limiter_flag_ch else ([],[], "")
        use_fallback=fallback_no_flags and not cruise_flag_ch and not limiter_flag_ch
        if use_fallback: diag["fallback_no_flags_used"]=True
        n=len(act_vals)
        def trunc(a): return a[:n] if a else [None]*n
        cruise_set_vals=trunc(cruise_set_vals)
        limiter_set_vals=trunc(limiter_set_vals)
        cruise_flag_vals=trunc(cruise_flag_vals)
        limiter_flag_vals=trunc(limiter_flag_vals)
        set_speed=[]; overshoot_mask=[]; cruise_events=0; limiter_events=0; diffs=[]
        for i in range(n):
            cur=None; mode=None
            if cruise_flag_vals and cruise_flag_vals[i]==flag_value and cruise_set_vals[i] is not None:
                cur=cruise_set_vals[i]; mode="cruise"
            elif limiter_flag_vals and limiter_flag_vals[i]==flag_value and limiter_set_vals[i] is not None:
                cur=limiter_set_vals[i]; mode="limiter"
            elif use_fallback:
                if cruise_set_vals and cruise_set_vals[i] not in (None,0):
                    cur=cruise_set_vals[i]; mode="cruise"
                elif limiter_set_vals and limiter_set_vals[i] not in (None,0):
                    cur=limiter_set_vals[i]; mode="limiter"
            set_speed.append(cur)
            if cur is not None and isinstance(act_vals[i],(int,float)) and isinstance(cur,(int,float)):
                if act_vals[i] > cur + overshoot_margin:
                    overshoot_mask.append(True)
                    diffs.append(act_vals[i]-cur)
                    if mode=="cruise": cruise_events+=1
                    elif mode=="limiter": limiter_events+=1
                else:
                    overshoot_mask.append(False)
            else:
                overshoot_mask.append(False)
        total_overshoot=sum(overshoot_mask)
        max_diff=max(diffs) if diffs else 0.0
        avg_diff=(sum(diffs)/len(diffs)) if diffs else 0.0
        pct_time=(total_overshoot/n*100.0) if n else 0.0
        summary.append({
            "file":f.name,"status":"ok",
            "cruise_events":cruise_events,
            "limiter_events":limiter_events,
            "total_overshoot":total_overshoot,
            "max_overshoot_kmh":round(max_diff,3),
            "avg_overshoot_kmh":round(avg_diff,3),
            "pct_time_overshoot":round(pct_time,3),
            "actual_ch":actual_ch,
            "cruise_set_ch":cruise_set_ch,
            "limiter_set_ch":limiter_set_ch,
            "cruise_flag_ch":cruise_flag_ch,
            "limiter_flag_ch":limiter_flag_ch,
            "fallback_mode":bool(use_fallback)
        })
        diag["ok"]+=1
        overshoot_times=[t_act[i] for i,v in enumerate(overshoot_mask) if v]
        overshoot_vals=[act_vals[i] for i,v in enumerate(overshoot_mask) if v]
        fig_data=[
            {"type":"scatter","mode":"lines","name":"Actual Speed","x":t_act,"y":act_vals,"line":{"color":"#ffffff","width":1.5}},
            {"type":"scatter","mode":"lines","name":"Set Speed","x":t_act,"y":set_speed,"line":{"color":"#FF9800","width":2.5,"dash":"dash"}}
        ]
        if overshoot_times:
            fig_data.append({"type":"scatter","mode":"markers","name":"Overshoot","x":overshoot_times,"y":overshoot_vals,
                             "marker":{"color":"#ff4136","size":6,"symbol":"circle"}})
        fig_layout={"title":f"CC/SL Overshoot: {f.name}",
                    "xaxis":{"title":"Time (s)"},
                    "yaxis":{"title":"Speed (km/h)"},
                    "template":"plotly_dark",
                    "margin":{"t":50,"l":70,"r":30,"b":50},
                    "height":480,
                    "legend":{"orientation":"h"}}
        plots.append({"name":f"Overshoot - {f.name}","plotly_json":{"data":fig_data,"layout":fig_layout}})
    return {"summary":summary,"plots":plots,"diag":diag,
            "config":{"margin_kmh":overshoot_margin,"flag_value":flag_value,
                      "fallback_no_flags":fallback_no_flags}}

def report_ccsl(files: List[Path])->Dict:
    rows=[]
    for f in files:
        try:
            with open_mdf(f) as m:
                if not m: continue
                def pick(*need):
                    nd=[n.lower() for n in need]
                    for ch in m.channels():
                        lc=ch.lower()
                        if all(n in lc for n in nd): return ch
                    return None
                v_act=pick("spd","veh") or pick("veh","speed")
                v_cc=pick("vsreg") or pick("cruise","set")
                v_sl=pick("vslim") or pick("speed","limit")
                if not v_act: continue
                _, act,_ = m.get(v_act)
                if v_cc:
                    _, cc,_=m.get(v_cc)
                    errs=[a-c for a,c in zip(act,cc) if isinstance(a,(int,float)) and isinstance(c,(int,float)) and c>0]
                    if errs:
                        rows.append({"file":f.name,"mode":"CruiseControl","samples":len(errs),
                                     "avg_error":round(sum(errs)/len(errs),6),
                                     "max_over":round(max(errs),6),
                                     "max_under":round(min(errs),6)})
                if v_sl:
                    _, sl,_=m.get(v_sl)
                    errs=[a-s for a,s in zip(act,sl) if isinstance(a,(int,float)) and isinstance(s,(int,float)) and s>0]
                    if errs:
                        rows.append({"file":f.name,"mode":"SpeedLimiter","samples":len(errs),
                                     "avg_error":round(sum(errs)/len(errs),6),
                                     "max_over":round(max(errs),6),
                                     "max_under":round(min(errs),6)})
        except: pass
    return {"summary":rows}

# Aggregated CC/SL plots
def make_ccsl_plot(df, mode="cruise", thr=2.0):
    if go is None or df is None or df.empty:
        return None, None
    actual_candidates=[
        "Veh_spdVeh","Ext_spdVeh","VehV_v","VITESSE_VEHICULE_ROUES",
        "vehicle_speed","veh_speed","vehspd","spd"
    ]
    if mode=="cruise":
        set_candidates=["Ext_spdVehVSRegReq","VSCtl_spdVehVSRegReq","vsreg","cruise_set","vsregreq"]
        title="Cruise Control Overshoot"
    else:
        set_candidates=["Ext_spdVehVSLimReq","VSCtl_spdVehVSLimReq","vslim","limiter_set","vslimreq","speed_limit"]
        title="Speed Limiter Overshoot"
    high_col=next((c for c in actual_candidates if c in df.columns),None)
    low_col =next((c for c in set_candidates if c in df.columns),None)
    if not high_col or not low_col:
        return None, None
    overshoot_limit=df[low_col]+thr
    overshoot_flag=(df[high_col] > overshoot_limit) & df[low_col].notna()
    overshoot_idx=df.index[overshoot_flag]
    fig=go.Figure()
    # Vibrant colorful scheme - FEV brand colors and complementary vibrant colors
    if mode == "cruise":
        actual_color = "#2196F3"  # Vibrant blue
        set_color = "#FF9800"     # Vibrant orange
        threshold_color = "#F44336" # Vibrant red
        marker_color = "#E91E63"  # FEV pink
    else:
        actual_color = "#4CAF50"  # Vibrant green
        set_color = "#9C27B0"     # Vibrant purple
        threshold_color = "#F44336" # Vibrant red
        marker_color = "#E91E63"  # FEV pink
    
    fig.add_trace(go.Scatter(x=df.index,y=df[high_col],name="Actual Speed",line=dict(color=actual_color, width=2.5)))
    fig.add_trace(go.Scatter(x=df.index,y=df[low_col],name=f"{mode.title()} Set",line=dict(color=set_color, dash="dash", width=2.5)))
    fig.add_trace(go.Scatter(x=df.index,y=overshoot_limit,name=f"Threshold (+{thr:.1f} km/h)",line=dict(color=threshold_color, dash="dot", width=2)))
    if len(overshoot_idx):
        fig.add_trace(go.Scatter(x=overshoot_idx,y=df.loc[overshoot_idx,high_col],
                                 mode="markers",name="Overshoot Exceedance",
                                 marker=dict(color=marker_color, size=10, symbol="x", 
                                            line=dict(color="#FFFFFF", width=1.5))))
    fig.update_layout(title=title,xaxis_title="Time (s)",yaxis_title="Speed (km/h)",
                      template="plotly_dark",legend=dict(orientation="h",y=-0.25),
                      margin=dict(t=50,l=70,r=30,b=50),height=480,
                      paper_bgcolor='black',  # Deep black background like IUPR/fuel
                      plot_bgcolor='black',  # Deep black background like IUPR/fuel
                      font=dict(color='#dce1e6'))  # Light text for dark mode
    summary={
        "Mode":mode,
        "Threshold (+km/h)":thr,
        "Overshoot Events":int(overshoot_flag.sum()),
        "Max Overshoot (km/h)": float(((df[high_col]-df[low_col]) - thr).where(overshoot_flag).max()
                                      if overshoot_flag.any() else 0.0)
    }
    return fig.to_plotly_json(), summary

def build_mode_aggregated_ccsl(files: List[Path], thr: float)->Dict[str,Any]:
    if pd is None or go is None:
        return {"plots":{},"tables":{}}
    cruise_rows=[]; limiter_rows=[]
    time_offset=0.0
    def choose(chs,patterns):
        lowers=[(c,c.lower()) for c in chs]
        for p in patterns:
            for orig,low in lowers:
                if p in low: return orig
        return None
    for f in files:
        chs=list_channels(f)
        if not chs: continue
        actual=choose(chs,_CC_PATTERNS["actual"])
        cruise=choose(chs,_CC_PATTERNS["cruise_set"])
        limiter=choose(chs,_CC_PATTERNS["limiter_set"])
        if not actual: continue
        t_act,act_vals,_=read_signal(f,actual)
        if not t_act or not act_vals: continue
        base_shift=time_offset - t_act[0] if t_act else time_offset
        t_adj=[ti+base_shift for ti in t_act]
        if t_adj:
            time_offset = t_adj[-1] + ((t_adj[-1]-t_adj[0])/(len(t_adj)-1) if len(t_adj)>1 else 1.0)
        if cruise:
            _, cruise_vals,_=read_signal(f,cruise)
            if cruise_vals:
                n=min(len(t_adj),len(cruise_vals),len(act_vals))
                cruise_rows.append(pd.DataFrame({"time":t_adj[:n],"Actual":act_vals[:n],"CruiseSet":cruise_vals[:n]}))
        if limiter:
            _, limiter_vals,_=read_signal(f,limiter)
            if limiter_vals:
                n=min(len(t_adj),len(limiter_vals),len(act_vals))
                limiter_rows.append(pd.DataFrame({"time":t_adj[:n],"Actual":act_vals[:n],"LimiterSet":limiter_vals[:n]}))
    plots={}; tables={}
    if cruise_rows:
        dfc=pd.concat(cruise_rows,ignore_index=True).set_index("time")
        j,s=make_ccsl_plot(dfc.rename(columns={"Actual":"Veh_spdVeh","CruiseSet":"Ext_spdVehVSRegReq"}),mode="cruise",thr=thr)
        if j:
            plots["Cruise Control Overshoot"]={"type":"plotly","plotly_json":j}
            tables["Cruise Overshoot Summary"]=[s]
    if limiter_rows:
        dfl=pd.concat(limiter_rows,ignore_index=True).set_index("time")
        j,s=make_ccsl_plot(dfl.rename(columns={"Actual":"Veh_spdVeh","LimiterSet":"Ext_spdVehVSLimReq"}),mode="limiter",thr=thr)
        if j:
            plots["Speed Limiter Overshoot"]={"type":"plotly","plotly_json":j}
            tables["Limiter Overshoot Summary"]=[s]
    return {"plots":plots,"tables":tables}

# ------------------------------------------------------------------ DFC code frequency & segment plots
def _extract_dfc_rows_info(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    code_keys=['code','Code','dfc_code','DFC']
    name_keys=['DFC_name','dfc_name','name','Name']
    count_keys=['count','occurrences','runs','row_count','event_count','frequency','freq']
    segment_container_keys=['segments','runs','occurrences_list']
    segment_start_keys=['start','start_time','begin','t0']
    segment_end_keys=['end','end_time','finish','t1']
    norm=[]
    for row in summary_rows or []:
        code_val=None
        for ck in code_keys:
            if ck in row:
                try: code_val=int(str(row[ck]),0)
                except: pass
                if code_val is not None: break
        if code_val is None: continue
        name_val=None
        for nk in name_keys:
            if nk in row and row[nk]:
                name_val=str(row[nk]); break
        if not name_val: name_val=f"Code_{code_val}"
        count_val=0
        if 'event_count' in row and isinstance(row['event_count'],(int,float)):
            count_val=int(row['event_count'])
        elif 'row_count' in row and isinstance(row['row_count'],(int,float)):
            count_val=int(row['row_count'])
        else:
            for ck in count_keys:
                if ck in row and isinstance(row[ck],(int,float)):
                    count_val=int(row[ck]); break
        # Extract enhanced fields if available
        enhanced_fields = {}
        for field in ['severity', 'priority', 'code_type', 'dtc_format', 
                     'first_seen', 'last_seen', 'max_duration', 'total_duration', 'status_byte']:
            if field in row:
                enhanced_fields[field] = row[field]
        segments=[]
        for cont in segment_container_keys:
            v=row.get(cont)
            if isinstance(v,list):
                for seg in v:
                    if not isinstance(seg,dict): continue
                    s_val=None; e_val=None
                    for sk in segment_start_keys:
                        if sk in seg:
                            try: s_val=float(seg[sk])
                            except: pass
                            if s_val is not None: break
                    for ek in segment_end_keys:
                        if ek in seg:
                            try: e_val=float(seg[ek])
                            except: pass
                            if e_val is not None: break
                    if s_val is not None and e_val is not None and e_val>=s_val:
                        segments.append({'start':s_val,'end':e_val})
        norm_entry = {'code':code_val,'name':name_val,'count':count_val,'segments':segments}
        norm_entry.update(enhanced_fields)  # Add enhanced fields
        norm.append(norm_entry)
    return norm

def build_dfc_code_plots(dfc_obj: Dict[str, Any]) -> Dict[str, Any]:
    if go is None:
        return {'plots':{},'tables':{}}
    summary_rows=dfc_obj.get('summary',[]) or []
    rows_info=_extract_dfc_rows_info(summary_rows)
    if not rows_info: return {'plots':{},'tables':{}}
    # Enhanced freq_table with severity/priority if available
    freq_table_data = []
    for ri in rows_info:
        entry = {'code':ri['code'],'DFC_name':ri['name'],'count':ri['count']}
        # Add enhanced fields if available
        if 'severity' in ri:
            entry['severity'] = ri['severity']
        if 'priority' in ri:
            entry['priority'] = ri['priority']
        if 'code_type' in ri:
            entry['code_type'] = ri['code_type']
        freq_table_data.append(entry)
    freq_table=sorted(freq_table_data, key=lambda r:r['count'], reverse=True)
    plots={}
    tables={"DFC Code Frequency":freq_table}
    codes=[r['code'] for r in freq_table]
    names=[r['DFC_name'] for r in freq_table]
    counts=[r['count'] for r in freq_table]
    agg_fig=go.Figure()
    # Use severity-based coloring if available
    if 'severity' in freq_table[0] if freq_table else {}:
        severity_colors = {"critical": "#C30C36", "high": "#ff6f8a", "medium": "#f28e2b", "low": "#4e79a7"}
        colors = [severity_colors.get(r.get('severity', 'low'), "#4e79a7") for r in freq_table]
    else:
        colors = "#C30C36"
    
    agg_fig.add_trace(go.Bar(
        x=[str(c) for c in codes],
        y=counts,
        text=names,
        name="Count",
        marker=dict(color=colors if isinstance(colors, list) else colors)
    ))
    agg_fig.update_layout(
        title="DFC Code Frequency",
        xaxis_title="Code",
        yaxis_title="Count",
        template="plotly_dark",
        margin=dict(t=60,l=70,r=30,b=60),
        height=480
    )
    plots["DFC Code Frequency"]={"type":"plotly","plotly_json":agg_fig.to_json()}
    per_code_limit=DFC_CODE_PLOT_MAX
    for ri in freq_table[:per_code_limit]:
        code_val=ri['code']
        info=next((x for x in rows_info if x['code']==code_val),None)
        if not info: continue
        fig=go.Figure()
        # Enhanced title with priority/severity if available
        title_parts = [f"DFC Code {code_val}", info['DFC_name']]
        if 'priority' in info:
            title_parts.append(f"[{info['priority']}]")
        if 'severity' in info:
            title_parts.append(f"({info['severity']})")
        title = " – ".join(title_parts)
        if info['segments']:
            mid_x=[]; mid_y=[]; shapes=[]
            for idx,seg in enumerate(info['segments']):
                s=seg['start']; e=seg['end']; mid=(s+e)/2.0
                mid_x.append(mid); mid_y.append(idx+1)
                shapes.append(dict(
                    type="rect", xref="x", yref="y",
                    x0=s,x1=e,y0=idx+1-0.3,y1=idx+1+0.3,
                    line=dict(color="#ff6f8a",width=1),
                    fillcolor="rgba(195,12,54,0.25)"
                ))
            fig.add_trace(go.Scatter(
                x=mid_x,y=mid_y,mode="markers",
                marker=dict(color="#C30C36",size=8),
                name="Segments"
            ))
            fig.update_layout(
                title=title,
                xaxis_title="Time (s)",
                yaxis=dict(title="Segment #",autorange="reversed"),
                template="plotly_dark",
                height=400,
                margin=dict(t=50,l=80,r=30,b=50),
                shapes=shapes,
                legend=dict(orientation="h", y=-0.25)
            )
        else:
            fig.add_trace(go.Bar(
                x=[str(code_val)],
                y=[info['count']],
                marker=dict(color="#C30C36"),
                name="Count"
            ))
            fig.update_layout(
                title=title,
                xaxis_title="Code",
                yaxis_title="Count",
                template="plotly_dark",
                height=400,
                margin=dict(t=50,l=70,r=30,b=50)
            )
        plots[title]={"type":"plotly","plotly_json":fig.to_json()}
    return {'plots':plots,'tables':tables}

# ------------------------------------------------------------------ Plotly JSON normalization
def _ensure_plotly_json(pj):
    if isinstance(pj,str): return pj
    try: return json.dumps(pj)
    except Exception: return json.dumps({"error":"plotly_json_serialization_failed"})

# ------------------------------------------------------------------ DFC payload
def _build_dfc_payload(files_list):
    mapping_file = os.getenv("DFC_MAPPING", "")
    dfc = compute_dfc(
        files=[Path(p) for p in files_list],
        mapping_file=Path(mapping_file) if mapping_file else None,
        channels=None,
        compress_runs=True,
        topn=10,
        include_plots=True,
        enable_advanced_features=True,  # Enable enhanced DTC analysis
    )
    tables = {"DFC Summary": dfc.get("summary", [])}
    if dfc.get("channels"):
        tables["DFC Evidence Channels"] = dfc["channels"]
    plots={}
    for i,p in enumerate(dfc.get("plots",[]) or []):
        name=p.get("name",f"DFC Plot {i+1}")
        pj=p.get("plotly_json")
        plots[name]={"type":"plotly","plotly_json": pj if isinstance(pj,str) else json.dumps(pj)}
    # Add code frequency & per-code plots
    try:
        code_assets=build_dfc_code_plots(dfc)
        for tk,tv in (code_assets.get('tables') or {}).items():
            if tk not in tables: tables[tk]=tv
        for pk,pv in (code_assets.get('plots') or {}).items():
            if pk not in plots:
                plots[pk]={"type":"plotly","plotly_json":pv["plotly_json"] if isinstance(pv.get("plotly_json"),str)
                           else json.dumps(pv.get("plotly_json"))}
    except Exception as e:
        if DEBUG_MODE: print("[DEBUG] build_dfc_code_plots failed:", e)
    # Optional demo DFC_ST plot
    want_demo = os.getenv("DFC_ST_DEMO_PLOT","0").lower() in ("1","true","yes","on")
    has_interpolated = any("DFC_ST Signals" in k for k in plots.keys())
    if want_demo and not has_interpolated and "DFC_ST Demo" not in plots:
        try:
            import numpy as np
            if go is None: raise ImportError("Plotly not available")
            x=np.array([1,2,3,4,5]); y=np.array([1,3,2,3,1])
            demo=go.Figure()
            demo.add_trace(go.Scatter(x=x,y=y,name="linear",line_shape="linear"))
            demo.add_trace(go.Scatter(x=x,y=y+5,name="spline",
                                      text=["'smoothing' controls shape"],
                                      hoverinfo="text+name",line_shape="spline"))
            demo.add_trace(go.Scatter(x=x,y=y+10,name="vhv",line_shape="vhv"))
            demo.add_trace(go.Scatter(x=x,y=y+15,name="hvh",line_shape="hvh"))
            demo.add_trace(go.Scatter(x=x,y=y+20,name="vh",line_shape="vh"))
            demo.add_trace(go.Scatter(x=x,y=y+25,name="hv",line_shape="hv"))
            demo.update_traces(hoverinfo="text+name",mode="lines+markers")
            demo.update_layout(title="DFC_ST Line Shape Demo",template="plotly_dark",
                               legend=dict(y=0.5,traceorder="reversed",font_size=12),
                               margin=dict(t=50,l=70,r=30,b=50),height=480)
            plots["DFC_ST Demo"]={"type":"plotly","plotly_json":demo.to_json()}
        except Exception as _e:
            if DEBUG_MODE: print("[DEBUG] DFC_ST demo plot failed:", _e)
    meta=dfc.get("meta",{})
    meta["generated_at"]=time.time()
    meta["dfc_code_enhanced"]=True
    return {"tables":tables,"plots":plots,"meta":meta}

# ------------------------------------------------------------------ IUPR payload
def _build_iupr_payload(files: List[Path])->Dict[str,Any]:
    if compute_iupr_plotly is None:
        return {"tables":{"Final Ratios":[],"Min/Max":[]},
                "plots":{},
                "meta":{"generated_at":int(time.time()),
                        "iupr_error":"compute_iupr_plotly not available"}}
    try:
        out=compute_iupr_plotly(files)
    except Exception as e:
        return {"tables":{"Final Ratios":[],"Min/Max":[]},
                "plots":{},
                "meta":{"generated_at":int(time.time()),
                        "iupr_error":f"{e.__class__.__name__}: {e}"}}
    if "tables" in out or "plots" in out:
        norm_plots={}
        for k,v in (out.get("plots") or {}).items():
            pj=v.get("plotly_json") if isinstance(v,dict) else v
            norm_plots[k]={"type":"plotly","plotly_json":_ensure_plotly_json(pj)}
        return {"tables":out.get("tables",{}),
                "plots":norm_plots,
                "meta":out.get("meta",{"generated_at":int(time.time())})}
    tables={"Final Ratios":out.get("final",[]),
            "Min/Max":out.get("min_max",[])}
    plots={}
    for i,p in enumerate(out.get("plots",[]) or []):
        name=p.get("name",f"IUPR Plot {i+1}")
        pj=p.get("plotly_json")
        plots[name]={"type":"plotly","plotly_json":_ensure_plotly_json(pj)}
    return {"tables":tables,"plots":plots,"meta":{"generated_at":int(time.time())}}

# ------------------------------------------------------------------ CC/SL payload
def _build_ccsl_payload(files: List[Path])->Dict[str,Any]:
    if analyze_cc_sl_behavior_mdf is not None:
        try:
            res=analyze_cc_sl_behavior_mdf(files)
            tables={"CC/SL Overshoot":res.get("summary",[])}
            if "legacy" in res:
                tables["CC/SL Legacy Errors"]=res.get("legacy",[])
            elif "legacy_summary" in res:
                tables["CC/SL Legacy Errors"]=res.get("legacy_summary",[])
            plots={}
            for it in (res.get("plots") or []):
                name=it.get("name","Overshoot")
                pj=it.get("plotly_json")
                plots[name]={"type":"plotly","plotly_json":_ensure_plotly_json(pj)}
            diag=res.get("diag",{})
            meta={"generated_at":int(time.time()),
                  "ccsl_diag":diag,
                  "ccsl_config":res.get("config",{})}
            if not plots and not any(r.get("status")=="ok" for r in tables["CC/SL Overshoot"]):
                meta["ccsl_reason"]="No valid overshoot data; verify required channels / flags."
            agg=build_mode_aggregated_ccsl(files, ENV_OVERSHOOT_MARGIN)
            if agg["plots"]: plots.update(agg["plots"])
            for k,v in agg["tables"].items(): tables[k]=v
            return {"tables":tables,"plots":plots,"meta":meta}
        except Exception as e:
            if DEBUG_MODE: print("[WARN] custom_cc_sl analyze failed; fallback:", e)
    overshoot=analyze_ccsl_overshoot(files)
    legacy=report_ccsl(files)
    tables={"CC/SL Overshoot":overshoot.get("summary",[])}
    if legacy.get("summary"):
        tables["CC/SL Legacy Errors"]=legacy["summary"]
    plots={}
    for p in (overshoot.get("plots") or []):
        name=p.get("name","Overshoot")
        pj=p.get("plotly_json")
        plots[name]={"type":"plotly","plotly_json":_ensure_plotly_json(pj)}
    agg=build_mode_aggregated_ccsl(files, ENV_OVERSHOOT_MARGIN)
    if agg["plots"]: plots.update(agg["plots"])
    for k,v in agg["tables"].items(): tables[k]=v
    meta={"generated_at":int(time.time()),
          "ccsl_diag":overshoot.get("diag",{}),
          "ccsl_config":overshoot.get("config",{}),
          "files":[f.name for f in files]}
    if not any("Cruise Control Overshoot" in k for k in plots.keys()) and not any(r.get("status")=="ok" for r in overshoot.get("summary",[])):
        meta["ccsl_reason"]="No valid overshoot data; check statuses or enable fallback."
    return {"tables":tables,"plots":plots,"meta":meta}

# ------------------------------------------------------------------ Fuel Consumption payload
def _build_fuel_payload(files: List[Path], high_quality: bool = False)->Dict[str,Any]:
    if compute_fuel_plotly is None:
        return {"tables":{"Fuel Summary":[],"Operating Point Analysis":[],"Fuel Channels Found":[]},
                "plots":{},
                "meta":{"generated_at":int(time.time()),
                        "fuel_error":"compute_fuel_plotly not available"}}
    try:
        out=compute_fuel_plotly([str(f) for f in files], high_quality=high_quality)
    except Exception as e:
        if DEBUG_MODE: traceback.print_exc()
        return {"tables":{"Fuel Summary":[],"Operating Point Analysis":[],"Fuel Channels Found":[]},
                "plots":{},
                "meta":{"generated_at":int(time.time()),
                        "fuel_error":f"{e.__class__.__name__}: {e}"}}
    tables=out.get("tables",{})
    plots={}
    for k,v in (out.get("plots") or {}).items():
        pj=v.get("plotly_json") if isinstance(v,dict) else v
        plots[k]={"type":"plotly","plotly_json":_ensure_plotly_json(pj)}
    return {"tables":tables,
            "plots":plots,
            "meta":out.get("meta",{"generated_at":int(time.time())})}

# ==================================================================
# ================== INTEGRATED ANALYSIS MODULES ===================
# ==================================================================

# ------------------------------------------------------------------
# -------------------- Integrated: Gear Hunt -----------------------
# ------------------------------------------------------------------
# Note: Gear hunting analysis is now handled by custom_gear.py module
# This function is kept for backward compatibility but delegates to the advanced module
def analyze_gear_hunting_legacy(files: List[Path]) -> Dict[str, Any]:
    """
    Legacy gear hunting function - now delegates to custom_gear module.
    """
    try:
        from custom_gear import analyze_gear_hunting
        return analyze_gear_hunting(files)
    except ImportError:
        return {
            "tables": {"Hunting Events": [], "File Summary": [], "Statistics": []},
            "plots": {},
            "meta": {"ok": False, "error": "custom_gear module not available"}
        }

# ------------------------------------------------------------------
# ----------------- Integrated: Empirical Map --------------------
# ------------------------------------------------------------------
def analyze_empirical_map(files: List[Path],
                          outdir: Path,
                          python_exe: str = None,
                          min_samples: int = None,
                          timeout_sec: int = 60*60) -> Dict[str, Any]:
    """
    Run custom_map.py. Prefers direct in-process call; falls back to subprocess.
    """
    # ----------------- robust map invocation: prefer in-process then fallback to subprocess -----------------
    # collect files for processing from UPLOAD_DIR (expand directory contents)
    files_list = [str(p) for p in UPLOAD_DIR.iterdir() if p.suffix.lower() in ('.mf4', '.mdf', '.csv', '.xlsx', '.xls')]
    if not files_list:
        return {
            "tables": {"Map Summary": []},
            "plots": {},
            "meta": {
                "ok": False,
                "error": "no_input_data",
                "file_count": 0,
                "outdir": str(outdir),
                "problems": [{"file": str(UPLOAD_DIR), "error": "not_found"}]
            }
        }

    # --- Prefer in-process compute if compute_map_plotly exists ---
    if compute_map_plotly:
        try:
            kwargs = {
                "rpm_bins": payload.get("rpm_bins"),
                "tq_bins": payload.get("tq_bins"),
                "min_samples_per_bin": int(min_samples) if min_samples else MIN_SAMPLES_PER_BIN,
                "interp_method": payload.get("interp_method"),
                "smoothing": payload.get("smoothing"),
                "contour_levels": payload.get("contour_levels"),
                "enable_contours": payload.get("enable_contours", True),
                "enable_surface": payload.get("enable_surface", True),
                "overrides": payload.get("overrides"),
                "preset": payload.get("preset"),
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            app.logger.info("Calling compute_map_plotly in-process with %d files (kwargs=%s)", len(files_list), kwargs)
            result = compute_map_plotly(files_list, **kwargs)

            if not isinstance(result, dict):
                raise RuntimeError("compute_map_plotly returned non-dict result")

            # save canonical JSON output for debugging and frontend to load later if needed
            outdir.mkdir(parents=True, exist_ok=True)
            out_json_path = outdir / "map_output.json"
            try:
                # Convert result into JSON-serializable form before writing
                serializable_result = make_json_serializable(result)
                with open(out_json_path, "w", encoding="utf-8") as fh:
                    json.dump(serializable_result, fh, indent=2, ensure_ascii=False)
                app.logger.info("Wrote map_output.json to %s", out_json_path)
            except Exception as e:
                # If conversion still fails for some unexpected item, write a safe repr and continue
                app.logger.exception("Failed to write map_output.json after serialization step; writing fallback text.")
                with open(out_json_path, "w", encoding="utf-8") as fh:
                    # write fallback debug output to file
                    fh.write(json.dumps({"error": "serialization_failed", "message": str(e), "result_preview": str(result)[:2000]}, ensure_ascii=False, indent=2))
            
            serializable_result = make_json_serializable(result)
            return {
                "tables": serializable_result.get("tables", {}),
                "plots": serializable_result.get("plots", {}),
                "meta": { **serializable_result.get("meta", {}), "ok": True, "file_count": len(files_list), "outdir": str(outdir), "outfile": str(out_json_path) }
            }

        except Exception as e:
            tb = traceback.format_exc()
            app.logger.error("compute_map_plotly in-process failed: %s\n%s", str(e), tb)
            return {
                "tables": {"Map Summary": []},
                "plots": {},
                "meta": {
                    "ok": False,
                    "error": "compute_map_plotly_failed",
                    "detail": str(e),
                    "traceback": tb,
                    "file_count": len(files_list),
                    "outdir": str(outdir)
                }
            }

    # --- Subprocess fallback (explicit file list, same interpreter) ---
    py = python_exe or os.getenv("PYTHON") or sys.executable or "python"
    cmd = [py, str(APP_DIR / "custom_map.py"), "--files"] + files_list + ["--output", "plotly_json"]
    if min_samples is not None:
        cmd += ["--min-samples", str(int(min_samples))]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_sec)
    except Exception as e:
        tb = traceback.format_exc()
        return {
            "tables": {"Map Summary": []},
            "plots": {},
            "meta": {"ok": False, "error": "subprocess_failed_to_start", "detail": str(e), "traceback": tb, "outdir": str(outdir)}
        }

    if proc.returncode != 0:
        return {
            "tables": {"Map Summary": []},
            "plots": {},
            "meta": {
                "ok": False,
                "error": "map_subprocess_failed",
                "returncode": proc.returncode,
                "stdout": proc.stdout[:4000],
                "stderr": proc.stderr[:4000],
                "file_count": len(files_list),
                "outdir": str(outdir)
            }
        }

    # successful subprocess run — attempt to locate JSON output
    expected_files = [
        outdir / "map_output.json",
        outdir / "map_output-v1.json",
        outdir / "map.json",
        outdir / "maps.json",
        outdir / "test_map.json",
        outdir / "map_output-plotly.json"
    ]
    out_json = None
    for ef in expected_files:
        if ef.exists():
            out_json = ef
            break

    if out_json is None:
        js = list(outdir.glob("*.json"))
        out_json = js[0] if js else None

    if out_json:
        try:
            with open(out_json, 'r', encoding='utf-8') as fh:
                res = json.load(fh)
            # Ensure final result is serializable
            res_serializable = make_json_serializable(res)
            return {"tables": res_serializable.get("tables", {}), "plots": res_serializable.get("plots", {}), "meta": {**res_serializable.get("meta",{}), "ok": True, "file_count": len(files_list), "outdir": str(outdir), "outfile": str(out_json)}}
        except Exception as e:
            return {"tables": {"Map Summary": []}, "plots": {}, "meta": {"ok": True, "warning": "failed_to_read_json", "detail": str(e), "stdout": proc.stdout, "outdir": str(outdir)}}
    else:
        # no json found — return stdout so the caller can inspect script output
        return {"tables": {"Map Summary": []}, "plots": {}, "meta": {"ok": True, "stdout": proc.stdout, "outdir": str(outdir)}}
    # ---------------------------------------------------------------------------------------------------------

@app.post("/api/compute_map")
def api_compute_map():
    """Generate empirical maps from uploaded MDF files with progress tracking."""
    progress_updates = []
    
    def progress_callback(stage, progress, message):
        progress_updates.append({"stage": stage, "progress": progress, "message": message})
        app.logger.info(f"Progress: {stage} - {progress}% - {message}")

    try:
        data = request.get_json(force=True)
        files = data.get("files") or []
        paths = []
        for f in files:
            p = Path(f)
            # Handle different path formats from frontend
            if not p.is_absolute():
                # Remove 'uploads/' prefix if present to avoid double path
                f_clean = str(f).replace('uploads/', '').replace('\\', '/').lstrip('/')
                p = (UPLOAD_DIR / f_clean).resolve()
            else:
                p = Path(p).resolve()
            
            # Verify file exists
            if not p.exists():
                app.logger.warning(f"File not found: {f} -> {p}")
                # Try alternative path resolution
                alt_path = UPLOAD_DIR / Path(f).name
                if alt_path.exists():
                    p = alt_path.resolve()
                    app.logger.info(f"Using alternative path: {p}")
                else:
                    app.logger.error(f"File not found: {f} (resolved to {p})")
            
            paths.append(str(p))

        min_samples = data.get("min_samples_per_bin", data.get("min_samples", None))
        fmt = data.get("output_format", "plotly_json")

        kwargs = {
            "rpm_bins": data.get("rpm_bins"),
            "tq_bins": data.get("tq_bins"),
            "min_samples_per_bin": int(min_samples) if min_samples is not None else None,
            "output_format": fmt,
            "preset": data.get("preset"),
            "overrides": data.get("overrides"),
            "interp_method": data.get("interp_method"),
            "smoothing": data.get("smoothing"),
            "map_type": data.get("map_type"),
            "contour_levels": data.get("contour_levels"),
            "enable_surface": data.get("enable_surface", True),
            "enable_contours": data.get("enable_contours", True),
            "progress_callback": progress_callback,
        }

        if compute_map_plotly is None:
            return json_error("Map module not available", 503)

        # Verify at least one file exists before processing
        existing_paths = [p for p in paths if Path(p).exists()]
        if not existing_paths:
            return json_error(f"No valid files found. Checked: {paths}", 400)
        
        app.logger.info(f"Processing {len(existing_paths)} file(s): {[Path(p).name for p in existing_paths]}")
        res = compute_map_plotly(existing_paths, **{k: v for k, v in kwargs.items() if v is not None})
        
        # Normalize plots structure - ensure consistent format for frontend
        if res.get("plots"):
            normalized_plots = {}
            for k, v in res["plots"].items():
                if isinstance(v, dict):
                    if "plotly_json" in v:
                        # Ensure plotly_json is a string
                        pj = v["plotly_json"]
                        if isinstance(pj, str):
                            normalized_plots[k] = {"type": "plotly", "plotly_json": pj}
                        else:
                            # Convert to string if it's already parsed JSON
                            normalized_plots[k] = {"type": "plotly", "plotly_json": _ensure_plotly_json(pj)}
                    elif "type" in v and v.get("type") == "plotly":
                        # Already normalized
                        normalized_plots[k] = v
                    else:
                        # Unknown structure, try to convert
                        normalized_plots[k] = {"type": "plotly", "plotly_json": _ensure_plotly_json(v)}
                elif isinstance(v, str):
                    # Direct JSON string
                    normalized_plots[k] = {"type": "plotly", "plotly_json": v}
                else:
                    # Try to serialize
                    normalized_plots[k] = {"type": "plotly", "plotly_json": _ensure_plotly_json(v)}
            res["plots"] = normalized_plots
            app.logger.info(f"✅ Normalized {len(normalized_plots)} plots: {list(normalized_plots.keys())[:5]}...")
        
        # Transform progress updates to frontend-expected format
        progress_stages = {
            "file_loading": {"progress": 0, "status": "pending", "message": "Waiting..."},
            "signal_mapping": {"progress": 0, "status": "pending", "message": "Waiting..."},
            "data_validation": {"progress": 0, "status": "pending", "message": "Waiting..."},
            "physics_calculations": {"progress": 0, "status": "pending", "message": "Waiting..."},
            "map_generation": {"progress": 0, "status": "pending", "message": "Waiting..."},
            "optimization": {"progress": 0, "status": "pending", "message": "Waiting..."},
            "visualization": {"progress": 0, "status": "pending", "message": "Waiting..."}
        }
        
        # Update stages from progress updates
        overall_progress = 0
        for update in progress_updates:
            stage = update.get("stage", "")
            progress_val = update.get("progress", 0)
            message = update.get("message", "")
            
            # Map backend stages to frontend stages
            if stage in progress_stages:
                progress_stages[stage] = {
                    "progress": progress_val,
                    "status": "processing" if progress_val < 100 else "complete",
                    "message": message
                }
                overall_progress = max(overall_progress, progress_val)
        
        if "meta" in res and isinstance(res["meta"], dict):
            res["meta"]["progress"] = {
                "stages": progress_stages,
                "overall_progress": overall_progress,
                "updates": progress_updates  # Keep raw updates for debugging
            }
        else:
            res["meta"] = {
                "progress": {
                    "stages": progress_stages,
                    "overall_progress": overall_progress,
                    "updates": progress_updates
                }
            }

        try:
            run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
            outdir = MAPS_DIR / run_id
            outdir.mkdir(parents=True, exist_ok=True)

            try:
                serializable = make_json_serializable(res)
            except Exception:
                try:
                    serializable = json.loads(json.dumps(res, default=str))
                except Exception:
                    serializable = {"tables": res.get("tables", {}), "plots": res.get("plots", {}), "meta": res.get("meta", {})}

            if isinstance(serializable, dict) and 'table' not in serializable:
                serializable['table'] = serializable.get('tables', {}).get('Map Summary', [])

            with open(outdir / "map_output.json", "w", encoding="utf-8") as fh:
                json.dump(serializable, fh, indent=2, ensure_ascii=False)
        except Exception as e:
            app.logger.warning(f"Failed to persist map_output.json: {e}", exc_info=True)
            if 'meta' not in res:
                res['meta'] = {}
            res['meta']['persistence_error'] = str(e)

        # Use safe_jsonify to properly handle NaN values in JSON serialization
        return safe_jsonify(res)
    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error("Error in api_compute_map: %s\n%s", str(e), tb)
        return json_error(f"Map generation failed: {e}", 500)

@app.post("/api/export_map")
def api_export_empirical_map():
    """Export empirical map to various formats (MATLAB .mat, CSV, Excel)."""
    try:
        data = request.get_json(force=True)
        export_format = data.get("format", "csv").lower()
        map_name = data.get("map_name", "map")
        map_data = data.get("map_data")  # Full map dictionary from frontend
        
        if not map_data:
            return json_error("map_data is required", 400)
        
        # Get export functions from custom_map module
        if compute_map is None:
            return json_error("Map module not available", 503)
        
        from custom_map import export_map_to_matlab, export_map_to_csv, export_map_to_excel
        
        # Create temporary output file
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        output_dir = MAPS_DIR / f"exports_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if export_format == "matlab" or export_format == "mat":
            output_path = output_dir / f"{map_name}.mat"
            try:
                exported_path = export_map_to_matlab(map_data, output_path, map_name)
                return jsonify({
                    "ok": True,
                    "format": "matlab",
                    "file_path": exported_path,
                    "download_url": f"/api/download_map?path={exported_path}"
                })
            except Exception as e:
                return json_error(f"MATLAB export failed: {e}", 500)
        
        elif export_format == "csv":
            output_path = output_dir / f"{map_name}.csv"
            include_stats = data.get("include_stats", True)
            try:
                exported_path = export_map_to_csv(map_data, output_path, include_stats)
                return jsonify({
                    "ok": True,
                    "format": "csv",
                    "file_path": exported_path,
                    "download_url": f"/api/download_map?path={exported_path}"
                })
            except Exception as e:
                return json_error(f"CSV export failed: {e}", 500)
        
        elif export_format == "excel" or export_format == "xlsx":
            output_path = output_dir / f"{map_name}.xlsx"
            try:
                exported_path = export_map_to_excel(map_data, output_path, map_name)
                return jsonify({
                    "ok": True,
                    "format": "excel",
                    "file_path": exported_path,
                    "download_url": f"/api/download_map?path={exported_path}"
                })
            except Exception as e:
                return json_error(f"Excel export failed: {e}", 500)
        
        else:
            return json_error(f"Unsupported export format: {export_format}. Supported: matlab, csv, excel", 400)
    
    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error("Error in api_export_empirical_map: %s\n%s", str(e), tb)
        return json_error(f"Export failed: {e}", 500)

@app.get("/api/download_map")
def api_download_map():
    """Download exported map file."""
    try:
        file_path = request.args.get("path")
        if not file_path:
            return json_error("path parameter required", 400)
        
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return json_error("File not found", 404)
        
        # Security: ensure path is within allowed directory
        if not str(path.resolve()).startswith(str(MAPS_DIR.resolve())):
            return json_error("Access denied", 403)
        
        return send_file(
            path,
            as_attachment=True,
            download_name=path.name,
            mimetype='application/octet-stream'
        )
    except Exception as e:
        app.logger.error("Error in api_download_map: %s", e)
        return json_error(f"Download failed: {e}", 500)

# ------------------------------------------------------------------
def pick_display_alias(canonical: str, aliases: List[str]) -> str:
    """
    Pick the best compact/display alias for a canonical channel name.
    Heuristics:
     - Prefer the alias identical to canonical but without MDF suffixes (strip .EA/.E8/etc)
     - Prefer shorter names
     - Prefer names that don't contain 'OBD', '#', or '['
     - Return canonical if nothing better found
    """
    import re
    if not aliases:
        return canonical
    # prefer alias equal to canonical cleaned
    def score(a):
        s = 0
        if 'obd' in a.lower(): s += 5
        if '#' in a: s += 5
        if '[' in a or ']' in a: s += 3
        # penalize having lots of dots
        s += a.count('.') * 1
        # shorter is better
        s += len(a) / 100.0
        return s

    # normalize: remove known MDF suffix code groups like ".EA", ".E8", ".EE", ".E8.001"
    def strip_suffixes(a):
        parts = a.split('.')
        # drop trailing parts that look hex-like or short token codes
        while len(parts) > 1 and re.fullmatch(r'[A-Za-z0-9]{1,8}', parts[-1]):
            parts = parts[:-1]
        return '.'.join(parts)

    candidates = []
    for a in aliases:
        try:
            candidates.append(strip_suffixes(a))
        except Exception:
            candidates.append(a)

    # dedupe preserving order
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    # rank candidates by simple heuristic score, choose lowest score
    best = min(uniq, key=lambda x: score(x))
    # if best equals canonical or is empty, fallback to canonical simplified
    if not best:
        return canonical
    # final cleanup: collapse multiple spaces and trim
    return ' '.join(best.replace('_', ' ').split()).strip()

# ------------------------------------------------------------------
@app.get("/api/list_display_channels")
def api_list_display_channels():
    """
    Return JSON with a list of {display, canonical, aliases[]} for the file specified
    Query params:
      - fname: relative path (e.g. uploads/xxx.mf4) OR full path
    If fname omitted, returns aggregated channels for all uploaded files (may be large).
    """
    fname = request.args.get("fname")
    files = []
    if fname:
        p = Path(fname)
        if not p.exists():
            p = UPLOAD_DIR / Path(fname).name
        if not p.exists():
            return safe_jsonify({"ok": False, "error": "file_not_found", "path": str(p)}, 404)
        files = [p]
    else:
        # aggregate all uploads
        files = list(UPLOAD_DIR.glob("*.mf4"))

    results = []
    # build a per-file canonical list
    for f in files:
        try:
            # get canonical names (raw)
            with open_mdf(f) as m:
                if not m:
                    continue
                try:
                    canonical_names = m.channels()
                except Exception:
                    try:
                        canonical_names = list(getattr(m._o, "channels_db", {}).keys())
                    except Exception:
                        canonical_names = []
        except Exception:
            # fallback to cached expanded list and reverse engineer canonical if possible
            canonical_names = []
            try:
                expanded = list_channels(f)
                # take unique entries that look canonical (heuristic: contain capital letters or spaces)
                for e in expanded:
                    if e not in canonical_names:
                        canonical_names.append(e)
            except Exception:
                canonical_names = []

        # for each canonical, build aliases and pick a display alias
        for can in canonical_names:
            try:
                aliases = normalize_channel_aliases(can)
            except Exception:
                aliases = [can]
            display = pick_display_alias(can, aliases)
            results.append({"display": display, "canonical": can, "aliases": aliases})
    return safe_jsonify({"ok": True, "file": str(files[0]) if fname else None, "channels": results})

# ----------------------------
# Empirical Map API endpoints
# ----------------------------
# ---------------- Robust api_map_runs ----------------
@app.route('/api/map_runs', methods=['GET'])
def api_map_runs():
    """List available map runs (folders that contain map_output.json). Defensive to FS errors."""
    try:
        # ensure maps dir exists (idempotent)
        if not MAPS_DIR.exists():
            try:
                MAPS_DIR.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                app.logger.exception("Failed to create MAPS_DIR: %s", e)
                return safe_jsonify({"ok": False, "runs": [], "error": "maps_dir_create_failed", "detail": str(e)}, status=200)

        runs = []
        # defensively list entries; if listing fails, return empty list rather than 500
        try:
            entries = []
            for p in MAPS_DIR.iterdir():
                try:
                    mtime = p.stat().st_mtime if p.exists() else 0
                    entries.append((p, mtime))
                except Exception:
                    app.logger.exception("Skipping unreadable entry in MAPS_DIR: %s", str(p))
                    continue
            entries.sort(key=lambda t: t[1], reverse=True)
            entries = [t[0] for t in entries]
        except Exception as e:
            app.logger.exception("Failed to list MAPS_DIR entries: %s", e)
            return safe_jsonify({"ok": True, "runs": []}, status=200)

        for d in entries:
            try:
                if not d.is_dir():
                    continue
                path = d / 'map_output.json'
                if not path.exists():
                    js = list(d.glob("*.json"))
                    path = js[0] if js else None
                if path and path.exists():
                    try:
                        stat = path.stat()
                        runs.append({'run_id': d.name, 'path': str(path), 'mtime': float(stat.st_mtime)})
                    except Exception as e:
                        app.logger.exception("Failed to stat map json for run %s: %s", d.name, e)
                        runs.append({'run_id': d.name, 'path': str(path), 'mtime': None})
            except Exception as e:
                app.logger.exception("Skipping run folder %s due to error: %s", str(d), e)
                continue

        return safe_jsonify({"ok": True, "runs": runs}, status=200)

    except Exception as e:
        app.logger.exception("Unexpected error in api_map_runs: %s", e)
        return safe_jsonify({"ok": True, "runs": [], "warning": "map_runs_failed", "detail": str(e)}, status=200)

@app.route('/api/map_vars', methods=['GET'])
def api_map_vars():
    """Return numeric variables present in a run's map_output.json table.
       Also accept optional 'template' param to return template-required fields.
    """
    run = request.args.get('run')
    template = request.args.get('template')  # optional: 'ci'|'si'|'electric'|'generic'
    if not run:
        return safe_jsonify(ok=False, error="missing run param"), 400
    
    path = MAPS_DIR / run / 'map_output.json'
    if not path.exists():
        return safe_jsonify(ok=False, error="run not found"), 404
        
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    table, src = _load_primary_table_from_map_json(data)
    if table is None:
        # fallback: check traces for x/y/z triples
        for t in data.get('traces', []):
            if all(k in t for k in ('x','y','z')):
                return safe_jsonify(ok=True, vars=['x','y','z'], hint='traces_raw'), 200
        return safe_jsonify(ok=False, error='no table or traces found in map JSON'), 500

    # Normalize to list-of-dicts or dict-of-arrays and discover numeric columns
    numeric_cols = []
    if np:
        try:
            if isinstance(table, dict) and table:
                cols = list(table.keys())
                sample_len = min(1000, len(next(iter(table.values()))))
                for c in cols:
                    try:
                        arr = np.array(table[c][:sample_len], dtype=float)
                        if not np.isnan(arr).all():
                           numeric_cols.append(c)
                    except (ValueError, TypeError, IndexError):
                        pass
            elif isinstance(table, list) and len(table) > 0 and isinstance(table[0], dict):
                cols = list(table[0].keys())
                sample = table[:1000]
                for c in cols:
                    try:
                        vals = [r.get(c, None) for r in sample]
                        arr = np.array([float(v) for v in vals if v is not None], dtype=float)
                        if arr.size > 0:
                            numeric_cols.append(c)
                    except (ValueError, TypeError):
                        pass
            else:
                 return safe_jsonify(ok=False, error='unexpected table format'), 500
        except Exception as e:
            return safe_jsonify(ok=False, error='error scanning table: ' + str(e)), 500

    # If template param provided, also return required/optional fields per MathWorks templates
    templates = {
        'ci': {
            'required': ['Torque','EngSpd'],  # Core required signals (per MathWorks CI docs)
            'optional': ['FuelMassCmd','AirMassFlwRate','FuelMassFlwRate','ExhTemp','BSFC',
                         'HCMassFlwRate','COMassFlwRate','NOxMassFlwRate','CO2MassFlwRate','PMMassFlwRate']
        },
        'si': {
            'required': ['Torque','EngSpd'],
            'optional': ['AirMassFlwRate','FuelMassFlwRate','ExhTemp','BSFC','HCMassFlwRate',
                         'COMassFlwRate','NOxMassFlwRate','CO2MassFlwRate','PMMassFlwRate']
        },
        'electric': {
            'required': ['motor_speed','motor_torque'],
            'optional': ['motor_efficiency','electrical_power','mechanical_power']
        },
        'generic': {
            'required': [],
            'optional': []
        }
    }

    response = {'ok': True, 'vars': numeric_cols}
    if template and template.lower() in templates:
        tdef = templates[template.lower()]
        # report which of the required/optional are present
        present_required = [s for s in tdef['required'] if s in numeric_cols]
        missing_required = [s for s in tdef['required'] if s not in numeric_cols]
        response.update({'template': template.lower(),
                         'required_present': present_required,
                         'required_missing': missing_required,
                         'optional': [s for s in tdef['optional'] if s in numeric_cols]})
    return safe_jsonify(response), 200

@app.route('/api/get_map_data', methods=['POST'])
def api_get_map_data():
    """
    Template-aware get_map_data:
    POST JSON:
      { "run": "<run_id>", "x": "colX", "y": "colY", "z": "colZ",
        "bins_x": 40, "bins_y": 40, "interp": "none|linear|nearest",
        "template": "ci|si|electric|generic" (optional) }
    Returns:
      { ok: true, x: [...], y: [...], z: [[...]], counts: [[...]] }
    """
    if np is None:
        return safe_jsonify(ok=False, error="numpy is not available on the server"), 503

    payload = request.get_json(force=True)
    run = payload.get('run'); xcol = payload.get('x'); ycol = payload.get('y'); zcol = payload.get('z')
    bins_x = int(payload.get('bins_x', 40)); bins_y = int(payload.get('bins_y', 40))
    interp = payload.get('interp', 'none')
    template = (payload.get('template') or 'generic').lower()

    # Validate inputs
    if not run:
        return safe_jsonify(ok=False, error='missing run'), 400
    if not xcol or not ycol or not zcol:
        return safe_jsonify(ok=False, error='x/y/z must be specified'), 400

    path = MAPS_DIR / run / 'map_output.json'
    if not path.exists():
        return safe_jsonify(ok=False, error="run not found"), 404
        
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    table, src = _load_primary_table_from_map_json(data)
    if table is None:
        return safe_jsonify(ok=False, error="no table in map JSON"), 500

    # Template-specific required signal check (mirror api_map_vars templates)
    templates = {
        'ci': ['Torque','EngSpd'],
        'si': ['Torque','EngSpd'],
        'electric': ['motor_speed','motor_torque'],
        'generic': []
    }
    required = templates.get(template, [])

    # quick discovery of available columns
    available_cols = []
    if isinstance(table, dict):
        available_cols = list(table.keys())
    elif isinstance(table, list) and len(table) > 0 and isinstance(table[0], dict):
        available_cols = list(table[0].keys())
    else:
        return safe_jsonify(ok=False, error='unexpected table structure'), 500

    missing_required = [r for r in required if r not in available_cols]
    if missing_required:
        return safe_jsonify(ok=False, error='missing required signals for template', missing=missing_required), 400

    # Extract arrays
    try:
        if isinstance(table, dict):
            x = np.array(table[xcol], dtype=float)
            y = np.array(table[ycol], dtype=float)
            z = np.array(table[zcol], dtype=float)
        else:
            x = np.array([r.get(xcol, np.nan) for r in table], dtype=float)
            y = np.array([r.get(ycol, np.nan) for r in table], dtype=float)
            z = np.array([r.get(zcol, np.nan) for r in table], dtype=float)
    except Exception as e:
        return safe_jsonify(ok=False, error=f"error extracting columns: {e}"), 500

    # Remove NaNs and ensure we have enough samples
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[valid], y[valid], z[valid]
    if len(x) < 50:
        return safe_jsonify(ok=False, error='insufficient valid rows after stripping NaNs', rows=len(x)), 400

    # Binning & grid generation
    xi_bins = np.linspace(np.nanmin(x), np.nanmax(x), bins_x + 1)
    yi_bins = np.linspace(np.nanmin(y), np.nanmax(y), bins_y + 1)
    xi_centers = (xi_bins[:-1] + xi_bins[1:]) / 2.0
    yi_centers = (yi_bins[:-1] + yi_bins[1:]) / 2.0

    zgrid = np.full((bins_y, bins_x), np.nan)
    counts = np.zeros((bins_y, bins_x), dtype=int)
    ix = np.digitize(x, xi_bins) - 1
    iy = np.digitize(y, yi_bins) - 1
    valid_idx = (ix >= 0) & (ix < bins_x) & (iy >= 0) & (iy < bins_y)
    
    # Use a temporary grid for summing to correctly calculate the mean
    z_sum = np.zeros((bins_y, bins_x))
    
    for i in range(len(x)):
        if valid_idx[i]:
            row, col = iy[i], ix[i]
            z_sum[row, col] += z[i]
            counts[row, col] += 1
            
    # Calculate the mean, avoiding division by zero
    non_zero_counts = counts > 0
    zgrid[non_zero_counts] = z_sum[non_zero_counts] / counts[non_zero_counts]

    # Optional interpolation
    if interp in ('linear', 'nearest') and griddata:
        Xc, Yc = np.meshgrid(xi_centers, yi_centers)
        known = ~np.isnan(zgrid)
        if np.sum(known) >= 3:
            pts = np.column_stack((Xc[known].ravel(), Yc[known].ravel()))
            vals = zgrid[known].ravel()
            interp_pts = np.column_stack((Xc.ravel(), Yc.ravel()))
            try:
                filled = griddata(pts, vals, interp_pts, method=interp)
                zgrid = filled.reshape(Xc.shape)
            except Exception:
                pass

    return safe_jsonify(ok=True, x=xi_centers, y=yi_centers, z=zgrid, counts=counts)

# ------------------------------------------------------------------ Routes / meta
@app.get("/")
def root():
    idx=APP_DIR/"frontend.html"
    if idx.exists(): return send_from_directory(str(APP_DIR),"frontend.html")
    return "<h3>frontend.html missing</h3>",404

@app.get("/favicon.ico")
def favicon():
    png_bytes=b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82'
    return Response(png_bytes,mimetype='image/png')

@app.get("/api/ping")
def ping(): return safe_jsonify({"pong":True})

@app.get("/api/version")
def version():
    return safe_jsonify({
        "version":"1.0.36-robust-alias",
        "asammdf":bool(MDF),
        "mdfreader":_HAVE_MDFREADER,
        "active_files":len(ACTIVE_FILES),
        "dfc_mapping_entries":0,
        "debug":DEBUG_MODE,
        "overshoot_env":{
            "margin_kmh":ENV_OVERSHOOT_MARGIN,
            "flag_value":ENV_OVERSHOOT_FLAG_VALUE,
            "fallback_no_flags":ENV_OVERSHOOT_FALLBACK_NO_FLAGS
        }
    })

# ---- DELETE ALL UPLOADS ----
@app.post("/api/delete_all")
def api_delete_all():
    try:
        UPLOAD_DIR.mkdir(exist_ok=True)
        exts={".mf4",".mdf",".mdf4",".csv",".xlsx",".xls",".parquet"}
        deleted=0
        for p in UPLOAD_DIR.glob("**/*"):
            if p.is_file() and (p.suffix.lower() in exts or p.name.lower() in {"merged.mf4","merged.mdf"}):
                try: p.unlink(); deleted+=1
                except: pass
        return jsonify({"ok":True,"deleted":deleted})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}),500

@app.get("/api/files")
def api_files():
    files=[{"name":p.name,"size":p.stat().st_size,"path":str(p.resolve())} for p in list_mdf_files(UPLOAD_DIR)]
    return safe_jsonify({"files":files,"count":len(files)})

@app.post("/api/delete_file")
def api_delete_file():
    fn=request.form.get("filename")
    if not fn: return json_error("No filename",400)
    p=UPLOAD_DIR/Path(fn).name
    if not p.exists(): return json_error("Not found",404)
    try: p.unlink()
    except Exception as e: return json_error("Delete failed",500,exc=e)
    abs_p=str(p.resolve())
    if abs_p in ACTIVE_FILES: ACTIVE_FILES.remove(abs_p)
    return safe_jsonify({"ok":True,"deleted":fn})

@app.get("/api/active_files")
def api_active_files():
    return safe_jsonify({"active_files":ACTIVE_FILES,"count":len(ACTIVE_FILES)})

@app.post("/api/reset_files")
def reset_files():
    ACTIVE_FILES.clear(); CHANNELS_CACHE.clear(); SERIES_CACHE.clear()
    return safe_jsonify({"ok":True,"active_files":[]})

@app.post("/api/purge_all")
def api_purge_all():
    try:
        purge_all()
        return safe_jsonify({"ok":True,"message":"All files purged","active_files":[]})
    except Exception as e:
        return json_error("purge_failed",500,exc=e)

@app.get("/api/channels")
def api_channels():
    mode=(request.args.get("mode") or "union").lower()
    file_paths=ACTIVE_FILES if ACTIVE_FILES else [str(p.resolve()) for p in list_mdf_files(UPLOAD_DIR)]
    files=[Path(p) for p in file_paths if Path(p).exists()]
    if not files:
        return safe_jsonify({"channels":[], "count":0, "mode":mode})
    meta=discover_channels(files)
    if mode=="intersection":
        req=len(files)
        meta=[m for m in meta if m.get("present_count")==req]
    return safe_jsonify({"channels":meta,"count":len(meta),"mode":mode})

# ---------------- Robust smart_merge_upload ----------------
@app.post("/smart_merge_upload")
def smart_merge_upload():
    """
    Defensive file upload:
    - protects against partial writes and size overruns
    - compatible with older Python pathlib (no missing_ok reliance)
    - will not return 500 for individual file failures; reports which files were added
    """
    mode = (request.args.get("mode") or request.form.get("mode") or "replace").lower()
    if mode not in ("append", "replace"):
        mode = "replace"

    if mode == "replace":
        try:
            purge_mdf_files_only()
            ACTIVE_FILES.clear(); CHANNELS_CACHE.clear(); SERIES_CACHE.clear()
        except Exception as e:
            app.logger.exception("Error during replace purge in smart_merge_upload: %s", e)
            # continue accepting uploaded files

    uploaded = []
    try:
        # helper to write a file-like object safely
        def _write_file_stream(fobj, dest_path):
            tmp = dest_path.with_suffix(dest_path.suffix + ".part")
            total = 0
            with tmp.open("wb") as out:
                while True:
                    chunk = fobj.stream.read(UPLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                        out.close()
                        try:
                            if tmp.exists():
                                tmp.unlink()
                        except Exception:
                            pass
                        raise ValueError(f"{fobj.filename} exceeds {MAX_UPLOAD_SIZE_MB}MB")
                    out.write(chunk)
            # atomically move into place (use os.replace with string paths to be safe)
            try:
                if dest_path.exists():
                    try:
                        dest_path.unlink()
                    except Exception:
                        pass
                os.replace(str(tmp), str(dest_path))
            except Exception:
                # final fallback: try shutil.move
                try:
                    shutil.move(str(tmp), str(dest_path))
                except Exception as e:
                    # cleanup tmp if present
                    try:
                        if tmp.exists(): tmp.unlink()
                    except Exception:
                        pass
                    raise

        # Multi-file list (form field "files")
        if "files" in request.files:
            for f in request.files.getlist("files"):
                if not getattr(f, "filename", None):
                    continue
                try:
                    safe = _sanitize_filename(f.filename)
                    dest = (UPLOAD_DIR / safe).resolve()
                    _write_file_stream(f, dest)
                    uploaded.append(str(dest))
                except ValueError as ve:
                    # file too large - return client-friendly error for that file
                    app.logger.warning("Upload rejected for %s: %s", getattr(f, "filename", "<unknown>"), ve)
                    return json_error(str(ve), 400)
                except Exception as e:
                    app.logger.exception("Failed to save uploaded file %s: %s", getattr(f, "filename", "<unknown>"), e)
                    # skip this file but continue processing other files
                    continue

        # Single-file fallback (field "file")
        if "file" in request.files and not uploaded:
            f = request.files["file"]
            if getattr(f, "filename", None):
                try:
                    safe = _sanitize_filename(f.filename)
                    dest = (UPLOAD_DIR / safe).resolve()
                    _write_file_stream(f, dest)
                    uploaded.append(str(dest))
                except ValueError as ve:
                    app.logger.warning("Upload rejected for %s: %s", getattr(f, "filename", "<unknown>"), ve)
                    return json_error(str(ve), 400)
                except Exception as e:
                    app.logger.exception("Failed to save single file upload %s: %s", getattr(f, "filename", "<unknown>"), e)

    except Exception as e:
        app.logger.exception("save_error in smart_merge_upload: %s", e)
        return json_error(f"save_error: {e.__class__.__name__}", 400, exc=e)

    if not uploaded:
        return json_error("no_files_received", 400)

    # update ACTIVE_FILES conservatively
    try:
        if mode == "append":
            seen = set(ACTIVE_FILES)
            for p in uploaded:
                if p not in seen:
                    ACTIVE_FILES.append(p); seen.add(p)
        else:
            ACTIVE_FILES[:] = uploaded
    except Exception as e:
        app.logger.exception("Failed to update ACTIVE_FILES after upload: %s", e)

    # attempt channel discovery but don't fail the request on discovery errors
    try:
        files_objs = [Path(p) for p in ACTIVE_FILES if Path(p).exists()]
        channel_meta = discover_channels(files_objs) if files_objs else []
    except Exception as e:
        app.logger.exception("Failed to discover channels after upload: %s", e)
        channel_meta = []

    return safe_jsonify({"ok": True, "mode": mode,
                         "active_files": ACTIVE_FILES,
                         "added": uploaded,
                         "channels": channel_meta,
                         "channel_count": len(channel_meta)})

@app.get("/api/debug_signal_map")
def debug_signal_map():
    files=[Path(p) for p in ACTIVE_FILES if Path(p).exists()]
    ch_map={}
    for f in files:
        for ch in list_channels(f):
            ch_map.setdefault(ch,ch)
            ch_map.setdefault(_clean_name(ch),ch)
            ch_map.setdefault(ch.lower(),ch)
            ch_map.setdefault(_clean_name(ch).lower(),ch)
    sample=dict(list(ch_map.items())[:100])
    return safe_jsonify({"entries":len(ch_map),"sample":sample})

@app.get("/api/debug_ccsl")
def debug_ccsl():
    files=[Path(p) for p in ACTIVE_FILES if Path(p).exists()]
    out=[]
    for f in files:
        chs=list_channels(f)
        data={"file":f.name}
        for key,patterns in _CC_PATTERNS.items():
            data[key+"_match"]=_match_channel_any(chs,patterns)
        samples={}
        for field in ("actual_match","cruise_set_match","limiter_set_match"):
            chname=data.get(field)
            if chname:
                t,v,_=read_signal(f,chname)
                short=[]
                for i,(ti,vi) in enumerate(zip(t,v)):
                    if i>=5: break
                    if isinstance(vi,(int,float)):
                        short.append([round(float(ti),3), float(vi)])
                samples[field.replace("_match","_samples")]=short
        data["samples"]=samples
        out.append(data)
    return safe_jsonify({"debug_ccsl":out,
                         "env":{"margin_kmh":ENV_OVERSHOOT_MARGIN,
                                "flag_value":ENV_OVERSHOOT_FLAG_VALUE,
                                "fallback_no_flags":ENV_OVERSHOOT_FALLBACK_NO_FLAGS}})

@app.get("/api/debug_dfc")
def debug_dfc():
    files=[Path(p) for p in ACTIVE_FILES if Path(p).exists()]
    if not files: return safe_jsonify({"error":"no_files"})
    dfc=compute_dfc(files=[Path(p) for p in files], mapping_file=None,
                   channels=None, compress_runs=True, topn=10, include_plots=True,
                   enable_advanced_features=True)
    return safe_jsonify({
        "summary_len":len(dfc.get("summary",[]) or []),
        "plots_len":len(dfc.get("plots",[]) or []),
        "keys":list(dfc.keys()),
        "has_error":False,
        "mapping_applied":False
    })

@app.post("/api/ccsl_config")
def api_ccsl_config():
    global ENV_OVERSHOOT_FALLBACK_NO_FLAGS, ENV_OVERSHOOT_MARGIN, ENV_OVERSHOOT_FLAG_VALUE
    changed={}
    if "fallback_no_flags" in request.form:
        ENV_OVERSHOOT_FALLBACK_NO_FLAGS=request.form["fallback_no_flags"].lower() in ("1","true","yes","on")
        changed["fallback_no_flags"]=ENV_OVERSHOOT_FALLBACK_NO_FLAGS
    if "margin_kmh" in request.form:
        try:
            ENV_OVERSHOOT_MARGIN=float(request.form["margin_kmh"])
            changed["margin_kmh"]=ENV_OVERSHOOT_MARGIN
        except: pass
    if "flag_value" in request.form:
        try:
            ENV_OVERSHOOT_FLAG_VALUE=int(request.form["flag_value"])
            changed["flag_value"]=ENV_OVERSHOOT_FLAG_VALUE
        except: pass
    return safe_jsonify({"ok":True,"changed":changed,
                         "current":{"margin_kmh":ENV_OVERSHOOT_MARGIN,
                                    "flag_value":ENV_OVERSHOOT_FLAG_VALUE,
                                    "fallback_no_flags":ENV_OVERSHOOT_FALLBACK_NO_FLAGS}})
# --- helper: newest file in UPLOAD_DIR by mtime ---
def _latest_by_mtime(folder):
    from pathlib import Path
    import os, glob
    candidates = []
    for ext in ("*.mf4", "*.mdf", "*.csv", "*.xlsx", "*.xls"):
        candidates += glob.glob(str(Path(folder) / ext))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(candidates[0]).resolve()


# ------------------------------------------------------------------ Unified /analytics endpoint
@app.post("/analytics")
def analytics():
    started = time.time()
    try:
        # Playground JSON mode
        if request.is_json:
            data = request.get_json(silent=True) or {}

            # Allow /analytics to also return channels for the Playground
            if data.get("fn") == "channels":
                if ACTIVE_FILES:
                    files = [Path(p) for p in ACTIVE_FILES if Path(p).exists()]
                else:
                    latest = _latest_by_mtime(UPLOAD_DIR)
                    files  = [latest] if latest and latest.exists() else []
                if not files:
                    return safe_jsonify({"ok": False, "channels": [], "error": "no_files"}, 400)

                lookup = build_signal_lookup(files)
                channels = [{"id": rid,
                             "name": meta.get("name", rid),
                             "unit": meta.get("unit", "")} for rid, meta in lookup.items()]
                return safe_jsonify({"ok": True, "channels": channels})

            if data.get("fn") == "series":
                ids      = data.get("ids") or []
                tmin_raw = data.get("tmin"); tmax_raw = data.get("tmax")
                algo     = (data.get("algo") or "lttb").lower()
                down     = int(data.get("downsample") or 20000)

                def _flt(v):
                    try: return float(v)
                    except: return None
                tmin=_flt(tmin_raw); tmax=_flt(tmax_raw)

                # Respect ACTIVE_FILES; otherwise use only the newest file in UPLOAD_DIR
                if ACTIVE_FILES:
                    files = [Path(p) for p in ACTIVE_FILES if Path(p).exists()]
                else:
                    latest = _latest_by_mtime(UPLOAD_DIR)
                    files  = [latest] if latest and latest.exists() else []
                if not files:
                    return safe_jsonify({"ok": False, "error": "no_files"}, 400)

                lookup=build_signal_lookup(files)
                resolved=[]; unresolved=[]; seen=set()
                
                # Try to resolve each requested ID using all aliases
                for rid in ids:
                    if rid in seen: continue
                    seen.add(rid)
                    raw = None
                    
                    # Try exact match
                    raw = lookup.get(rid)
                    if raw and raw not in resolved: 
                        resolved.append(raw)
                        continue
                    
                    # Try lowercase
                    raw = lookup.get(rid.lower())
                    if raw and raw not in resolved: 
                        resolved.append(raw)
                        continue
                    
                    # Try all aliases of the requested ID (reverse lookup)
                    # Generate aliases for the requested ID and check if any match
                    request_aliases = _aliases_for_channel(rid)
                    for alias in request_aliases:
                        raw = lookup.get(alias)
                        if raw and raw not in resolved:
                            resolved.append(raw)
                            break
                    
                    # If still not found, try fuzzy matching on lookup keys
                    if not raw or raw in resolved:
                        # Try to find a key that contains parts of the requested ID
                        rid_parts = [p for p in rid.split('_') if len(p) > 3]  # Meaningful parts
                        for key in lookup.keys():
                            if any(part in key or key in part for part in rid_parts):
                                raw = lookup.get(key)
                                if raw and raw not in resolved:
                                    resolved.append(raw)
                                    break
                    
                    # If still not resolved, mark as unresolved
                    if not raw or raw in resolved:
                        unresolved.append(rid)

                if not resolved:
                    return safe_jsonify({"ok":True,"series":{},"unresolved":unresolved,
                                         "meta":{"requested":len(ids),"resolved":0,"unresolved":len(unresolved),
                                                 "algo":algo,"downsample_target":down,
                                                 "duration_ms":round((time.time()-started)*1000,2)}})

                internal_points = max(down*3, down+2000)
                raw_series = extract_series(files, resolved, include_time=True, normalize=False, max_points=internal_points)

                out={}
                for key, s in raw_series.items():
                    xs=list(s.get("timestamps") or []); ys=list(s.get("values") or [])
                    if not xs or not ys: continue
                    if tmin is not None or tmax is not None:
                        sel=[i for i,val in enumerate(xs) if (tmin is None or val>=tmin) and (tmax is None or val<=tmax)]
                        if sel:
                            xs=[xs[i] for i in sel]; ys=[ys[i] for i in sel]
                    if down and len(xs)>down:
                        if algo=="lttb":
                            try: xs,ys=lttb_downsample(xs,ys,down)
                            except: 
                                step=max(1,len(xs)//down); xs=xs[::step]; ys=ys[::step]
                        else:
                            step=max(1,len(xs)//down); xs=xs[::step]; ys=ys[::step]
                    # Frontend expects timestamps and values, not x and y
                    out[key]={"timestamps":xs,"values":ys,"name":s.get("name",key),"unit":s.get("unit",""),
                              "x":xs,"y":ys}  # Keep x/y for backward compatibility

                meta={"requested":len(ids),"resolved":len(out),"unresolved":len(unresolved),
                      "algo":algo,"downsample_target":down,"tmin":tmin,"tmax":tmax,
                      "duration_ms":round((time.time()-started)*1000,2),
                      "files_count":len(files),
                      "files_used":[str(f.name) for f in files]}
                if DEBUG_MODE: meta["debug"]={"incoming_ids":ids,"resolved_raw_keys":list(out.keys()),"files_used":[str(f.name) for f in files]}
                return safe_jsonify({"ok":True,"series":out,"unresolved":unresolved,"meta":meta,"active_files":[str(f.name) for f in files]})

        # Legacy (form) mode continues here unchanged...
        # Legacy mode
        file_paths=ACTIVE_FILES if ACTIVE_FILES else [str(p.resolve()) for p in list_mdf_files(UPLOAD_DIR)]
        files=[Path(p) for p in file_paths if Path(p).exists()]
        if not files: return safe_jsonify({"ok":False,"error":"No MDF files available"})

        sigs=(request.form.get("signals") or request.values.get("signals") or "").strip()
        if not sigs and not request.is_json:
            return safe_jsonify({"ok":False,"error":"No signals provided"})
        if not sigs and request.is_json:
            data=request.get_json(silent=True) or {}
            sigs=data.get("signals","")
            if not sigs:
                return safe_jsonify({"ok":False,"error":"No signals provided"})
        requested=[s.strip() for s in sigs.split(",") if s.strip()]
        
        # build a simple lookup (we keep this for UI lists), but prefer aliasing below
        lookup = build_signal_lookup(files)

        # robust resolution using the same alias generator as extract_series
        resolved = []
        unknown = []
        # build a flattened set of all available canonical channels (values)
        all_canonicals = set()
        for f in files:
            for ch in list_channels(f):
                all_canonicals.add(ch)

        # Build a temporary alias->canonical map consistent with extract_series
        tmp_alias_map = {}
        for ch in sorted(all_canonicals):
            for alias in _aliases_for_channel(ch):
                if alias not in tmp_alias_map:
                    tmp_alias_map[alias] = ch

        for r in requested:
            # try exact
            raw = tmp_alias_map.get(r)
            if not raw:
                # try normalized/cleaned/lower case variations
                raw = tmp_alias_map.get(_clean_name(r)) or tmp_alias_map.get(r.lower()) or tmp_alias_map.get(_clean_name(r).lower())
            if not raw:
                # try URL-decoded
                try:
                    dec = urllib.parse.unquote(r)
                    raw = tmp_alias_map.get(dec) or tmp_alias_map.get(_clean_name(dec))
                except Exception:
                    raw = None
            if not raw:
                # last resort: fuzzy match against cleaned names (gives suggestions)
                cleaned_candidates = list({_clean_name(c) for c in all_canonicals})
                close = difflib.get_close_matches(_clean_name(r), cleaned_candidates, n=3, cutoff=0.6)
                if close:
                    # map close candidates back to canonical original(s) (take first)
                    for c in close:
                        # find canonical channel that has this clean name
                        for ch in all_canonicals:
                            if _clean_name(ch) == c:
                                raw = ch
                                break
                        if raw:
                            break

            if raw:
                resolved.append(raw)
            else:
                unknown.append(r)

        include_time=_bool(request.form.get("include_time"),True)
        normalize=_bool(request.form.get("normalize"),False)
        max_points=int(request.form.get("max_points") or 100000)
        want_fft=_bool(request.form.get("want_fft"),False)
        want_hist=_bool(request.form.get("want_hist"),False)
        seen=set()
        resolved_unique=[r for r in resolved if not (r in seen or seen.add(r))]
        series=extract_series(files,resolved_unique,include_time,normalize,max_points)
        if DEBUG_MODE:
            print(f"[DEBUG]/analytics legacy requested={requested} resolved={len(resolved_unique)} unknown={unknown} series_keys={list(series.keys())}")
        if not series:
            all_clean=set(_clean_name(ch) for ch in lookup.values())
            suggestions={}
            for u in unknown:
                low=u.lower()
                cand=[c for c in sorted(all_clean) if low in c.lower()][:5]
                if cand: suggestions[u]=cand
            
            if DEBUG_MODE:
                print("[DEBUG] requested:", requested)
                print("[DEBUG] resolved_unique:", resolved_unique)
                ch_map = {}
                for f in files:
                    for ch in list_channels(f):
                        for alias in _aliases_for_channel(ch):
                            if alias not in ch_map:
                                ch_map[alias] = ch
                sample_keys = list(sorted(ch_map.keys()))[:40]
                print("[DEBUG] ch_map sample keys:", sample_keys)

            return safe_jsonify({"ok":False,"error":"No series extracted",
                                 "requested_signals":requested,
                                 "matched_signals":[],
                                 "unknown_signals":unknown,
                                 "suggestions":suggestions,
                                 "active_files":ACTIVE_FILES})
        stats=basic_stats(series)
        fft=compute_fft(series) if want_fft else {}
        hist=compute_hist(series) if want_hist else {}
        return safe_jsonify({"ok":True,
                             "requested_signals":requested,
                             "matched_signals":list(series.keys()),
                             "unknown_signals":unknown,
                             "series":series,
                             "stats":stats,
                             "fft":fft,
                             "hist":hist,
                             "active_files":ACTIVE_FILES})
    except Exception as e:
        return json_error("analytics_failed",500,exc=e)

# ------------------------------------------------------------------ Server-side Histogram
@app.post("/api/histogram")
def api_histogram():
    """
    Single-signal histogram (numeric).
    JSON: { "id": "SignalName", "bins":50, "tmin":..., "tmax":..., "downsample":50000 }
    """
    try:
        data=request.get_json(silent=True) or {}
        sig_id=data.get("id")
        if not sig_id:
            return safe_jsonify({"ok":False,"error":"missing_id"},400)
        bins=int(data.get("bins") or 50)
        down=int(data.get("downsample") or 50_000)
        tmin=data.get("tmin"); tmax=data.get("tmax")
        try: tmin_f=float(tmin) if tmin is not None else None
        except: tmin_f=None
        try: tmax_f=float(tmax) if tmax is not None else None
        except: tmax_f=None
        file_paths=ACTIVE_FILES if ACTIVE_FILES else [str(p.resolve()) for p in list_mdf_files(UPLOAD_DIR)]
        files=[Path(p) for p in file_paths if Path(p).exists()]
        if not files:
            return safe_jsonify({"ok":False,"error":"no_files"},400)
        ser=get_single_series(files,sig_id,tmin_f,tmax_f,down)
        if not ser:
            return safe_jsonify({"ok":False,"error":"signal_not_found"},404)
        vals=[v for v in ser["values"] if isinstance(v,(int,float))]
        if not vals:
            return safe_jsonify({"ok":False,"error":"no_numeric_samples"},400)
        try:
            import numpy as np
        except Exception:
            return safe_jsonify({"ok":False,"error":"numpy_not_available"},500)
        h,edges=np.histogram(vals,bins=bins)
        return safe_jsonify({"ok":True,
                             "id":sig_id,
                             "resolved_id":ser["resolved_id"],
                             "counts":h.tolist(),
                             "bins":edges.tolist(),
                             "n":len(vals)})
    except Exception as e:
        return json_error("histogram_failed",500,exc=e)

# ------------------------------------------------------------------ Dash / multi-signal plot endpoint (legacy)
@app.post("/api/dash_plot")
def api_dash_plot():
    try:
        file_paths=ACTIVE_FILES if ACTIVE_FILES else [str(p.resolve()) for p in list_mdf_files(UPLOAD_DIR)]
        files=[Path(p) for p in file_paths if Path(p).exists()]
        signals=[s.strip() for s in (request.form.get("signals") or "").split(",") if s.strip()]
        if not files or not signals:
            return json_error("No signals/files",400)
        max_points=int(request.form.get("max_points") or 10000)
        tickmode=request.form.get("tickmode","auto")
        features=set(filter(None,(request.form.get("features") or "").split(",")))
        lookup=build_signal_lookup(files)
        raw_list=[]
        for sig in signals:
            raw=lookup.get(sig) or lookup.get(sig.lower())
            if raw and raw not in raw_list:
                raw_list.append(raw)
        clean_counts={}
        for raw in raw_list:
            c=_clean_name(raw)
            clean_counts[c]=clean_counts.get(c,0)+1
        label_map={}
        for raw in raw_list:
            c=_clean_name(raw)
            label = c if clean_counts[c]==1 else raw
            base=label; n=2
            while label in label_map:
                label=f"{base}#{n}"; n+=1
            label_map[label]=raw
        assembled={}
        for label,raw in label_map.items():
            xs,ys=[],[]
            for f in files:
                t,v,_=read_signal(f,raw)
                if not t or not v: continue
                xs.extend(t); ys.extend(v)
            if xs and ys:
                if len(xs)>max_points:
                    step=max(1,len(xs)//max_points)
                    xs=xs[::step]; ys=ys[::step]
                assembled[label]=(xs,ys)
        trace_type="scattergl"
        traces=[]; layout={"title":"",
                           "xaxis":{"title":"Time","tickmode":tickmode},
                           "legend":{"orientation":"h"}}
        labels=list(assembled.keys())
        if "subplots" in features and len(labels)>1:
            for i,lbl in enumerate(labels):
                X,Y=assembled[lbl]
                traces.append({"x":X,"y":Y,"type":trace_type,"mode":"lines","name":lbl,"xaxis":"x","yaxis":f"y{i+1}"})
                layout[f"yaxis{i+1}"]={"title":lbl,"anchor":"x",
                                       "domain":[i/len(labels),(i+1)/len(labels)],
                                       "tickmode":tickmode}
        elif "multi_y" in features and len(labels)>1:
            for i,lbl in enumerate(labels):
                X,Y=assembled[lbl]
                yn="y" if i==0 else f"y{i+1}"
                traces.append({"x":X,"y":Y,"type":trace_type,"mode":"lines","name":lbl,"yaxis":yn})
                layout[yn]={"title":lbl,
                            "overlaying":"y" if i>0 else None,
                            "side":"left" if i%2==0 else "right",
                            "tickmode":tickmode}
        else:
            for lbl,(X,Y) in assembled.items():
                traces.append({"x":X,"y":Y,"type":trace_type,"mode":"lines","name":lbl})
            layout["yaxis"]={"title":"Value","tickmode":tickmode}
        return safe_jsonify({"ok":True,"data":traces,"layout":layout,"active_files":ACTIVE_FILES})
    except Exception as e:
        return json_error("dash_plot_failed",500,exc=e)

# ------------------------------------------------------------------ Combined legacy analysis
@app.post("/api/custom_analysis")
def api_custom_analysis():
    try:
        file_paths=ACTIVE_FILES if ACTIVE_FILES else [str(p.resolve()) for p in list_mdf_files(UPLOAD_DIR)]
        files=[Path(p) for p in file_paths if Path(p).exists()]
        if not files: return json_error("No MDF files uploaded",400)
        dfc=compute_dfc(files=files, mapping_file=None, channels=None,
                        compress_runs=True, topn=10, include_plots=False,
                        enable_advanced_features=True)
        try:
            iupr=compute_iupr_plotly(files) if compute_iupr_plotly else {"final":[]}
        except Exception as e:
            iupr={"error":str(e),"final":[]}
        return safe_jsonify({
            "dfc":{"summary":dfc.get("summary",[]),
                   "_meta":{k:dfc[k] for k in dfc if k not in ("summary","plots")}},
            "iupr":{"final":iupr.get("final",[]),
                    "min_max":iupr.get("min_max",[])},
            "active_files":ACTIVE_FILES
        })
    except Exception as e:
        return json_error("custom_analysis_failed",500,exc=e)

# ------------------------------------------------------------------ Hardened Report section
@app.post("/api/report_section")
def api_report_section():
    started=time.time()
    section=(_get_payload_param("section") or "").lower().strip()
    subset_raw=_get_payload_param("files","") or ""
    subset=[p.strip() for p in subset_raw.split(",") if p.strip()]
    min_samples_raw=_get_payload_param("min_samples")
    high_quality_raw=_get_payload_param("high_quality")
    high_quality = high_quality_raw and str(high_quality_raw).lower() in ("1", "true", "yes", "on")
    if DEBUG_MODE:
        print(f"[DEBUG]/api/report_section section='{section}' subset={len(subset)} high_quality={high_quality} ct={request.content_type}")
    try:
        if subset:
            files=[Path(p) for p in subset if Path(p).exists()]
        else:
            files=[Path(p) for p in (ACTIVE_FILES if ACTIVE_FILES else
                                     [str(p.resolve()) for p in list_mdf_files(UPLOAD_DIR)])
                   if Path(p).exists()]
        
        # Filter files by section requirements
        if section in ("dfc", "iupr", "ccsl", "gear", "fuel"):
            # These sections only work with MDF/MF4 files
            files = [f for f in files if f.suffix.lower() in {".mdf", ".mf4", ".mf3"}]
        
        if not files:
            return json_error("No files",400,extra={"section":section})
        if section=="dfc":
            try: payload=_build_dfc_payload(files)
            except Exception as e:
                if DEBUG_MODE: traceback.print_exc()
                payload={"tables":{"DFC Summary":[]},
                         "plots":{},
                         "meta":{"generated_at":int(time.time()),"dfc_error":f"{e.__class__.__name__}: {e}"}}
        elif section=="iupr":
            try: payload=_build_iupr_payload(files)
            except Exception as e:
                if DEBUG_MODE: traceback.print_exc()
                payload={"tables":{"Final Ratios":[],"Min/Max":[]},
                         "plots":{},
                         "meta":{"generated_at":int(time.time()),"iupr_error":f"{e.__class__.__name__}: {e}"}}
        elif section=="ccsl":
            try: payload=_build_ccsl_payload(files)
            except Exception as e:
                if DEBUG_MODE: traceback.print_exc()
                payload={"tables":{"CC/SL Overshoot":[]},
                         "plots":{},
                         "meta":{"generated_at":int(time.time()),
                                 "ccsl_reason":"Failed to compute CC/SL",
                                 "ccsl_exception":f"{e.__class__.__name__}: {e}"}}
        elif section=="gear":
            try:
                from custom_gear import analyze_gear_hunting
                payload = analyze_gear_hunting(files)
                payload["meta"] = payload.get("meta", {})
                payload["meta"]["generated_at"] = int(time.time())
                # Ensure plotly_json is serialized to string if needed
                if "plots" in payload:
                    for plot_name, plot_data in payload["plots"].items():
                        if isinstance(plot_data, dict) and "plotly_json" in plot_data:
                            if not isinstance(plot_data["plotly_json"], str):
                                try:
                                    plot_data["plotly_json"] = json.dumps(plot_data["plotly_json"])
                                except Exception:
                                    pass
            except Exception as e:
                if DEBUG_MODE: traceback.print_exc()
                payload={"tables":{"Hunting Events":[],"File Summary":[],"Statistics":[]},
                         "plots":{},
                         "meta":{"generated_at":int(time.time()),"gear_error":f"{e.__class__.__name__}: {e}"}}
        elif section=="misfire":
            try:
                if compute_misfire_plotly is None:
                    payload = {
                        "tables": {"Misfire Events": [], "File Summary": [], "Statistics": []},
                        "plots": {},
                        "meta": {
                            "ok": False,
                            "error": "compute_misfire_plotly not available",
                            "generated_at": int(time.time())
                        }
                    }
                else:
                    payload = compute_misfire_plotly(files)
                    payload["meta"] = payload.get("meta", {})
                    payload["meta"]["generated_at"] = int(time.time())
                    # Ensure plotly_json is serialized to string if needed
                if "plots" in payload:
                    for plot_name, plot_data in payload["plots"].items():
                        if isinstance(plot_data, dict) and "plotly_json" in plot_data:
                            if not isinstance(plot_data["plotly_json"], str):
                                try:
                                    plot_data["plotly_json"] = json.dumps(plot_data["plotly_json"])
                                except Exception:
                                    pass
            except Exception as e:
                if DEBUG_MODE: traceback.print_exc()
                payload={"tables":{"Misfire Events":[],"File Summary":[],"Statistics":[]},
                         "plots":{},
                         "meta":{"generated_at":int(time.time()),"misfire_error":f"{e.__class__.__name__}: {e}"}}
        elif section == "map":
            maps_root = APP_DIR / "maps_outputs"
            _safe_mkdir(maps_root)
            run_id = uuid.uuid4().hex[:8]
            outdir = maps_root / f"run_{run_id}"
            _safe_mkdir(outdir)
            
            min_samples = None
            if min_samples_raw:
                try: min_samples = int(min_samples_raw)
                except: pass
            
            payload = analyze_empirical_map(files, outdir=outdir, min_samples=min_samples)
            payload.setdefault("meta", {})
            payload["meta"]["generated_at"] = int(time.time())
        elif section == "fuel":
            try:
                payload = _build_fuel_payload(files, high_quality=high_quality)
            except Exception as e:
                if DEBUG_MODE: traceback.print_exc()
                payload={"tables":{"Fuel Summary":[],"Operating Point Analysis":[],"Fuel Channels Found":[]},
                         "plots":{},
                         "meta":{"generated_at":int(time.time()),"fuel_error":f"{e.__class__.__name__}: {e}"}}

        else:
            return json_error("Unknown section",400,extra={"section":section})
        payload["active_files"]=ACTIVE_FILES
        payload.setdefault("meta",{})["duration_ms"]=round((time.time()-started)*1000,2)
        payload["meta"]["section"]=section
        payload["meta"]["file_count"]=len(files)
        payload["meta"]["high_quality"]=high_quality
        return safe_jsonify(payload)
    except Exception as e:
        if DEBUG_MODE: traceback.print_exc()
        return json_error("report_section_failed",500,exc=e,extra={"section":section})

# ------------------------------------------------------------------ Full report
@app.post("/api/report")
def api_report():
    try:
        file_paths=ACTIVE_FILES if ACTIVE_FILES else [str(p.resolve()) for p in list_mdf_files(UPLOAD_DIR)]
        files=[Path(p) for p in file_paths if Path(p).exists()]
        if not files: return json_error("No MDF files uploaded",400)
        dfc=compute_dfc(files=files, mapping_file=None, channels=None,
                        compress_runs=True, topn=10, include_plots=True,
                        enable_advanced_features=True)
        try:
            iupr=compute_iupr_plotly(files) if compute_iupr_plotly else {"final":[], "plots":[]}
        except Exception as e:
            iupr={"error":str(e),"final":[], "plots":[]}
        from custom_gear import analyze_gear_hunting as analyze_gear_hunting_advanced
        gear=analyze_gear_hunting_advanced(files)
        try:
            misfire=compute_misfire_plotly(files) if compute_misfire_plotly else {"tables":{"Misfire Events":[]}, "plots":{}}
        except Exception as e:
            misfire={"error":str(e),"tables":{"Misfire Events":[]}, "plots":{}}
        cc_legacy=report_ccsl(files)
        overshoot=analyze_ccsl_overshoot(files)
        if overshoot.get("plots"):
            for pl in overshoot["plots"]:
                pj=pl.get("plotly_json")
                pl["plotly_json"]=_ensure_plotly_json(pj)
        agg=build_mode_aggregated_ccsl(files, ENV_OVERSHOOT_MARGIN)
        overshoot_plots=overshoot.get("plots",[])+[
            {"name":k,"plotly_json":v["plotly_json"]} for k,v in agg.get("plots",{}).items()
        ]
        return safe_jsonify({
            "dfc":dfc,
            "iupr":iupr,
            "gear":gear,
            "misfire":misfire,
            "cc_sl_legacy":cc_legacy,
            "cc_sl_overshoot":overshoot,
            "cc_sl_mode_plots":overshoot_plots,
            "cc_sl_mode_tables":agg.get("tables", {}),
            "meta":{"files":[f.name for f in files]},
            "active_files":ACTIVE_FILES
        })
    except Exception as e:
        return json_error("report_failed",500,exc=e)

# ------------------------------------------------------------------ Quick DFC_ST only endpoint
@app.post("/api/dfc_analysis")
def api_dfc_analysis():
    """
    Returns only the interpolated DFC_ST multi-line plot for a single file.
    JSON body: { "file": "/path/to/file" }
    If omitted, uses the first active file.
    """
    try:
        data=request.get_json(silent=True) or {}
        file_path=data.get("file")
        target=None
        if file_path:
            p=Path(file_path)
            if p.exists(): target=p
        if not target:
            for af in ACTIVE_FILES:
                p=Path(af)
                if p.exists():
                    target=p; break
        if not target:
            return safe_jsonify({"error":"No file provided / no active files"},400)
        try:
            from custom_dfc import quick_dfc_st
            result=quick_dfc_st(target)
        except Exception as e:
            return safe_jsonify({"error":f"Processing failed: {e}"},500)
        if "error" in result:
            return safe_jsonify(result,400)
        return safe_jsonify({"ok":True, **result})
    except Exception as e:
        return json_error("dfc_analysis_failed",500,exc=e)

# ------------------------------------------------------------------ Debug route: file header & library attempts
@app.get("/api/file_header")
def api_file_header():
    """
    Debug endpoint to inspect an uploaded MF4/MDF quickly.
    Query params:
      - fname : filename (relative to UPLOAD_DIR) OR full path
    Returns:
      - size, first 512 bytes (hex and text), and channel listing attempts using asammdf/mdfreader.
      - if a library fails, returns traceback (when BACKEND_DEBUG enabled).
    """
    try:
        fname = request.args.get("fname") or request.args.get("file")
        if not fname:
            return safe_jsonify({"ok": False, "error": "no_file_provided", "message": "Provide ?fname=uploads/yourfile.mf4 or full path"}, 400)
        p = Path(fname)
        # If a relative name passed, prefer uploads dir
        if not p.exists():
            p = UPLOAD_DIR / Path(fname).name
        if not p.exists():
            return safe_jsonify({"ok": False, "error": "file_not_found", "path": str(p)}, 404)

        info = {"ok": True, "path": str(p.resolve())}
        try:
            info["size_bytes"] = p.stat().st_size
        except Exception as e:
            info["size_bytes"] = None
            if DEBUG_MODE:
                info["stat_error"] = str(e)

        # read header bytes
        try:
            with open(p, "rb") as fh:
                head = fh.read(512)
            info["header_hex"] = head[:256].hex()
            try:
                info["header_text"] = head.decode("utf-8", errors="replace")[:512]
            except Exception:
                info["header_text"] = repr(head[:256])
        except Exception as e:
            info["header_error"] = str(e)

        # Try asammdf channel probe
        asam_res = {"available": MDF is not None}
        if MDF is not None:
            try:
                from asammdf import MDF as _MDF
                m = _MDF(str(p))
                try:
                    try:
                        chs = list(m.channels_db.keys())
                    except Exception:
                        # fallback to channels() if present
                        try:
                            chs = [c for c in m.channels]
                        except Exception:
                            chs = []
                    asam_res["channels_count"] = len(chs)
                    asam_res["channels_sample"] = chs[:200]
                finally:
                    try: m.close()
                    except: pass
            except Exception as e:
                asam_res["error"] = str(e)
                if DEBUG_MODE:
                    asam_res["trace"] = traceback.format_exc().splitlines()[-30:]
        info["asammdf_probe"] = asam_res

        # Try mdfreader channel probe
        mdfr_res = {"available": _HAVE_MDFREADER}
        if _HAVE_MDFREADER:
            try:
                import mdfreader
                m = mdfreader.Mdf(str(p))
                try:
                    try:
                        keys = list(getattr(m, "channels").keys())
                        mdfr_res["channels_count"] = len(keys)
                        mdfr_res["channels_sample"] = keys[:200]
                    except Exception:
                        # try keys() or top-level attributes
                        try:
                            mdfr_res["repr"] = repr(m)[:200]
                        except Exception:
                            pass
                finally:
                    try: m.close()
                    except: pass
            except Exception as e:
                mdfr_res["error"] = str(e)
                if DEBUG_MODE:
                    mdfr_res["trace"] = traceback.format_exc().splitlines()[-30:]
        info["mdfreader_probe"] = mdfr_res

        return safe_jsonify(info)
    except Exception as e:
        return json_error("file_header_failed",500,exc=e)

# ------------------------------------------------------------------ Frontend Error Logging
@app.post("/api/log_error")
def api_log_error():
    try:
        data = request.get_json(force=True, silent=True) or {}
        if data:
            app.logger.error("CLIENT_ERROR :: %s", json.dumps(data)[:2000])
        return safe_jsonify({"ok": True})
    except Exception as e:
        # Avoid causing more errors in the error handler
        app.logger.error("Failed to log client error: %s", e)
        return safe_jsonify({"ok": False, "error": "log_failed"})


# ------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    if os.getenv("FLASK_ACCESS_LOG","1")=="0":
        logging.getLogger("werkzeug").setLevel(logging.WARNING)
    host=os.getenv("FLASK_HOST","127.0.0.1")
    port=int(os.getenv("FLASK_PORT","5000"))
    debug=os.getenv("FLASK_DEBUG","").lower() in ("1","true","yes","on")
    print("ROUTES:", [str(r) for r in app.url_map.iter_rules()])
    app.run(host=host, port=port, debug=debug, use_reloader=False)