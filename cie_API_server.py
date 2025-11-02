# api_server.py
"""
Fully integrated Flask API server for CIE.
Place alongside cie.py and cie.html.

Run:
    python api_server.py

Notes:
 - Exposes /api/* endpoints expected by cie.html
 - Tries to auto-extract training data from CSVs when 'data' not supplied
 - Supports a custom extractor module if you provide one (module names tried: project_extractor, extractor, cie_extractor)
 - Saves models under ./models/*.joblib
"""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Local engine (must be in same dir)
import cie
from cie import (
    CalibrationIntelligenceEngine,
    standardize_response,
)

# Optional imports used by CSV extraction
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cie_api")

app = Flask(__name__, static_folder=".")
CORS(app)

# Where models will be stored
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Define an upload directory
UPLOAD_FOLDER = Path.cwd() / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)

# --- Instantiate the main CIE Engine and its components ---
cie_engine = CalibrationIntelligenceEngine(model_dir=MODEL_DIR)
model_mgr = cie_engine.model_manager
opt_engine = cie_engine.optimization_engine
map_gen = cie_engine.map_generator
interp_engine = cie_engine.interpolation_engine
rec_engine = cie_engine.recommendation_engine
doe_engine = cie_engine.doe_engine


# Global variable to store data from uploaded files
UPLOADED_DATA: Optional[pd.DataFrame] = None

# Fallback directories to search automatically for CSV training data
DEFAULT_FALLBACK_DIRS = [
    str(UPLOAD_FOLDER), # Prioritize uploads
    r"D:\OBD",
    r"D:\OBD\CC21\cc21_obd_xls_dfc",
    r"D:\OBD\CC21\dashboar_merge\FINAL",
    str(Path.cwd()),
]

# ------------------------
# Helper functions
# ------------------------
def _ok(data: Any = None, message: str = ""):
    return jsonify(standardize_response(data=data if data is not None else {}, message=message, success=True))


def _err(msg: str = "error", exc: Optional[Exception] = None, status_code: int = 400):
    payload = {"error": msg}
    if exc:
        payload["trace"] = traceback.format_exc()
    logger.error(f"API Error: {msg}", exc_info=exc)
    return jsonify(standardize_response(data=payload, message=str(msg), success=False)), status_code


def _find_csv_files(root_dir: str, pattern: str = "*.csv", max_files: int = 200) -> List[str]:
    """Recursively find CSV files (limit to max_files), newest first."""
    root = Path(root_dir)
    if not root.exists():
        return []
    files = list(map(str, root.rglob(pattern)))
    files = sorted(files, key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return files[:max_files]


def _try_import_custom_extractor() -> Optional[Callable]:
    """
    Try importing a user-provided custom extractor function.
    Accept module/file names: project_extractor, extractor, cie_extractor
    Function expected: extract_series_for_training(data_path, features, targets) -> (X_df, y_df)
    """
    candidates = [
        ("project_extractor", "extract_series_for_training"),
        ("extractor", "extract_series_for_training"),
        ("cie_extractor", "extract_series_for_training"),
    ]
    for module_name, func_name in candidates:
        try:
            mod = __import__(module_name, fromlist=[func_name])
            func = getattr(mod, func_name, None)
            if callable(func):
                logger.info(f"Using custom extractor: {module_name}.{func_name}")
                return func
        except Exception:
            continue
    return None


def _auto_extract_from_csvs(
    data_path: str, features: List[str], targets: List[str], max_files: int = 100, chunksize: int = 200_000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scan CSV files under data_path, read only requested columns if available,
    concatenate chunks and return aligned X,y DataFrames.
    Raises informative errors if unable to build datasets.
    """
    files = _find_csv_files(data_path, pattern="*.csv", max_files=max_files)
    if not files:
        raise FileNotFoundError(f"No CSV files found under {data_path}")

    needed_cols = list(dict.fromkeys((features or []) + (targets or [])))
    collected = []
    total_rows = 0

    for f in files:
        # Read header safely (try utf-8 then latin-1)
        try:
            header = pd.read_csv(f, nrows=0, encoding="utf-8", engine="c")
            cols = list(header.columns)
        except Exception:
            try:
                header = pd.read_csv(f, nrows=0, encoding="latin-1", engine="c")
                cols = list(header.columns)
            except Exception:
                cols = []

        usecols = [c for c in needed_cols if c in cols]
        if not usecols:
            continue

        # Read in chunks to avoid OOM
        try:
            reader = pd.read_csv(f, usecols=usecols, encoding="utf-8", chunksize=chunksize, low_memory=False)
        except Exception:
            reader = pd.read_csv(f, usecols=usecols, encoding="latin-1", chunksize=chunksize, low_memory=False)

        for chunk in reader:
            collected.append(chunk)
            total_rows += len(chunk)
        # stop early if large enough
        if total_rows > 500_000:
            break

    if not collected:
        raise ValueError(f"No usable columns found in CSV files at {data_path} for features/targets: {needed_cols}")

    combined = pd.concat(collected, axis=0, ignore_index=True)
    # Ensure requested columns exist
    X = combined[[c for c in features if c in combined.columns]].copy() if features else pd.DataFrame()
    y = combined[[c for c in targets if c in combined.columns]].copy() if targets else pd.DataFrame()

    # Align rows where both X and y exist
    if not X.empty and not y.empty:
        df_all = pd.concat([X, y], axis=1)
        df_all = df_all.replace([np.inf, -np.inf], np.nan).dropna()
        X = df_all[X.columns]
        y = df_all[y.columns]
    else:
        # If either is empty, drop NaNs separately and check
        if not X.empty:
            X = X.replace([np.inf, -np.inf], np.nan).dropna()
        if not y.empty:
            y = y.replace([np.inf, -np.inf], np.nan).dropna()
        if X.empty or y.empty:
            raise ValueError("Insufficient columns present across CSVs to build both X and y. Provide 'data' or adjust feature/target lists.")

    return X.reset_index(drop=True), y.reset_index(drop=True)


# ------------------------
# API endpoints
# ------------------------
@app.route("/api/list_models", methods=["GET"])
def list_models():
    try:
        models = model_mgr.list_models()
        return _ok(models)
    except Exception as e:
        return _err("list_models failed", e)


@app.route("/api/load_model/<model_id>", methods=["GET"])
def load_model(model_id: str):
    try:
        model = model_mgr.load_model(model_id)
        info = model.get_info()
        return _ok({"model_id": model_id, "model_info": info})
    except Exception as e:
        return _err(f"load_model {model_id} failed", e)


@app.route("/api/delete_model", methods=["POST"])
def delete_model():
    try:
        body = request.get_json(force=True, silent=True) or {}
        model_id = body.get("model_id") or request.form.get("model_id")
        if not model_id:
            return _err("model_id required")
        ok = model_mgr.delete_model(model_id)
        return _ok({"deleted": ok})
    except Exception as e:
        return _err("delete_model failed", e)


@app.route("/api/train_model", methods=["POST"])
def train_model():
    """
    Enhanced train_model:
     - Priority 1: Use data from file uploads if available.
     - Priority 2: If request contains JSON 'data' => use it (list of rows)
     - Priority 3: Else if 'data_path' provided => auto-scan CSVs under that folder
     - Priority 4: Else try env CIE_DATA_DIR or fallback common dirs
     - Supports custom extractor module if present (project_extractor.extract_series_for_training)
    """
    global UPLOADED_DATA
    try:
        # Normalize request body (accept form or JSON)
        if request.is_json:
            body = request.get_json()
        else:
            body = dict(request.form) or {}
            # Handle multi-select form fields
            for k, v in request.form.lists():
                if len(v) > 1:
                    body[k] = v
                elif len(v) == 1:
                    body[k] = v[0]

        features = body.get("features", [])
        targets = body.get("targets", [])
        model_type = body.get("model_type", "random_forest")
        model_id = body.get("model_id", "main_model")

        # Ensure features/targets are lists
        if isinstance(features, str): features = [features]
        if isinstance(targets, str): targets = [targets]

        # 1) Check for globally available uploaded data first
        if UPLOADED_DATA is not None and not UPLOADED_DATA.empty:
            logger.info(f"Using uploaded data with shape: {UPLOADED_DATA.shape}")
            df = UPLOADED_DATA
            
            if not features or not targets:
                return _err(f"When using uploaded data, 'features' and 'targets' are required. Available columns: {df.columns.tolist()}", status_code=400)

            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                return _err(f"Features not found in uploaded data: {missing_features}. Available: {df.columns.tolist()}", status_code=400)
            
            missing_targets = [t for t in targets if t not in df.columns]
            if missing_targets:
                return _err(f"Targets not found in uploaded data: {missing_targets}. Available: {df.columns.tolist()}", status_code=400)

            X_raw = df[features].copy()
            y_raw = df[targets].copy()

            # Clean and align data
            combined = pd.concat([X_raw, y_raw], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            
            if combined.empty:
                return _err("No valid data rows remaining after cleaning (removing NaN/Inf). Check data quality.", status_code=400)

            X = combined[features]
            y = combined[targets]
            
            logger.info(f"Training with {len(X)} samples from uploaded data.")
            metrics = model_mgr.train_model(model_id=model_id, X=X, y=y, model_type=model_type)
            info = model_mgr.get_model(model_id).get_info()
            return _ok({"model_id": model_id, "metrics": metrics, "model_info": info, "source": "uploaded_data"})

        # 2) If explicit 'data' provided in request body, use it
        data = body.get("data")
        if data:
            logger.info("Using 'data' provided in the request body.")
            if isinstance(data, str):
                data = json.loads(data)
            df = pd.DataFrame(data)
            if not features or not targets:
                return _err("When providing 'data', you must also provide 'features' and 'targets' lists.")
            X = df[features].copy()
            y = df[targets].copy()
            metrics = model_mgr.train_model(model_id=model_id, X=X, y=y, model_type=model_type)
            info = model_mgr.get_model(model_id).get_info()
            return _ok({"model_id": model_id, "metrics": metrics, "model_info": info, "source": "request_data"})

        # 3) Try custom extractor if available and data_path exists
        custom_extractor = _try_import_custom_extractor()
        data_path = body.get("data_path") or os.environ.get("CIE_DATA_DIR")
        if custom_extractor and data_path:
            logger.info(f"Using custom extractor with path: {data_path}")
            try:
                X, y = custom_extractor(data_path, features, targets)
                metrics = model_mgr.train_model(model_id=model_id, X=X, y=y, model_type=model_type)
                info = model_mgr.get_model(model_id).get_info()
                return _ok({"model_id": model_id, "metrics": metrics, "model_info": info, "source": f"custom_extractor:{data_path}"})
            except Exception as e:
                logger.warning("Custom extractor failed", exc_info=e)

        # 4) Try data_path or fallback directories for automatic CSV extraction
        logger.info("No uploaded data found, falling back to CSV auto-extraction.")
        tried_paths = []
        search_paths = ([data_path] if data_path else []) + DEFAULT_FALLBACK_DIRS
        for p in search_paths:
            if p in tried_paths: continue
            tried_paths.append(p)
            try:
                logger.info(f"Attempting auto-extraction from: {p}")
                X, y = _auto_extract_from_csvs(p, features, targets)
                metrics = model_mgr.train_model(model_id=model_id, X=X, y=y, model_type=model_type)
                info = model_mgr.get_model(model_id).get_info()
                return _ok({"model_id": model_id, "metrics": metrics, "model_info": info, "source": f"csv_auto:{p}"})
            except (FileNotFoundError, ValueError) as e:
                logger.info(f"Auto-extract from {p} failed: {e}")
                continue
            except Exception as e:
                logger.warning(f"An unexpected error occurred during auto-extraction from {p}", exc_info=e)
                continue

        return _err(f"No training data found or processed. Provide data via file upload or configure a data path. Tried paths: {tried_paths}", status_code=404)
    except Exception as e:
        return _err("train_model failed unexpectedly", e, status_code=500)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Body JSON:
      {
        "model_id": "main_model",
        "inputs": { "rpm": 2000, "load": 50 } OR [{...}, {...}]
      }
    """
    try:
        body = request.get_json(force=True)
        model_id = body.get("model_id")
        inputs = body.get("inputs")
        
        model = model_mgr.get_model(model_id) # get_model handles active model logic
        model_id = model_id or model_mgr.active_model

        if isinstance(inputs, dict):
            df = pd.DataFrame([inputs])
        elif isinstance(inputs, list):
            df = pd.DataFrame(inputs)
        else:
            return _err("Invalid 'inputs', provide dict or list of dicts")

        preds = model.predict(df)
        return _ok({"model_id": model_id, "predictions": preds.to_dict(orient="records")})
    except Exception as e:
        return _err("predict failed", e)


@app.route("/api/predict_with_uncertainty", methods=["POST"])
def predict_with_uncertainty():
    try:
        body = request.get_json(force=True)
        model_id = body.get("model_id")
        inputs = body.get("inputs")

        model = model_mgr.get_model(model_id)
        model_id = model_id or model_mgr.active_model

        if isinstance(inputs, dict):
            df = pd.DataFrame([inputs])
        elif isinstance(inputs, list):
            df = pd.DataFrame(inputs)
        else:
            return _err("Invalid 'inputs', provide dict or list of dicts")

        if not hasattr(model, "predict_with_uncertainty"):
            return _err(f"Model type '{model.__class__.__name__}' does not support uncertainty prediction.")

        preds, uncert = model.predict_with_uncertainty(df)
        out = []
        for i in range(len(preds)):
            row = {}
            row.update(preds.iloc[i].to_dict())
            row.update(uncert.iloc[i].to_dict())
            out.append(row)
        return _ok({"model_id": model_id, "predictions": out})
    except Exception as e:
        return _err("predict_with_uncertainty failed", e)


@app.route("/api/optimize", methods=["POST"])
def optimize():
    try:
        body = request.get_json(force=True)
        model_id = body.get("model_id")
        targets = body.get("targets", {})
        constraints = body.get("constraints", {})
        initial_guess = body.get("initial_guess", {})
        res = opt_engine.optimize(model_id=model_id, targets=targets, constraints=constraints, initial_guess=initial_guess)
        return _ok(res)
    except Exception as e:
        return _err("optimize failed", e)


@app.route("/api/bayesian_optimize", methods=["POST"])
def bayesian_optimize():
    try:
        body = request.get_json(force=True)
        model_id = body.get("model_id")
        targets = body.get("targets", {})
        constraints = body.get("constraints", {})
        n_calls = int(body.get("n_calls", 50))
        res = opt_engine.bayesian_optimize(model_id=model_id, targets=targets, constraints=constraints, n_calls=n_calls)
        return _ok(res)
    except Exception as e:
        return _err("bayesian_optimize failed", e)


@app.route("/api/generate_map", methods=["POST"])
def generate_map():
    """
    Body JSON:
      {
        "model_id": "main_model",
        "grid_definitions": { "rpm": [1000,2000,3000], "load":[10,50,100] }
      }
    """
    try:
        body = request.get_json(force=True)
        model_id = body.get("model_id")
        grid_defs = body.get("grid_definitions")
        if not model_id or not grid_defs:
            return _err("model_id and grid_definitions required")
        map_data = map_gen.generate_map(model_id=model_id, grid_definitions=grid_defs)
        return _ok(map_data)
    except Exception as e:
        return _err("generate_map failed", e)


@app.route("/api/export_map", methods=["POST"])
def export_map():
    try:
        body = request.get_json(force=True)
        map_data = body.get("map_data")
        fmt = body.get("fmt", "csv")
        if not map_data:
            return _err("map_data required")
        exported = map_gen.export_map(map_data, fmt=fmt)
        return _ok({"exported": exported})
    except Exception as e:
        return _err("export_map failed", e)


@app.route("/api/interp_build_map", methods=["POST"])
def interp_build_map():
    """
    Body JSON:
      {
        "rows": [{axis1: val, axis2: val, target: val}, ...],
        "axes": ["rpm", "load"],
        "target": "Torque",
        "method": "auto",
        "resolution": 30
      }
    """
    try:
        body = request.get_json(force=True)
        rows = body.get("rows")
        axes = body.get("axes")
        target = body.get("target")
        method = body.get("method", "auto")
        resolution = int(body.get("resolution", 30))
        if not rows or not axes or not target:
            return _err("rows, axes and target required")
        df = pd.DataFrame(rows)
        result = interp_engine.build_map(df=df, axes=axes, target=target, method=method, resolution=resolution)
        return _ok(result)
    except Exception as e:
        return _err("interp_build_map failed", e)

# -----------------------------------
# NEW/STUBBED ENDPOINTS FROM ANALYSIS
# -----------------------------------

@app.route("/api/train_map_model", methods=["POST"])
def train_map_model():
    """Stub endpoint for training a map-specific model."""
    logger.info("Received request for /api/train_map_model. This is a stub and will reuse the main training logic.")
    # For now, this is just a proxy to the main train_model endpoint with a different model_id
    # A more advanced implementation would have specialized logic here.
    return train_model()


@app.route("/api/get_recommendations", methods=["POST"])
def get_recommendations():
    """Get optimization recommendations based on a model or map."""
    try:
        body = request.get_json(force=True)
        
        # This endpoint can be model-based or map-based. We'll decide based on inputs.
        model_id = body.get("model_id")
        df_measured_data = body.get("df_measured")
        
        if model_id and df_measured_data:
            logger.info(f"Performing model-based recommendation for model: {model_id}")
            df_measured = pd.DataFrame(df_measured_data)
            # The target for recommendation needs to be determined. Let's assume the first target of the model.
            model = model_mgr.get_model(model_id)
            target = model.targets[0] if model.targets else df_measured.columns[-1]
            
            recommendations = rec_engine.recommend(model_id, df_measured, target)
            return _ok({"recommendations": recommendations.to_dict(orient="records")})
        else:
            # Map-based recommendation logic would go here if needed
            return _err("For model-based recommendations, 'model_id' and 'df_measured' are required.")
    except Exception as e:
        return _err("get_recommendations failed", e, status_code=500)


@app.route("/api/generate_doe", methods=["POST"])
def generate_doe():
    """Generate Design of Experiments samples."""
    try:
        body = request.get_json(force=True)
        param_ranges = body.get("param_ranges", {})
        n_samples = int(body.get("n_samples", 20))
        method = body.get("method", "lhs")
        
        if not param_ranges:
            return _err("'param_ranges' are required to generate a DOE.")
            
        samples = doe_engine.generate_doe(param_ranges, n_samples, method)
        
        return _ok({"doe_samples": samples.to_dict(orient="records")})
    except Exception as e:
        return _err("generate_doe failed", e, status_code=500)

# ------------------------
# File Upload Endpoint
# ------------------------
@app.route('/api/smart_merge_upload', methods=['POST'])
def smart_merge_upload():
    """
    Handle file uploads from the frontend.
    Processes uploaded files (CSV, MDF, MF4) into a pandas DataFrame
    and stores it in a global variable for training.
    """
    global UPLOADED_DATA
    try:
        if 'files' not in request.files:
            return _err("No 'files' part in the request")

        files = request.files.getlist('files')
        if not files or all(not f.filename for f in files):
            return _err("No files selected")
            
        mode = request.form.get('mode', 'append')
        
        saved_files = []
        all_channels = []
        processed_data_frames = []
        
        for file in files:
            if not (file and file.filename):
                continue

            filename = secure_filename(file.filename)
            filepath = UPLOAD_FOLDER / filename
            file.save(str(filepath))
            saved_files.append(str(filepath))
            logger.info(f"Saved uploaded file: {filepath}")
            
            try:
                if filename.lower().endswith('.csv'):
                    logger.info(f"Processing CSV file: {filename}")
                    df = pd.read_csv(filepath)
                    processed_data_frames.append(df)
                    channels = [{"id": col, "name": col, "type": "csv", "file": filename} for col in df.columns]
                    all_channels.extend(channels)
                    
                elif filename.lower().endswith(('.mdf', '.mf4')):
                    logger.info(f"Processing MDF/MF4 file: {filename}")
                    try:
                        from asammdf import MDF
                        mdf = MDF(str(filepath))
                        
                        available_channels = list(mdf.channels_db.keys())
                        if not available_channels:
                            logger.warning(f"No channels found in MDF file: {filename}")
                            continue
                        
                        # Convert MDF to a single DataFrame
                        # Note: This can be memory intensive for large files with many channels.
                        df = mdf.to_dataframe()
                        processed_data_frames.append(df)
                        
                        for channel in df.columns:
                            all_channels.append({"id": channel, "name": channel, "type": "mdf", "file": filename})
                            
                    except ImportError:
                        msg = "MDF processing requires 'asammdf' package. Please install it (`pip install asammdf`)."
                        logger.error(msg)
                        return _err(msg, status_code=501)
                    except Exception as e:
                        logger.warning(f"Could not process MDF file {filename}: {e}")
                        all_channels.append({"id": f"error_{filename}", "name": f"Processing error: {e}", "type": "error"})

            except Exception as e:
                logger.error(f"Failed to process file {filename}", exc_info=e)

        # Combine all processed dataframes into one
        if processed_data_frames:
            logger.info(f"Combining {len(processed_data_frames)} dataframes from uploaded files.")
            if mode == 'append' and UPLOADED_DATA is not None:
                # Append to existing data
                UPLOADED_DATA = pd.concat([UPLOADED_DATA] + processed_data_frames, ignore_index=True, sort=False)
            else:
                # Overwrite with new data
                UPLOADED_DATA = pd.concat(processed_data_frames, ignore_index=True, sort=False)
            logger.info(f"Combined data shape is now: {UPLOADED_DATA.shape}")
        else:
            logger.warning("No data was processed from the uploaded files.")

        # Make unique channels list
        unique_channels = [dict(t) for t in {tuple(d.items()) for d in all_channels}]

        return _ok({
            "ok": True,
            "added": saved_files,
            "channels": unique_channels,
            "upload_dir": str(UPLOAD_FOLDER),
            "mode": mode,
            "data_processed": UPLOADED_DATA is not None,
            "data_shape": UPLOADED_DATA.shape if UPLOADED_DATA is not None else (0, 0)
        })
        
    except Exception as e:
        return _err(f"File upload failed: {str(e)}", e, status_code=500)


# Serve the static cie.html
@app.route("/", methods=["GET"])
def root_html():
    try:
        return send_from_directory(".", "cie_Version2.html")
    except Exception:
        return "Place cie_Version2.html in the same folder as api_server.py", 404


# health check
@app.route("/api/health", methods=["GET"])
def health():
    try:
        return _ok({"status": "ok", "models_dir": str(MODEL_DIR.resolve())})
    except Exception as e:
        return _err("health check failed", e)


if __name__ == "__main__":
    host = os.environ.get("CIE_HOST", "127.0.0.1")
    port = int(os.environ.get("CIE_PORT", 5000))
    print("Please install required packages using: pip install Flask Flask-Cors joblib numpy pandas scikit-learn scipy scikit-optimize asammdf")
    logger.info(f"Starting CIE API server on http://{host}:{port}")
    app.run(host=host, port=port, debug=True)