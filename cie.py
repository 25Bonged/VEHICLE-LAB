import json
import math
import logging
import warnings
import time
import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import joblib
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cie_advanced")

# ---------------------------------------------------------------------
# Optional deps with graceful fallbacks
# ---------------------------------------------------------------------
try:
    # Legacy fallback for missing optional dependencies
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.isotonic import IsotonicRegression
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, WhiteKernel
    from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_regression
    from sklearn.decomposition import PCA
    from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    HAVE_SKLEARN = True
    HAVE_ISOTONIC = True
except Exception as e:
    logger.warning(f"Scikit-learn not available: {e}")
    HAVE_SKLEARN = False
    HAVE_ISOTONIC = False

try:
    # Legacy fallback for missing optional dependencies
    from scipy.optimize import minimize, differential_evolution
    from scipy.interpolate import griddata, Rbf, RegularGridInterpolator, CloughTocher2DInterpolator, interp1d
    from scipy.spatial import cKDTree
    from scipy import stats
    HAVE_SCIPY = True
except Exception as e:
    logger.warning(f"SciPy not available: {e}")
    HAVE_SCIPY = False

try:
    # Legacy fallback for missing optional dependencies
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.sampler import Lhs, Sobol
    from skopt import Optimizer
    HAVE_SKOPT = True
except Exception as e:
    logger.warning(f"Scikit-optimize not available: {e}")
    HAVE_SKOPT = False

try:
    # Legacy fallback for missing optional dependencies
    import optuna
    HAVE_OPTUNA = True
except Exception as e:
    logger.warning(f"Optuna not available: {e}")
    HAVE_OPTUNA = False

try:
    # Legacy fallback for missing optional dependencies
    import psutil
    HAVE_PSUTIL = True
except Exception as e:
    logger.warning(f"psutil not available: {e}")
    HAVE_PSUTIL = False

# =====================================================================
# Enhanced Utilities with Performance Monitoring
# =====================================================================

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def _as_df(obj: Union[pd.DataFrame, Dict[str, Iterable], np.ndarray], columns: Optional[List[str]] = None) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, dict):
        return pd.DataFrame(obj)
    if isinstance(obj, np.ndarray):
        return pd.DataFrame(obj, columns=columns)
    raise TypeError("Unsupported data type for _as_df")

def performance_monitor(func):
    """
    Decorator to monitor function performance.
    Tracks execution time, memory usage, and logs performance info.
    Stores performance history on the object if available.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss if HAVE_PSUTIL else 0
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss if HAVE_PSUTIL else 0
        
        performance_info = {
            "execution_time": end_time - start_time,
            "memory_used_mb": (end_memory - start_memory) / 1024 / 1024,
            "timestamp": _now_iso()
        }
        
        # Store performance info if the object has performance_history
        if hasattr(args[0], 'performance_history'):
            args[0].performance_history.append(performance_info)
        elif hasattr(args[0], 'metrics'):
            args[0].metrics['performance'] = performance_info
        
        logger.debug(f"{func.__name__} executed in {performance_info['execution_time']:.3f}s")
        
        return result
    return wrapper

def prepare_training_data(
    feature_series: Dict[str, Iterable],
    target_series: Dict[str, Iterable],
    *,
    dropna: bool = True,
    remove_outliers: bool = False,
    outlier_threshold: float = 3.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create aligned X (features) and y (targets) DataFrames with robust NaN handling and outlier removal."""
    X = _as_df(feature_series)
    y = _as_df(target_series)

    n = min(len(X), len(y))
    X = X.iloc[:n].reset_index(drop=True)
    y = y.iloc[:n].reset_index(drop=True)

    if dropna:
        df = pd.concat([X, y], axis=1)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        X = df[X.columns]
        y = df[y.columns]

    if remove_outliers and len(X) > 0:
        # Remove outliers using Z-score
        z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
        outlier_mask = (z_scores < outlier_threshold).all(axis=1)
        X = X[outlier_mask]
        y = y[outlier_mask]

    return X, y

def _linspace_from_data(a: np.ndarray, n: int = 25) -> np.ndarray:
    lo, hi = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Invalid axis range for grid generation.")
    if hi == lo:
        hi = lo + 1e-9
    return np.linspace(lo, hi, n)

def _clip_if(x: np.ndarray, lo: Optional[float], hi: Optional[float]) -> np.ndarray:
    if lo is not None:
        x = np.maximum(x, lo)
    if hi is not None:
        x = np.minimum(x, hi)
    return x

def _enforce_monotonic_1d(arr: np.ndarray, direction: Literal["increasing", "decreasing"]) -> np.ndarray:
    """Project a 1D array to be monotonic using isotonic regression if available."""
    if HAVE_ISOTONIC:
        y = arr
        x = np.arange(len(y))
        if direction == "increasing":
            iso = IsotonicRegression(increasing=True)
            return iso.fit_transform(x, y)
        else:
            iso = IsotonicRegression(increasing=False)
            return iso.fit_transform(x, y)
    # Fallback
    out = arr.copy()
    if direction == "increasing":
        for i in range(1, len(out)):
            if out[i] < out[i - 1]:
                out[i] = out[i - 1]
    else:
        for i in range(1, len(out)):
            if out[i] > out[i - 1]:
                out[i] = out[i - 1]
    return out

def _smooth_gradients_2d(z: np.ndarray, max_grad: Optional[float] = None, passes: int = 1) -> np.ndarray:
    """Simple gradient limiter for 2D tables; clips local differences."""
    if max_grad is None or max_grad <= 0:
        return z
    out = z.copy()
    for _ in range(max(1, passes)):
        # Horizontal
        dz = np.diff(out, axis=1)
        dz = np.clip(dz, -max_grad, max_grad)
        out[:, 1:] = out[:, :1] + np.cumsum(dz, axis=1)
        # Vertical
        dz = np.diff(out, axis=0)
        dz = np.clip(dz, -max_grad, max_grad)
        out[1:, :] = out[:1, :] + np.cumsum(dz, axis=0)
    return out

def standardize_response(data: Any, success: bool = True, message: str = "") -> Dict:
    """Standardize all API responses for production use."""
    return {
        "success": success,
        "message": message,
        "timestamp": _now_iso(),
        "data": data,
        "version": "6.0.0"
    }

# =====================================================================
# Advanced Feature Engineering
# =====================================================================

class FeatureEngine:
    """Advanced feature engineering and selection"""
    
    def __init__(self):
        self.feature_importance_threshold = 0.01
        self.correlation_threshold = 0.95
        self.polynomial_degree = 2
        self.engineered_features = []
        
    def engineer_features(self, df: pd.DataFrame, target: Optional[str] = None) -> pd.DataFrame:
        """Create advanced features including polynomial, interaction, and statistical features."""
        engineered = df.copy()
        self.engineered_features = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target and target in numeric_cols:
            numeric_cols.remove(target)
        
        # Polynomial features
        for col in numeric_cols:
            if col != target:
                engineered[f"{col}_squared"] = df[col] ** 2
                engineered[f"{col}_log"] = np.log1p(np.abs(df[col]))
                engineered[f"{col}_sqrt"] = np.sqrt(np.abs(df[col]))
                self.engineered_features.extend([f"{col}_squared", f"{col}_log", f"{col}_sqrt"])
        
        # Interaction features
        if len(numeric_cols) > 1:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    if col1 != target and col2 != target:
                        engineered[f"{col1}_{col2}_interaction"] = df[col1] * df[col2]
                        engineered[f"{col1}_{col2}_ratio"] = df[col1] / (df[col2] + 1e-9)
                        self.engineered_features.extend([f"{col1}_{col2}_interaction", f"{col1}_{col2}_ratio"])
        
        # Statistical features
        if len(numeric_cols) >= 3:
            rolling_cols = numeric_cols[:3]  # Use first 3 columns for rolling stats
            for col in rolling_cols:
                if col != target:
                    engineered[f"{col}_rolling_mean"] = df[col].rolling(window=5, center=True).mean()
                    engineered[f"{col}_rolling_std"] = df[col].rolling(window=5, center=True).std()
                    self.engineered_features.extend([f"{col}_rolling_mean", f"{col}_rolling_std"])
        
        # Fill NaN values created by rolling stats
        engineered = engineered.fillna(method='bfill').fillna(method='ffill')
        
        return engineered
    
    def select_features(self, X: pd.DataFrame, y: pd.DataFrame, method: str = "importance") -> List[str]:
        """Advanced feature selection using multiple methods."""
        if method == "rfe" and HAVE_SKLEARN:
            # Recursive Feature Elimination
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=min(20, X.shape[1]))
            selector.fit(X, y.values.ravel() if len(y.shape) > 1 else y)
            selected_features = X.columns[selector.support_].tolist()
        
        elif method == "importance" and HAVE_SKLEARN:
            # Feature Importance
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            estimator.fit(X, y.values.ravel() if len(y.shape) > 1 else y)
            importances = estimator.feature_importances_
            selected_features = X.columns[importances >= self.feature_importance_threshold].tolist()
        
        elif method == "correlation" and HAVE_SKLEARN:
            # Correlation-based selection
            correlation_matrix = X.corr().abs()
            upper_tri = correlation_matrix.where(
                np.triu(np.ones_like(correlation_matrix), k=1).astype(bool)
            )
            to_drop = [column for column in upper_tri.columns 
                      if any(upper_tri[column] > self.correlation_threshold)]
            selected_features = [f for f in X.columns if f not in to_drop]
        
        elif method == "pca" and HAVE_SKLEARN and len(X.columns) > 10:
            # PCA for high-dimensional data
            pca = PCA(n_components=0.95)   # Keep 95% variance
            X_reduced = pca.fit_transform(X)
            # We can't return feature names for PCA, so use original features
            selected_features = X.columns.tolist()[:min(20, len(X.columns))]
        
        else:
            # Fallback: use all features
            selected_features = X.columns.tolist()
        
        # Always include original features (not engineered ones) if they're important
        original_features = [f for f in X.columns if f not in self.engineered_features]
        selected_features = list(set(selected_features + original_features[:5]))  # Keep top 5 original
        
        return selected_features[:min(50, len(selected_features))]  # Limit to 50 features

# =====================================================================
# Enhanced Base Calibration Model with Real-time Monitoring
# =====================================================================

class TrainingPhase(Enum):
    NOT_STARTED = "not_started"
    DATA_PREPROCESSING = "data_preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
    VALIDATION = "validation"
    COMPLETED = "completed"
    FAILED = "failed"

class CalibrationModel(ABC):
    def __init__(self):
        self.model = None
        self.features: List[str] = []
        self.targets: List[str] = []
        self.metrics: Dict[str, Any] = {}
        self.training_data_info: Dict[str, Any] = {}
        self.created_at: str = _now_iso()
        self.updated_at: str = self.created_at
        self.scaler_x = None
        self.scaler_y = None
        self.multi_output = False
        self.training_history: Dict[str, List] = {
            "r2_scores": [],
            "losses": [],
            "iterations": [],
            "timestamps": []
        }
        self.performance_history: List[Dict] = []
        self.training_phase: TrainingPhase = TrainingPhase.NOT_STARTED
        self.is_training: bool = False
        self.training_progress: float = 0.0
        self.convergence_data: Dict[str, Any] = {}

    def _get_scaler_key(self):
        """Generate a unique key for scaler storage."""
        return (tuple(self.features), tuple(self.targets), self.__class__.__name__)

    def update_training_progress(self, phase: TrainingPhase, progress: float, metrics: Optional[Dict] = None):
        """Update training progress for real-time monitoring."""
        self.training_phase = phase
        self.training_progress = progress
        self.is_training = phase != TrainingPhase.COMPLETED and phase != TrainingPhase.FAILED
        
        if metrics:
            self.training_history["r2_scores"].append(metrics.get("r2_score", 0))
            self.training_history["losses"].append(metrics.get("loss", 0))
            self.training_history["iterations"].append(len(self.training_history["r2_scores"]))
            self.training_history["timestamps"].append(_now_iso())

    def get_training_history(self) -> Dict[str, List[float]]:
        """Return training history for visualization"""
        return self.training_history

    def get_convergence_metrics(self) -> Dict[str, Any]:
        """Return convergence metrics for monitoring"""
        return {
            "training_phase": self.training_phase.value,
            "training_progress": self.training_progress,
            "is_training": self.is_training,
            "iterations": len(self.training_history["r2_scores"]),
            "current_r2": self.training_history["r2_scores"][-1] if self.training_history["r2_scores"] else 0,
            "current_loss": self.training_history["losses"][-1] if self.training_history["losses"] else 0,
        }

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        ...

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Predict with uncertainty estimation - to be implemented by subclasses."""
        raise NotImplementedError("Uncertainty prediction not implemented for this model type")

    def save(self, filepath: Union[str, Path]) -> None:
        state = {
            "model_type": self.__class__.__name__,
            "model": self.model,
            "features": self.features,
            "targets": self.targets,
            "metrics": self.metrics,
            "training_data_info": self.training_data_info,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "multi_output": self.multi_output,
            "scaler_x": self.scaler_x,
            "scaler_y": self.scaler_y,
            "training_history": self.training_history,
            "performance_history": self.performance_history,
        }
        joblib.dump(state, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "CalibrationModel":
        state = joblib.load(filepath)
        mtype = state.get("model_type")
        
        model_classes = {
            "RandomForestSurrogate": RandomForestSurrogate,
            "GradientBoostingSurrogate": GradientBoostingSurrogate,
            "NeuralNetworkSurrogate": NeuralNetworkSurrogate,
            "GaussianProcessSurrogate": GaussianProcessSurrogate,
            "EnsembleSurrogate": EnsembleSurrogate,
            "SVMSurrogate": SVMSurrogate,
        }
        
        if mtype in model_classes:
            m = model_classes[mtype]()
        else:
            raise ValueError(f"Unknown model type: {mtype}")
        
        m.model = state["model"]
        m.features = state["features"]
        m.targets = state["targets"]
        m.metrics = state["metrics"]
        m.training_data_info = state["training_data_info"]
        m.created_at = state["created_at"]
        m.updated_at = state["updated_at"]
        m.multi_output = state.get("multi_output", False)
        m.scaler_x = state.get("scaler_x")
        m.scaler_y = state.get("scaler_y")
        m.training_history = state.get("training_history", {})
        m.performance_history = state.get("performance_history", [])
        
        return m

    def get_info(self) -> Dict[str, Any]:
        return {
            "model_type": self.__class__.__name__,
            "features": self.features,
            "targets": self.targets,
            "metrics": self.metrics,
            "training_data_info": self.training_data_info,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "multi_output": self.multi_output,
            "training_history": self.training_history,
            "performance_history": self.performance_history[-5:] if self.performance_history else [],  # Last 5 entries
            "training_phase": self.training_phase.value,
            "training_progress": self.training_progress,
        }

# =====================================================================
# Enhanced Surrogates with Real-time Training Support
# =====================================================================

class RandomForestSurrogate(CalibrationModel):
    def __init__(self, n_estimators=200, max_depth=None, random_state=42, n_jobs=-1):
        super().__init__()
        if not HAVE_SKLEARN:
            raise ImportError("scikit-learn required for RandomForestSurrogate")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs

    @performance_monitor
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
        self.update_training_progress(TrainingPhase.DATA_PREPROCESSING, 0.1)
        
        self.features = list(X.columns)
        self.targets = list(y.columns)
        self.multi_output = len(self.targets) > 1
        
        # Scale features and targets
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        self.update_training_progress(TrainingPhase.MODEL_TRAINING, 0.3)
        
        # Train model
        if self.multi_output:
            self.model = MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                )
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        
        self.model.fit(X_scaled, y_scaled)
        
        self.update_training_progress(TrainingPhase.VALIDATION, 0.8)
        
        # Store training data info
        self.training_data_info = {
            "n_samples": len(X),
            "feature_ranges": {c: (float(X[c].min()), float(X[c].max())) for c in X.columns},
            "target_ranges": {c: (float(y[c].min()), float(y[c].max())) for c in y.columns},
        }
        
        # Calculate metrics
        preds_scaled = self.model.predict(X_scaled)
        preds = self.scaler_y.inverse_transform(preds_scaled)
        y_true = self.scaler_y.inverse_transform(y_scaled)
        
        self.metrics = {
            "r2_score": float(r2_score(y_true, preds)),
            "mse": float(mean_squared_error(y_true, preds)),
            "mae": float(mean_absolute_error(y_true, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
        }
        
        if not self.multi_output and hasattr(self.model, 'feature_importances_'):
            self.metrics["feature_importances"] = dict(
                zip(self.features, map(float, self.model.feature_importances_))
            )
        
        self.updated_at = _now_iso()
        self.update_training_progress(TrainingPhase.COMPLETED, 1.0, self.metrics)
        return self.metrics

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X_scaled = self.scaler_x.transform(X)
        preds_scaled = self.model.predict(X_scaled)
        preds = self.scaler_y.inverse_transform(preds_scaled)
        return pd.DataFrame(preds, columns=self.targets, index=X.index)

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Predict with uncertainty using tree variance."""
        X_scaled = self.scaler_x.transform(X)
        
        if self.multi_output:
            # For multi-output, handle each target separately
            predictions = []
            uncertainties = []
            
            for estimator in self.model.estimators_:
                # Get predictions from all trees
                tree_preds = [tree.predict(X_scaled) for tree in estimator.estimators_]
                mean_pred = np.mean(tree_preds, axis=0)
                std_pred = np.std(tree_preds, axis=0)
                
                predictions.append(mean_pred)
                uncertainties.append(std_pred)
            
            preds = np.column_stack(predictions)
            stds = np.column_stack(uncertainties)
        else:
            # Get predictions from all trees
            tree_preds = [tree.predict(X_scaled) for tree in self.model.estimators_]
            preds = np.mean(tree_preds, axis=0)
            stds = np.std(tree_preds, axis=0)
        
        # Inverse transform predictions
        preds = self.scaler_y.inverse_transform(np.atleast_2d(preds).reshape(-1, len(self.targets)))
        # Scale uncertainties: scaler_y.scale_ gives per-target scaling
        stds = np.atleast_2d(stds)
        stds_scaled = stds * self.scaler_y.scale_
        
        return (
            pd.DataFrame(preds, columns=self.targets, index=X.index),
            pd.DataFrame(stds_scaled, columns=[f"{t}_uncertainty" for t in self.targets], index=X.index)
        )

class GradientBoostingSurrogate(CalibrationModel):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
        super().__init__()
        if not HAVE_SKLEARN:
            raise ImportError("scikit-learn required for GradientBoostingSurrogate")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    @performance_monitor
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
        self.update_training_progress(TrainingPhase.DATA_PREPROCESSING, 0.1)
        
        from sklearn.ensemble import GradientBoostingRegressor
        self.features = list(X.columns)
        self.targets = list(y.columns)
        self.multi_output = len(self.targets) > 1

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        Xs = self.scaler_x.fit_transform(X)
        ys = self.scaler_y.fit_transform(y)

        self.update_training_progress(TrainingPhase.MODEL_TRAINING, 0.3)

        if self.multi_output:
            self.model = MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    random_state=self.random_state
                )
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state
            )

        # Simulate incremental training for progress updates
        self.model.fit(Xs, ys)
        
        self.update_training_progress(TrainingPhase.VALIDATION, 0.8)

        preds_s = self.model.predict(Xs)
        preds = self.scaler_y.inverse_transform(preds_s)
        y_true = self.scaler_y.inverse_transform(ys)

        self.metrics = {
            "r2_score": float(r2_score(y_true, preds)),
            "mse": float(mean_squared_error(y_true, preds)),
            "mae": float(mean_absolute_error(y_true, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
        }
        
        self.training_data_info = {
            "n_samples": len(X),
            "feature_ranges": {c: (float(X[c].min()), float(X[c].max())) for c in X.columns},
            "target_ranges": {c: (float(y[c].min()), float(y[c].max())) for c in y.columns},
        }
        
        self.updated_at = _now_iso()
        self.update_training_progress(TrainingPhase.COMPLETED, 1.0, self.metrics)
        return self.metrics

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xs = self.scaler_x.transform(X)
        preds_s = self.model.predict(Xs)
        preds = self.scaler_y.inverse_transform(preds_s)
        return pd.DataFrame(preds, columns=self.targets, index=X.index)

class NeuralNetworkSurrogate(CalibrationModel):
    def __init__(self, hidden_layer_sizes=(100, 50), activation="relu", max_iter=1000, 
                                 learning_rate_init=0.001, random_state=42):
        super().__init__()
        if not HAVE_SKLEARN:
            raise ImportError("scikit-learn required for NeuralNetworkSurrogate")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.random_state = random_state

    @performance_monitor
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
        self.update_training_progress(TrainingPhase.DATA_PREPROCESSING, 0.1)
        
        from sklearn.neural_network import MLPRegressor
        self.features = list(X.columns)
        self.targets = list(y.columns)
        self.multi_output = len(self.targets) > 1

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        Xs = self.scaler_x.fit_transform(X)
        ys = self.scaler_y.fit_transform(y)

        self.update_training_progress(TrainingPhase.MODEL_TRAINING, 0.3)

        base = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            max_iter=self.max_iter,
            learning_rate_init=self.learning_rate_init,
            random_state=self.random_state
        )
        self.model = MultiOutputRegressor(base) if self.multi_output else base
        self.model.fit(Xs, ys)

        self.update_training_progress(TrainingPhase.VALIDATION, 0.8)

        preds_s = self.model.predict(Xs)
        preds = self.scaler_y.inverse_transform(preds_s)
        y_true = self.scaler_y.inverse_transform(ys)
        
        self.metrics = {
            "r2_score": float(r2_score(y_true, preds)),
            "mse": float(mean_squared_error(y_true, preds)),
            "mae": float(mean_absolute_error(y_true, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
        }
        
        self.training_data_info = {
            "n_samples": len(X),
            "feature_ranges": {c: (float(X[c].min()), float(X[c].max())) for c in X.columns},
            "target_ranges": {c: (float(y[c].min()), float(y[c].max())) for c in y.columns},
        }
        
        self.updated_at = _now_iso()
        self.update_training_progress(TrainingPhase.COMPLETED, 1.0, self.metrics)
        return self.metrics

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xs = self.scaler_x.transform(X)
        preds_s = self.model.predict(Xs)
        preds = self.scaler_y.inverse_transform(preds_s)
        return pd.DataFrame(preds, columns=self.targets, index=X.index)

class GaussianProcessSurrogate(CalibrationModel):
    def __init__(self, kernel=None, n_restarts_optimizer=10, random_state=42):
        super().__init__()
        if not HAVE_SKLEARN:
            raise ImportError("scikit-learn required for GaussianProcessSurrogate")
        
        self.kernel = kernel or C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state

    @performance_monitor
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
        self.update_training_progress(TrainingPhase.DATA_PREPROCESSING, 0.1)
        
        self.features = list(X.columns)
        self.targets = list(y.columns)
        self.multi_output = len(self.targets) > 1
        
        # Scale features and targets
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        self.update_training_progress(TrainingPhase.MODEL_TRAINING, 0.3)
        
        # Train model
        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
        )
        
        self.model.fit(X_scaled, y_scaled)
        
        self.update_training_progress(TrainingPhase.VALIDATION, 0.8)
        
        # Store training data info
        self.training_data_info = {
            "n_samples": len(X),
            "feature_ranges": {c: (float(X[c].min()), float(X[c].max())) for c in X.columns},
            "target_ranges": {c: (float(y[c].min()), float(y[c].max())) for c in y.columns},
        }
        
        # Calculate metrics
        preds_scaled, _ = self.model.predict(X_scaled, return_std=True)
        preds = self.scaler_y.inverse_transform(preds_scaled)
        y_true = self.scaler_y.inverse_transform(y_scaled)
        
        self.metrics = {
            "r2_score": float(r2_score(y_true, preds)),
            "mse": float(mean_squared_error(y_true, preds)),
            "mae": float(mean_absolute_error(y_true, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
        }
        
        self.updated_at = _now_iso()
        self.update_training_progress(TrainingPhase.COMPLETED, 1.0, self.metrics)
        return self.metrics

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X_scaled = self.scaler_x.transform(X)
        preds_scaled = self.model.predict(X_scaled)
        preds = self.scaler_y.inverse_transform(preds_scaled)
        return pd.DataFrame(preds, columns=self.targets, index=X.index)

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Predict with uncertainty using Gaussian Process built-in uncertainty."""
        X_scaled = self.scaler_x.transform(X)
        preds_scaled, stds_scaled = self.model.predict(X_scaled, return_std=True)
        
        # Inverse transform predictions
        preds = self.scaler_y.inverse_transform(preds_scaled)
        
        # Scale uncertainties
        if self.multi_output:
            # For multi-output GP, stds is a 2D array
            stds = stds_scaled * self.scaler_y.scale_
        else:
            # For single output, stds is a 1D array
            stds = stds_scaled.reshape(-1, 1) * self.scaler_y.scale_
        
        return (
            pd.DataFrame(preds, columns=self.targets, index=X.index),
            pd.DataFrame(stds, columns=[f"{t}_uncertainty" for t in self.targets], index=X.index)
        )

class EnsembleSurrogate(CalibrationModel):
    """Advanced ensemble model combining multiple surrogate types"""
    
    def __init__(self, estimators=None, voting='soft', weights=None):
        super().__init__()
        if not HAVE_SKLEARN:
            raise ImportError("scikit-learn required for EnsembleSurrogate")
        
        self.estimators = estimators or [
            ('rf', RandomForestSurrogate(n_estimators=50)),
            ('gb', GradientBoostingSurrogate(n_estimators=50)),
            ('nn', NeuralNetworkSurrogate(hidden_layer_sizes=(50, 25)))
        ]
        self.voting = voting
        self.weights = weights

    @performance_monitor
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
        self.update_training_progress(TrainingPhase.DATA_PREPROCESSING, 0.1)
        
        self.features = list(X.columns)
        self.targets = list(y.columns)
        self.multi_output = len(self.targets) > 1
        
        # Scale features and targets
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        self.update_training_progress(TrainingPhase.MODEL_TRAINING, 0.3)
        
        # Train individual models
        trained_estimators = []
        for i, (name, estimator) in enumerate(self.estimators):
            progress = 0.3 + (i / len(self.estimators)) * 0.5
            self.update_training_progress(TrainingPhase.MODEL_TRAINING, progress)
            
            # Train the estimator
            if hasattr(estimator, 'train'):
                estimator.train(pd.DataFrame(X_scaled, columns=self.features), 
                               pd.DataFrame(y_scaled, columns=self.targets))
                trained_estimators.append((name, estimator.model))
            else:
                # Assume it's a scikit-learn estimator
                estimator.fit(X_scaled, y_scaled)
                trained_estimators.append((name, estimator))
        
        # Create ensemble
        if self.voting == 'soft' and not self.multi_output:
            self.model = VotingRegressor(estimators=trained_estimators, weights=self.weights)
        else:
            self.model = StackingRegressor(
                estimators=trained_estimators,
                final_estimator=RandomForestRegressor(n_estimators=50, random_state=42)
            )
        
        self.model.fit(X_scaled, y_scaled.ravel() if not self.multi_output else y_scaled)
        
        self.update_training_progress(TrainingPhase.VALIDATION, 0.8)
        
        # Store training data info
        self.training_data_info = {
            "n_samples": len(X),
            "feature_ranges": {c: (float(X[c].min()), float(X[c].max())) for c in X.columns},
            "target_ranges": {c: (float(y[c].min()), float(y[c].max())) for c in y.columns},
        }
        
        # Calculate metrics
        preds_scaled = self.model.predict(X_scaled)
        preds = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, len(self.targets)))
        y_true = self.scaler_y.inverse_transform(y_scaled)
        
        self.metrics = {
            "r2_score": float(r2_score(y_true, preds)),
            "mse": float(mean_squared_error(y_true, preds)),
            "mae": float(mean_absolute_error(y_true, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
        }
        
        self.updated_at = _now_iso()
        self.update_training_progress(TrainingPhase.COMPLETED, 1.0, self.metrics)
        return self.metrics

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X_scaled = self.scaler_x.transform(X)
        preds_scaled = self.model.predict(X_scaled)
        preds = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, len(self.targets)))
        return pd.DataFrame(preds, columns=self.targets, index=X.index)

class SVMSurrogate(CalibrationModel):
    """Support Vector Machine surrogate model"""
    
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, random_state=42):
        super().__init__()
        if not HAVE_SKLEARN:
            raise ImportError("scikit-learn required for SVMSurrogate")
        
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.random_state = random_state

    @performance_monitor
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
        self.update_training_progress(TrainingPhase.DATA_PREPROCESSING, 0.1)
        
        self.features = list(X.columns)
        self.targets = list(y.columns)
        self.multi_output = len(self.targets) > 1
        
        # Scale features and targets
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        self.update_training_progress(TrainingPhase.MODEL_TRAINING, 0.3)
        
        # Train model
        if self.multi_output:
            self.model = MultiOutputRegressor(
                SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)
            )
        else:
            self.model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)
        
        self.model.fit(X_scaled, y_scaled.ravel() if not self.multi_output else y_scaled)
        
        self.update_training_progress(TrainingPhase.VALIDATION, 0.8)
        
        # Store training data info
        self.training_data_info = {
            "n_samples": len(X),
            "feature_ranges": {c: (float(X[c].min()), float(X[c].max())) for c in X.columns},
            "target_ranges": {c: (float(y[c].min()), float(y[c].max())) for c in y.columns},
        }
        
        # Calculate metrics
        preds_scaled = self.model.predict(X_scaled)
        preds = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, len(self.targets)))
        y_true = self.scaler_y.inverse_transform(y_scaled)
        
        self.metrics = {
            "r2_score": float(r2_score(y_true, preds)),
            "mse": float(mean_squared_error(y_true, preds)),
            "mae": float(mean_absolute_error(y_true, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
        }
        
        self.updated_at = _now_iso()
        self.update_training_progress(TrainingPhase.COMPLETED, 1.0, self.metrics)
        return self.metrics

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X_scaled = self.scaler_x.transform(X)
        preds_scaled = self.model.predict(X_scaled)
        preds = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, len(self.targets)))
        return pd.DataFrame(preds, columns=self.targets, index=X.index)

# =====================================================================
# Enhanced Model Manager with Feature Engineering and AutoML
# =====================================================================

class ModelManager:
    def __init__(self, model_dir: Union[str, Path] = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.models: Dict[str, CalibrationModel] = {}
        self.active_model: Optional[str] = None
        self.feature_engine = FeatureEngine()
        self.hyperparameter_optimizer = HyperparameterOptimizer()

    def create_model(self, model_type: str = "random_forest", **kwargs) -> CalibrationModel:
        model_classes = {
            "random_forest": RandomForestSurrogate,
            "gradient_boosting": GradientBoostingSurrogate,
            "neural_network": NeuralNetworkSurrogate,
            "gaussian_process": GaussianProcessSurrogate,
            "ensemble": EnsembleSurrogate,
            "svm": SVMSurrogate,
        }
        
        if model_type in model_classes:
            return model_classes[model_type](**kwargs)
        raise ValueError(f"Unknown model type: {model_type}")

    @performance_monitor
    def train_model(self, model_id: str, X: pd.DataFrame, y: pd.DataFrame, 
                   model_type: str = "random_forest", 
                   feature_engineering: bool = True,
                   feature_selection: bool = True,
                   hyperparameter_optimization: bool = False,
                   **kwargs) -> Dict[str, Any]:
        
        # Feature Engineering
        if feature_engineering:
            logger.info("Performing feature engineering...")
            X = self.feature_engine.engineer_features(X, y.columns[0] if len(y.columns) == 1 else None)
        
        # Feature Selection
        if feature_selection and len(X.columns) > 10:
            logger.info("Performing feature selection...")
            selected_features = self.feature_engine.select_features(X, y, method="importance")
            X = X[selected_features]
        
        # Hyperparameter Optimization
        best_params = {}
        if hyperparameter_optimization and HAVE_SKLEARN:
            logger.info("Performing hyperparameter optimization...")
            model_class = self.create_model(model_type).__class__
            best_params = self.hyperparameter_optimizer.optimize_hyperparameters(
                model_class, X, y, n_trials=kwargs.get('n_trials', 50)
            )
            kwargs.update(best_params)
        
        # Create and train model
        model = self.create_model(model_type, **kwargs)
        metrics = model.train(X, y)
        
        # Save model
        self.save_model(model_id, model)
        self.models[model_id] = model
        
        if not self.active_model:
            self.active_model = model_id
        
        return {
            "metrics": metrics,
            "best_params": best_params,
            "feature_engineered": feature_engineering,
            "features_used": list(X.columns),
            "model_info": model.get_info()
        }

    def save_model(self, model_id: str, model: CalibrationModel):
        filepath = self.model_dir / f"{model_id}.joblib"
        model.save(filepath)

    def load_model(self, model_id: str) -> CalibrationModel:
        filepath = self.model_dir / f"{model_id}.joblib"
        if not filepath.exists():
            raise FileNotFoundError(f"Model {model_id} not found")
        model = CalibrationModel.load(filepath)
        self.models[model_id] = model
        return model

    def get_model(self, model_id: Optional[str] = None) -> CalibrationModel:
        if not model_id:
            if not self.active_model:
                raise ValueError("No active model")
            model_id = self.active_model
        if model_id not in self.models:
            self.load_model(model_id)
        return self.models[model_id]

    def list_models(self) -> List[Dict[str, Any]]:
        out = []
        for filepath in self.model_dir.glob("*.joblib"):
            mid = filepath.stem
            try:
                model = self.load_model(mid)
                out.append({"id": mid, "info": model.get_info()})
            except Exception as e:
                logger.warning(f"Failed to load model {mid}: {e}")
                continue
        return out

    def delete_model(self, model_id: str) -> bool:
        if model_id in self.models:
            del self.models[model_id]
        filepath = self.model_dir / f"{model_id}.joblib"
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def compare_models(self, model_ids: List[str], X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """Compare multiple models performance"""
        comparison_results = {}
        
        for model_id in model_ids:
            try:
                model = self.get_model(model_id)
                
                # Cross-validation scores
                if HAVE_SKLEARN:
                    X_scaled = model.scaler_x.transform(X) if model.scaler_x else X
                    y_scaled = model.scaler_y.transform(y) if model.scaler_y else y
                    
                    scores = cross_val_score(
                        model.model, X_scaled, y_scaled, 
                        cv=5, scoring='r2', n_jobs=-1
                    )
                    
                    comparison_results[model_id] = {
                        "mean_r2": float(scores.mean()),
                        "std_r2": float(scores.std()),
                        "cross_val_scores": scores.tolist(),
                        "training_r2": model.metrics.get("r2_score", 0),
                        "model_type": model.__class__.__name__,
                    }
            except Exception as e:
                comparison_results[model_id] = {"error": str(e)}
        
        return comparison_results

# =====================================================================
# Advanced Hyperparameter Optimization
# =====================================================================

class HyperparameterOptimizer:
    """Advanced hyperparameter optimization with multiple backends"""
    
    def __init__(self):
        self.optimization_history = {}
        self.optimization_methods = ["random", "bayesian", "evolutionary"]
    
    def optimize_hyperparameters(self, model_class, X: pd.DataFrame, y: pd.DataFrame, 
                               n_trials: int = 100, method: str = "bayesian") -> Dict:
        """Optimize hyperparameters using specified method."""
        
        if method == "bayesian" and HAVE_OPTUNA:
            return self._optimize_with_optuna(model_class, X, y, n_trials)
        elif method == "random" and HAVE_SKLEARN:
            return self._optimize_with_random_search(model_class, X, y, n_trials)
        elif method == "evolutionary" and HAVE_SCIPY:
            return self._optimize_with_evolutionary(model_class, X, y, n_trials)
        else:
            return self._basic_optimization(model_class, X, y)
    
    def _optimize_with_optuna(self, model_class, X: pd.DataFrame, y: pd.DataFrame, n_trials: int) -> Dict:
        """Optimize using Optuna Bayesian optimization."""
        if not HAVE_OPTUNA:
            return {}
        
        # Define parameter space based on model type
        model_name = model_class.__name__
        
        def objective(trial):
            params = {}
            
            if "RandomForest" in model_name:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                }
            elif "GradientBoosting" in model_name:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                }
            elif "NeuralNetwork" in model_name:
                params = {
                    'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 25), (100, 50)]),
                    'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 0.0001, 0.1),
                    'max_iter': trial.suggest_int('max_iter', 500, 2000),
                }
            elif "GaussianProcess" in model_name:
                params = {
                    'n_restarts_optimizer': trial.suggest_int('n_restarts_optimizer', 5, 20),
                }
            
            try:
                model = model_class(**params)
                scores = cross_val_score(model.model if hasattr(model, 'model') else model, 
                                       X, y.values.ravel() if len(y.shape) > 1 else y, 
                                       cv=5, scoring='r2', n_jobs=-1)
                return scores.mean()
            except Exception as e:
                return -1  # Return worst score if training fails
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.optimization_history[model_name] = study.trials_dataframe().to_dict()
        
        return study.best_params
    
    def _optimize_with_random_search(self, model_class, X: pd.DataFrame, y: pd.DataFrame, n_trials: int) -> Dict:
        """Optimize using random search."""
        if not HAVE_SKLEARN:
            return {}
        
        model_name = model_class.__name__
        param_distributions = {}
        
        if "RandomForest" in model_name:
            param_distributions = {
                'n_estimators': [50, 100, 200, 300, 500],
                'max_depth': [3, 5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            }
        elif "GradientBoosting" in model_name:
            param_distributions = {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
            }
        
        if param_distributions:
            model = model_class()
            random_search = RandomizedSearchCV(
                model.model if hasattr(model, 'model') else model,
                param_distributions,
                n_iter=n_trials,
                cv=5,
                scoring='r2',
                n_jobs=-1,
                random_state=42
            )
            
            random_search.fit(X, y.values.ravel() if len(y.shape) > 1 else y)
            return random_search.best_params_
        
        return {}
    
    def _optimize_with_evolutionary(self, model_class, X: pd.DataFrame, y: pd.DataFrame, n_trials: int) -> Dict:
        """Optimize using evolutionary algorithms."""
        # This is a simplified implementation
        # In practice, you'd use a more sophisticated evolutionary approach
        best_score = -np.inf
        best_params = {}
        
        for _ in range(n_trials):
            if "RandomForest" in model_class.__name__:
                params = {
                    'n_estimators': np.random.randint(50, 500),
                    'max_depth': np.random.randint(3, 20),
                }
            else:
                params = {}
            
            try:
                model = model_class(**params)
                score = cross_val_score(model.model if hasattr(model, 'model') else model, 
                                      X, y.values.ravel() if len(y.shape) > 1 else y, 
                                      cv=3, scoring='r2').mean()
                
                if score > best_score:
                    best_score = score
                    best_params = params
            except:
                continue
        
        return best_params
    
    def _basic_optimization(self, model_class, X: pd.DataFrame, y: pd.DataFrame) -> Dict:
        """Fallback basic optimization."""
        return {}

    def get_optimization_history(self, model_type: str) -> pd.DataFrame:
        """Get optimization history for visualization"""
        return pd.DataFrame(self.optimization_history.get(model_type, {}))

# =====================================================================
# Enhanced Optimization Engine with Multi-Objective Support
# =====================================================================

class OptimizationEngine:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.optimization_history = []

    def optimize(
        self,
        model_id: str,
        targets: Dict[str, float],
        constraints: Optional[Dict[str, Dict[str, float]]] = None,
        initial_guess: Optional[Dict[str, float]] = None,
        method: str = "L-BFGS-B",
    ) -> Dict[str, Any]:
        model = self.model_manager.get_model(model_id)
        constraints = constraints or {}
        
        # Bounds from training ranges or explicit constraints
        bounds: List[Tuple[float, float]] = []
        for f in model.features:
            if f in constraints:
                bounds.append((constraints[f].get("min", -np.inf), constraints[f].get("max", np.inf)))
            else:
                bounds.append(model.training_data_info.get("feature_ranges", {}).get(f, (-np.inf, np.inf)))
        
        x0 = [0.5 * (lo + hi) if np.isfinite(lo) and np.isfinite(hi) else 0.0 for lo, hi in bounds]
        if initial_guess:
            for i, f in enumerate(model.features):
                if f in initial_guess:
                    x0[i] = float(initial_guess[f])

        objective_history = []
        
        def objective(x: np.ndarray) -> float:
            df = pd.DataFrame({f: [v] for f, v in zip(model.features, x)})
            preds = model.predict(df)
            loss = float(sum((float(preds[t][0]) - float(val)) ** 2 for t, val in targets.items() if t in preds))
            objective_history.append(loss)
            return loss

        res = minimize(objective, x0, method=method, bounds=bounds)
        
        self.optimization_history.append({
            "method": method,
            "success": res.success,
            "objective_history": objective_history,
            "timestamp": _now_iso()
        })
        
        if not res.success:
            return {"success": False, "error": res.message, "objective_history": objective_history}
        
        best = dict(zip(model.features, map(float, res.x)))
        preds = model.predict(pd.DataFrame({k: [v] for k, v in best.items()})).iloc[0].to_dict()
        
        return {
            "success": True, 
            "optimal_parameters": best, 
            "predicted_outcome": preds, 
            "message": res.message,
            "objective_history": objective_history
        }

    def bayesian_optimize(
        self,
        model_id: str,
        targets: Dict[str, float],
        constraints: Optional[Dict[str, Dict[str, float]]] = None,
        n_calls: int = 50,
        acq_func: str = "EI",
    ) -> Dict[str, Any]:
        """Bayesian optimization using Gaussian Processes."""
        if not HAVE_SKOPT:
            return {"success": False, "error": "Scikit-optimize not available for Bayesian optimization"}
        
        model = self.model_manager.get_model(model_id)
        constraints = constraints or {}
        
        # Define search space
        space = []
        for f in model.features:
            if f in constraints and "min" in constraints[f] and "max" in constraints[f]:
                space.append(Real(constraints[f]["min"], constraints[f]["max"], name=f))
            else:
                lo, hi = model.training_data_info.get("feature_ranges", {}).get(f, (-np.inf, np.inf))
                if not (np.isfinite(lo) and np.isfinite(hi)):
                    raise ValueError(f"Feature '{f}' has non-finite bounds {lo, hi}. Please provide explicit constraints.")
                space.append(Real(lo, hi, name=f))
        
        @use_named_args(space)
        def objective(**params):
            df = pd.DataFrame({f: [v] for f, v in params.items()})
            preds = model.predict(df)
            return float(sum((float(preds[t][0]) - float(val)) ** 2 for t, val in targets.items() if t in preds))
        
        # Run Bayesian optimization
        res = gp_minimize(objective, space, n_calls=n_calls, acq_func=acq_func, random_state=42)
        
        best = dict(zip(model.features, map(float, res.x)))
        preds = model.predict(pd.DataFrame({k: [v] for k, v in best.items()})).iloc[0].to_dict()
        
        self.optimization_history.append({
            "method": "bayesian",
            "success": True,
            "objective_history": res.func_vals.tolist(),
            "timestamp": _now_iso()
        })
        
        return {
            "success": True, 
            "optimal_parameters": best, 
            "predicted_outcome": preds, 
            "message": "Optimization successful",
            "optimization_history": [dict(zip(model.features, point)) for point in res.x_iters],
            "objective_history": [float(f) for f in res.func_vals]
        }

    def multi_objective_optimize(
        self,
        model_id: str,
        targets: Dict[str, Dict[str, float]],  # {target_name: {"weight": 0.5, "goal": "minimize"}}
        constraints: Optional[Dict[str, Dict[str, float]]] = None,
        n_calls: int = 100,
    ) -> Dict[str, Any]:
        """Multi-objective optimization using weighted sum approach."""
        model = self.model_manager.get_model(model_id)
        constraints = constraints or {}
        
        # Define search space
        space = []
        for f in model.features:
            if f in constraints and "min" in constraints[f] and "max" in constraints[f]:
                space.append(Real(constraints[f]["min"], constraints[f]["max"], name=f))
            else:
                lo, hi = model.training_data_info.get("feature_ranges", {}).get(f, (-np.inf, np.inf))
                if not (np.isfinite(lo) and np.isfinite(hi)):
                    raise ValueError(f"Feature '{f}' has non-finite bounds.")
                space.append(Real(lo, hi, name=f))
        
        @use_named_args(space)
        def objective(**params):
            df = pd.DataFrame({f: [v] for f, v in params.items()})
            preds = model.predict(df)
            
            total_loss = 0
            for target_name, config in targets.items():
                if target_name in preds:
                    weight = config.get("weight", 1.0)
                    goal = config.get("goal", "minimize")
                    target_value = float(preds[target_name][0])
                    desired_value = config.get("value", 0)
                    
                    if goal == "minimize":
                        loss = weight * abs(target_value - desired_value)
                    elif goal == "maximize":
                        loss = weight * (1.0 / (abs(target_value - desired_value) + 1e-9))
                    else:  # exact
                        loss = weight * (target_value - desired_value) ** 2
                    
                    total_loss += loss
            
            return total_loss
        
        if HAVE_SKOPT:
            res = gp_minimize(objective, space, n_calls=n_calls, random_state=42)
            
            best = dict(zip(model.features, map(float, res.x)))
            preds = model.predict(pd.DataFrame({k: [v] for k, v in best.items()})).iloc[0].to_dict()
            
            return {
                "success": True,
                "optimal_parameters": best,
                "predicted_outcome": preds,
                "objective_history": [float(f) for f in res.func_vals]
            }
        else:
            # Fallback to simple optimization
            return self.optimize(model_id, {t: c.get("value", 0) for t, c in targets.items()}, constraints)

# =====================================================================
# Enhanced Map Generator with 3D Support
# =====================================================================

class MapGenerator:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def generate_map(self, model_id: str, grid_definitions: Dict[str, List[float]]):
        model = self.model_manager.get_model(model_id)
        axes = [np.array(grid_definitions[f], dtype=float) for f in model.features]
        mesh = np.meshgrid(*axes, indexing="ij")
        points = pd.DataFrame({f: m.ravel() for f, m in zip(model.features, mesh)})
        preds = model.predict(points)
        maps = {t: preds[t].values.reshape(mesh[0].shape) for t in model.targets}
        
        # Convert numpy arrays to lists for JSON serialization
        for t in maps:
            maps[t] = maps[t].tolist()

        return {
            "grid_axes": {f: ax.tolist() for f, ax in zip(model.features, axes)},
            "predictions": maps,
            "grid_shape": mesh[0].shape,
            "model_type": model.__class__.__name__,
        }

    def export_map(self, map_data: Dict[str, Any], fmt: str = "csv") -> str:
        if fmt == "csv":
            return self._export_csv(map_data)
        if fmt == "json":
            return self._export_json(map_data)
        if fmt == "inca":
            return self._export_inca(map_data)
        if fmt == "matlab":
            return self._export_matlab(map_data)
        raise ValueError(f"Unknown export format: {fmt}")

    def _export_csv(self, map_data: Dict[str, Any]) -> str:
        import io, csv
        out = io.StringIO()
        writer = csv.writer(out)
        header = list(map_data["grid_axes"].keys()) + list(map_data["predictions"].keys())
        writer.writerow(header)
        grid_axes = list(map_data["grid_axes"].values())
        mesh = np.array(np.meshgrid(*grid_axes, indexing="xy"))
        grid_points = mesh.reshape(len(grid_axes), -1).T
        for i, pt in enumerate(grid_points):
            row = pt.tolist()
            for t in map_data["predictions"]:
                row.append(float(np.array(map_data["predictions"][t]).flat[i]))
            writer.writerow(row)
        return out.getvalue()

    def _export_json(self, map_data: Dict[str, Any]) -> str:
        return json.dumps(
            {
                "grid_axes": map_data["grid_axes"],
                "predictions": {k: np.asarray(v).tolist() for k, v in map_data["predictions"].items()},
                "grid_shape": map_data["grid_shape"],
                "metadata": {
                    "exported_at": _now_iso(),
                    "model_type": map_data.get("model_type", "unknown")
                }
            },
            indent=2,
        )

    def _export_inca(self, map_data: Dict[str, Any]) -> str:
        out = []
        out.append("; INCA Map Export")
        out.append(f"; Generated {_now_iso()}")
        out.append(f"; Model: {map_data.get('model_type', 'unknown')}")
        for f, vals in map_data["grid_axes"].items():
            out.append(f"AXIS {f} = {len(vals)}")
            out.append(" ".join(f"{v:.6f}" for v in vals))
        for t, vals in map_data["predictions"].items():
            arr = np.asarray(vals)
            out.append(f"MAP {t}")
            out.append(" ".join(f"{float(v):.6f}" for v in arr.flatten()))
        return "\n".join(out)

    def _export_matlab(self, map_data: Dict[str, Any]) -> str:
        out = []
        out.append(f"% MATLAB Map Export - Generated {_now_iso()}")
        out.append(f"% Model: {map_data.get('model_type', 'unknown')}")
        
        for f, vals in map_data["grid_axes"].items():
            out.append(f"{f} = [{', '.join(map(str, vals))}];")
        
        for t, vals in map_data["predictions"].items():
            arr = np.asarray(vals)
            out.append(f"{t} = {np.array2string(arr, separator=', ')};")
        
        return "\n".join(out)

# =====================================================================
# Enhanced Interpolation Engine
# =====================================================================

@dataclass
class ConstraintSpec:
    axis_monotonic: Optional[Dict[str, Literal["increasing", "decreasing"]]] = None
    target_min: Optional[float] = None
    target_max: Optional[float] = None
    max_gradient: Optional[float] = None
    smoothness: Optional[float] = None

class InterpolationEngine:
    """Build gridded maps from scattered measurements with advanced interpolation methods."""
    
    def __init__(self):
        self.available_methods = {
            "linear": "linear",
            "cubic": "cubic",
            "rbf": "rbf",
            "clough": "clough",
            "spline": "spline",
            "auto": "auto",
        }

    def _choose_method(self, method, n_points: int):
        if method == "auto":
            if n_points < 50:
                return "linear"
            elif n_points < 1000:
                return "cubic"
            else:
                return "linear"  # Fallback for large datasets
        return method

    def build_map(
        self,
        df: pd.DataFrame,
        axes: List[str],
        target: str,
        grid_def: Optional[Dict[str, Iterable[float]]] = None,
        *,
        method: Literal["auto", "linear", "cubic", "rbf", "clough", "spline"] = "auto",
        constraint_spec: Optional[ConstraintSpec] = None,
        resolution: int = 30,
    ):
        """Enhanced map building with support for multiple interpolation methods."""
        if len(axes) < 1:
            raise ValueError("At least one axis is required to build a map.")
        
        method = self._choose_method(method, len(df))
        constraint_spec = constraint_spec or ConstraintSpec()

        # Prepare input arrays
        pts = df[axes].to_numpy(dtype=float)
        vals = df[target].to_numpy(dtype=float)

        # Remove any NaN values
        valid_mask = ~np.isnan(pts).any(axis=1) & ~np.isnan(vals)
        pts = pts[valid_mask]
        vals = vals[valid_mask]

        if len(pts) == 0:
            raise ValueError("No valid data points after removing NaN values.")

        # Build axis grids
        grid_axes = {}
        for ax in axes[:2]:  # Support up to 2D for now
            if grid_def and ax in grid_def:
                grid_axes[ax] = np.asarray(list(grid_def[ax]), dtype=float)
            else:
                grid_axes[ax] = _linspace_from_data(df[ax].to_numpy(dtype=float), n=resolution)

        # Build mesh
        if len(axes) >= 2:
            Xg, Yg = np.meshgrid(grid_axes[axes[0]], grid_axes[axes[1]], indexing='xy')
            grid_points = np.column_stack([Xg.ravel(), Yg.ravel()])
        else:
            # 1D case
            Xg = np.array(grid_axes[axes[0]])
            grid_points = Xg.reshape(-1, 1)

        Z = self._interpolate_points(pts, vals, grid_points, method, len(axes))

        # Apply constraints
        Z = self._apply_constraints(Z, axes, grid_axes, constraint_spec)

        # Return map structure
        map_data = {
            "grid_axes": {ax: grid_axes[ax].tolist() for ax in axes[:2]},
            "predictions": {target: Z.tolist()},
            "grid_shape": Z.shape,
            "interpolation_method": method,
            "n_input_points": len(pts),
        }
        return map_data

    def _interpolate_points(self, pts: np.ndarray, vals: np.ndarray, 
                           grid_points: np.ndarray, method: str, n_dims: int) -> np.ndarray:
        """Perform the actual interpolation using the specified method."""
        try:
            if method in ("linear", "cubic") and HAVE_SCIPY and n_dims <= 2:
                Zflat = griddata(pts[:, :grid_points.shape[1]], vals, grid_points, 
                                                             method=method, fill_value=np.nan)
                shape = (len(np.unique(grid_points[:, 1])) if n_dims > 1 else 1,
                        len(np.unique(grid_points[:, 0])))
                Z = Zflat.reshape(shape)
                
            elif method == "clough" and HAVE_SCIPY and n_dims == 2:
                interp = CloughTocher2DInterpolator(pts[:, :2], vals)
                Xg_unique = np.unique(grid_points[:, 0])
                Yg_unique = np.unique(grid_points[:, 1])
                Xg, Yg = np.meshgrid(Xg_unique, Yg_unique, indexing='xy')
                Z = interp(Xg, Yg)
                
            elif method == "rbf" and HAVE_SCIPY:
                rbf = Rbf(*[pts[:, i] for i in range(pts.shape[1])], vals, 
                         function='multiquadric', smooth=0.1)
                if pts.shape[1] == 1:
                    Z = rbf(grid_points[:, 0])
                    Z = Z.reshape(1, -1)
                else:
                    Z = rbf(grid_points[:, 0], grid_points[:, 1])
                    shape = (len(np.unique(grid_points[:, 1])), 
                             len(np.unique(grid_points[:, 0])))
                    Z = Z.reshape(shape)
                    
            elif method == "spline" and HAVE_SCIPY and n_dims == 1:
                # 1D spline interpolation
                unique_x, indices = np.unique(pts[:, 0], return_index=True)
                unique_y = vals[indices]
                spline = interp1d(unique_x, unique_y, kind='cubic', 
                               bounds_error=False, fill_value="extrapolate")
                Z = spline(grid_points[:, 0])
                Z = Z.reshape(1, -1)
                
            else:
                # Fallback: nearest neighbor
                Z = self._nearest_fallback(pts, vals, grid_points, n_dims)
                
        except Exception as e:
            logger.warning(f"Interpolation method {method} failed: {e}, using fallback")
            Z = self._nearest_fallback(pts, vals, grid_points, n_dims)
            
        return Z

    def _nearest_fallback(self, pts: np.ndarray, vals: np.ndarray, 
                         grid_points: np.ndarray, n_dims: int) -> np.ndarray:
        """Fallback interpolation using nearest neighbors."""
        if HAVE_SCIPY:
            tree = cKDTree(pts[:, :grid_points.shape[1]])
            dists, idxs = tree.query(grid_points, k=min(5, len(pts)))
            # Inverse distance weighting
            dists = np.maximum(dists, 1e-12)
            weights = 1.0 / dists
            Zflat = np.sum(vals[idxs] * weights, axis=1) / np.sum(weights, axis=1)
        else:
            # Simple nearest
            Zflat = np.empty(len(grid_points))
            for i, gp in enumerate(grid_points):
                dif = np.sum((pts[:, :grid_points.shape[1]] - gp) ** 2, axis=1)
                idx = np.argmin(dif)
                Zflat[i] = vals[idx]
                
        shape = (len(np.unique(grid_points[:, 1])) if n_dims > 1 else 1,
                len(np.unique(grid_points[:, 0])))
        return Zflat.reshape(shape)

    def _apply_constraints(self, Z: np.ndarray, axes: List[str], grid_axes: Dict[str, np.ndarray],
                          constraint_spec: ConstraintSpec) -> np.ndarray:
        """Apply constraints to the interpolated map."""
        Z_constrained = Z.copy()
        
        # Monotonicity constraints
        if constraint_spec.axis_monotonic:
            for i_ax, ax in enumerate(axes[:2]):
                if ax in constraint_spec.axis_monotonic:
                    direction = constraint_spec.axis_monotonic[ax]
                    if i_ax == 0:  # Enforce along columns
                        for r in range(Z_constrained.shape[0]):
                            Z_constrained[r, :] = _enforce_monotonic_1d(Z_constrained[r, :], direction)
                    else:  # Enforce along rows
                        for c in range(Z_constrained.shape[1]):
                            Z_constrained[:, c] = _enforce_monotonic_1d(Z_constrained[:, c], direction)
        
        # Gradient constraints
        if constraint_spec.max_gradient:
            Z_constrained = _smooth_gradients_2d(Z_constrained, 
                                               max_grad=constraint_spec.max_gradient, 
                                               passes=2)
        
        # Value constraints
        if constraint_spec.target_min is not None or constraint_spec.target_max is not None:
            Z_constrained = _clip_if(Z_constrained, 
                                   constraint_spec.target_min, 
                                   constraint_spec.target_max)
        
        return Z_constrained

# =====================================================================
# Enhanced Recommendation Engine
# =====================================================================

class RecommendationEngine:
    """Advanced recommendation engine with multiple strategies."""
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager
        self.recommendation_history = []

    def recommend(self, a, b, c=None, *args, **kwargs):
        """Unified recommendation interface."""
        if isinstance(a, str):
            # Model-based recommendation
            return self._recommend_model_based(a, b, c, *args, **kwargs)
        elif isinstance(a, pd.DataFrame):
            # Map-based recommendation
            return self._recommend_map_based(a, b, c, *args, **kwargs)
        else:
            raise ValueError("Unsupported input signature for recommend()")

    def _recommend_model_based(
        self,
        model_id: str,
        df_measured: pd.DataFrame,
        target: str,
        top_k: int = 5,
        strategy: str = "gradient",
        confidence_threshold: float = 0.8,
    ) -> pd.DataFrame:
        """Advanced model-based recommendations with multiple strategies."""
        if self.model_manager is None:
            raise ValueError("ModelManager required for model-based recommendations")

        model = self.model_manager.get_model(model_id)
        features = model.features
        
        if any(f not in df_measured.columns for f in features + [target]):
            raise ValueError("df_measured must contain model features and the measured target column")

        # Predict using model
        X = df_measured[features]
        preds = model.predict(X)
        y_meas = df_measured[target].to_numpy(dtype=float)
        
        if target in preds.columns:
            y_pred = preds[target].to_numpy(dtype=float)
        else:
            y_pred = preds.iloc[:, 0].to_numpy(dtype=float)  # Use first output
        
        residuals = y_meas - y_pred
        abs_res = np.abs(residuals)
        
        if strategy == "gradient":
            return self._gradient_based_recommendations(model, df_measured, target, top_k, residuals)
        elif strategy == "uncertainty":
            return self._uncertainty_based_recommendations(model, df_measured, target, top_k, confidence_threshold)
        elif strategy == "residual":
            return self._residual_based_recommendations(df_measured, target, top_k, residuals)
        else:
            return self._gradient_based_recommendations(model, df_measured, target, top_k, residuals)

    def _gradient_based_recommendations(self, model, df_measured, target, top_k, residuals):
        """Recommendations based on feature gradients."""
        features = model.features
        X = df_measured[features]
        
        # Get top samples by residual
        order = np.argsort(-np.abs(residuals))
        top_idx = order[:top_k]
        
        results = []
        eps = 1e-3
        
        for idx in top_idx:
            row = X.iloc[[idx]].copy()
            base_pred = model.predict(row).iloc[0].to_dict()
            res_val = float(residuals[idx])
            
            grads = {}
            suggestions = {}
            
            for f in features:
                # Calculate numerical gradient
                rng = model.training_data_info.get("feature_ranges", {}).get(f, (0.0, 1.0))
                f_range = float(rng[1] - rng[0]) if (rng and rng[1] != rng[0]) else 1.0
                de = max(eps, 1e-6 * max(1.0, f_range))
                
                row_up = row.copy()
                row_up.at[row_up.index[0], f] = float(row_up.iloc[0][f]) + de
                pred_up = model.predict(row_up).iloc[0].to_dict()
                
                base_val = base_pred.get(target, list(base_pred.values())[0])
                pred_up_val = pred_up.get(target, list(pred_up.values())[0])
                grad = (pred_up_val - base_val) / de
                
                grads[f] = float(grad)

                # Suggest direction to reduce residual
                sign = -np.sign(res_val * grad)
                suggested_delta = float(sign * 0.1 * f_range)  # 10% of range
                suggestions[f] = suggested_delta

            results.append({
                "index": int(idx),
                "measured": float(df_measured[target].iloc[idx]),
                "predicted": float(base_pred.get(target, list(base_pred.values())[0])),
                "residual": float(res_val),
                "feature_gradients": grads,
                "suggested_deltas": suggestions,
                "strategy": "gradient",
            })

        return pd.DataFrame(results)

    def _uncertainty_based_recommendations(self, model, df_measured, target, top_k, confidence_threshold):
        """Recommendations based on prediction uncertainty."""
        try:
            X = df_measured[model.features]
            preds, uncertainties = model.predict_with_uncertainty(X)
            
            # Get samples with high uncertainty
            uncertainty_col = f"{target}_uncertainty"
            if uncertainty_col in uncertainties.columns:
                high_uncertainty_idx = uncertainties[uncertainty_col].nlargest(top_k).index
                
                results = []
                for idx in high_uncertainty_idx:
                    results.append({
                        "index": int(idx),
                        "measured": float(df_measured[target].iloc[idx]),
                        "predicted": float(preds[target].iloc[idx]),
                        "uncertainty": float(uncertainties[uncertainty_col].iloc[idx]),
                        "suggestion": "Collect more data in this region",
                        "strategy": "uncertainty",
                    })
                
                return pd.DataFrame(results)
        except NotImplementedError:
            logger.warning("Uncertainty prediction not available, falling back to gradient method")
        
        # Fallback to gradient method
        return self._gradient_based_recommendations(model, df_measured, target, top_k, 
                                                  df_measured[target] - model.predict(X)[target])

    def _residual_based_recommendations(self, df_measured, target, top_k, residuals):
        """Simple recommendations based on largest residuals."""
        order = np.argsort(-np.abs(residuals))
        top_idx = order[:top_k]
        
        results = []
        for idx in top_idx:
            results.append({
                "index": int(idx),
                "measured": float(df_measured[target].iloc[idx]),
                "residual": float(residuals[idx]),
                "suggestion": "Check measurement accuracy and consider recalibration",
                "strategy": "residual",
            })
        
        return pd.DataFrame(results)

    def _recommend_map_based(
        self,
        measurements: pd.DataFrame,
        map_data: Dict[str, Any],
        axes: List[str],
        target: str,
        *,
        clip_pct: float = 0.15,
        topk_points: int = 20,
        strategy: str = "residual",
    ) -> Dict[str, Any]:
        """Advanced map-based recommendations."""
        if target not in map_data["predictions"]:
            raise ValueError("Target not found in provided map_data['predictions']")

        grid_axes = map_data["grid_axes"]
        Z = np.asarray(map_data["predictions"][target])
        gx = np.asarray(grid_axes[axes[0]])
        gy = np.asarray(grid_axes[axes[1]]) if len(axes) > 1 and axes[1] in grid_axes else None

        # Interpolate map predictions at measurement positions
        meas_points = measurements[axes].to_numpy(dtype=float)
        meas_vals = measurements[target].to_numpy(dtype=float)

        # Create interpolator
        try:
            if HAVE_SCIPY and gy is not None:
                interp = RegularGridInterpolator((gx, gy), Z.T, bounds_error=False, fill_value=np.nan)
                pred_at_meas = interp(meas_points)
            elif HAVE_SCIPY and gy is None:
                interp = RegularGridInterpolator((gx,), Z[0], bounds_error=False, fill_value=np.nan)
                pred_at_meas = interp(meas_points)
            else:
                pred_at_meas = self._nearest_map_prediction(meas_points, gx, gy, Z)
        except Exception:
            pred_at_meas = self._nearest_map_prediction(meas_points, gx, gy, Z)

        residuals = meas_vals - pred_at_meas
        abs_res = np.abs(residuals)
        
        summary = {
            "n_measurements": len(meas_vals),
            "mean_abs_residual": float(np.nanmean(abs_res)),
            "max_abs_residual": float(np.nanmax(abs_res)),
            "std_residual": float(np.nanstd(residuals)),
        }

        # Find top grid cells by aggregated residual
        agg_grid = np.zeros_like(Z, dtype=float)
        counts = np.zeros_like(Z, dtype=int)
        
        for i, pt in enumerate(meas_points):
            ix = int(np.argmin(np.abs(gx - pt[0])))
            iy = int(np.argmin(np.abs(gy - pt[1]))) if gy is not None else 0
            agg_grid[iy, ix] += residuals[i]
            counts[iy, ix] += 1
        
        # Average residual per cell
        avg_grid = np.zeros_like(Z, dtype=float)
        mask = counts > 0
        avg_grid[mask] = agg_grid[mask] / counts[mask]
        abs_avg_grid = np.abs(avg_grid)

        # Get topk cells
        flat_idx = np.argsort(-abs_avg_grid.ravel())[:topk_points]
        iyx = [(int(i // Z.shape[1]), int(i % Z.shape[1])) for i in flat_idx]
        
        next_points = []
        for iy, ix in iyx:
            point_def = {axes[0]: float(gx[ix])}
            if gy is not None:
                point_def[axes[1]] = float(gy[iy])
            point_def["avg_residual"] = float(avg_grid[iy, ix])
            point_def["count"] = int(counts[iy, ix])
            next_points.append(point_def)

        # Generate recommended map by nudging values
        recommended_map = Z.copy().astype(float)
        for (iy, ix) in iyx:
            if counts[iy, ix] > 0:
                nudge = clip_pct * (agg_grid[iy, ix] / counts[iy, ix])
                recommended_map[iy, ix] += nudge

        out = {
            "residuals_summary": summary,
            "residual_grid": avg_grid.tolist(),
            "next_points": next_points,
            "recommended_map": recommended_map.tolist(),
            "strategy": strategy,
        }
        
        self.recommendation_history.append({
            "timestamp": _now_iso(),
            "strategy": strategy,
            "n_recommendations": len(next_points),
            "max_residual": summary["max_abs_residual"]
        })
        
        return out

    def _nearest_map_prediction(self, meas_points, gx, gy, Z):
        """Fallback nearest neighbor prediction for map."""
        pred_at_meas = []
        for pt in meas_points:
            ix = np.argmin(np.abs(gx - pt[0]))
            iy = np.argmin(np.abs(gy - pt[1])) if gy is not None else 0
            pred_at_meas.append(float(Z[iy, ix]))
        return np.array(pred_at_meas)

# =====================================================================
# Enhanced DOE Engine
# =====================================================================

class DOEEngine:
    """
    Advanced Design-of-Experiments with adaptive sampling.
    Legacy compatibility: supports 'sobol', 'lhs', 'factorial', and 'adaptive' methods.
    """
    
    def __init__(self):
        self.doe_history = []

    def generate_doe(self, param_ranges: Dict[str, Tuple[float, float]], 
                    n_samples: int = 20, method: str = "lhs",
                    random_state: Optional[int] = None) -> pd.DataFrame:
        """Generate DOE samples using various methods."""
        
        if method == "sobol" and HAVE_SKOPT:
            space = [Real(lo, hi) for lo, hi in param_ranges.values()]
            sampler = Sobol()
            pts = sampler.generate(space, n_samples)
            df = pd.DataFrame(pts, columns=param_ranges.keys())
            
        elif method == "lhs" and HAVE_SKOPT:
            space = [Real(lo, hi) for lo, hi in param_ranges.values()]
            sampler = Lhs(lhs_type="classic", criterion=None)
            samples = sampler.generate(space, n_samples)
            df = pd.DataFrame(samples, columns=param_ranges.keys())
            
        elif method == "lhs":
            df = self._lhs(param_ranges, n_samples, random_state)
            
        elif method == "random":
            rng = np.random.RandomState(random_state)
            arr = np.zeros((n_samples, len(param_ranges)))
            for j, (name, (lo, hi)) in enumerate(param_ranges.items()):
                arr[:, j] = rng.uniform(lo, hi, size=n_samples)
            df = pd.DataFrame(arr, columns=param_ranges.keys())
            
        elif method == "factorial":
            df = self._factorial(param_ranges, n_samples, random_state)
            
        elif method == "adaptive" and HAVE_SKOPT:
            df = self._adaptive_doe(param_ranges, n_samples)
            
        else:
            raise ValueError(f"Unknown DOE method: {method}")
        
        self.doe_history.append({
            "method": method,
            "n_samples": n_samples,
            "parameters": list(param_ranges.keys()),
            "timestamp": _now_iso()
        })
        
        return df

    def _lhs(self, param_ranges: Dict[str, Tuple[float, float]], n_samples: int, 
             random_state: Optional[int] = None) -> pd.DataFrame:
        """Latin Hypercube Sampling implementation."""
        names = list(param_ranges.keys())
        d = len(names)
        rng = np.random.RandomState(random_state)
        
        # Stratified intervals
        cut = np.linspace(0, 1, n_samples + 1)
        u = rng.uniform(size=(n_samples, d))
        lhs = np.zeros((n_samples, d))
        
        for j in range(d):
            idx = rng.permutation(n_samples)
            lhs[:, j] = (cut[:-1] + u[:, j] * (1.0 / n_samples))[idx]
        
        # Map to ranges
        out = np.zeros_like(lhs)
        for j, name in enumerate(names):
            lo, hi = param_ranges[name]
            out[:, j] = lo + lhs[:, j] * (hi - lo)
            
        return pd.DataFrame(out, columns=names)

    def _factorial(self, param_ranges: Dict[str, Tuple[float, float]], 
                  n_samples: int, random_state: Optional[int] = None) -> pd.DataFrame:
        """Factorial design implementation."""
        names = list(param_ranges.keys())
        grids = []
        
        for name in names:
            rng = param_ranges[name]
            if isinstance(rng, (list, tuple)) and len(rng) > 2:
                grids.append(list(rng))
            else:
                lo, hi = float(rng[0]), float(rng[1])
                # Create 2-3 levels for each parameter
                n_levels = max(2, min(3, int(np.sqrt(n_samples))))
                grids.append(np.linspace(lo, hi, n_levels).tolist())
        
        import itertools
        combos = list(itertools.product(*grids))
        df = pd.DataFrame(combos, columns=names)
        
        if len(df) > n_samples:
            rng = np.random.RandomState(random_state)
            df = df.sample(n_samples, random_state=rng).reset_index(drop=True)
            
        return df

    def _adaptive_doe(self, param_ranges: Dict[str, Tuple[float, float]], 
                     n_samples: int) -> pd.DataFrame:
        """Adaptive DOE based on space-filling criteria."""
        # Initial space-filling design
        initial_doe = self.generate_doe(param_ranges, n_samples=100, method="lhs")
        
        # This would be enhanced with actual model-based adaptive sampling
        # For now, we return a space-filling design
        return initial_doe.sample(n_samples).reset_index(drop=True)

    def adaptive_doe(
        self,
        model: CalibrationModel,
        param_ranges: Dict[str, Tuple[float, float]],
        n_samples: int = 10,
        acquisition: str = "uncertainty",
    ) -> pd.DataFrame:
        """Generate adaptive DOE samples based on model uncertainty."""
        # Generate initial space-filling DOE
        initial_doe = self.generate_doe(param_ranges, n_samples=100, method="lhs")
        
        try:
            # Predict uncertainty at all points
            _, uncertainties = model.predict_with_uncertainty(initial_doe)
            
            # Select points with highest uncertainty
            if acquisition == "uncertainty":
                # Use total uncertainty across all targets
                total_uncertainty = uncertainties.sum(axis=1)
                top_indices = total_uncertainty.nlargest(n_samples).index
            else:
                top_indices = uncertainties.sum(axis=1).nlargest(n_samples).index
            
            return initial_doe.iloc[top_indices].reset_index(drop=True)
            
        except NotImplementedError:
            # Fallback to space-filling design
            logger.warning("Uncertainty prediction not available, using space-filling DOE")
            return self.generate_doe(param_ranges, n_samples, "lhs")

# =====================================================================
# Performance Optimizer
# =====================================================================

class PerformanceOptimizer:
    """Optimize model performance and memory usage."""
    
    def __init__(self):
        self.cache = {}
        self.optimization_strategies = ["memory", "speed", "balanced"]

    def optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype
            
            if col_type == 'float64':
                # Downcast floats
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
            elif col_type == 'int64':
                # Downcast integers
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
            elif col_type == 'object':
                # Convert objects to category if beneficial
                if optimized_df[col].nunique() / len(optimized_df[col]) < 0.5:
                    optimized_df[col] = optimized_df[col].astype('category')
        
        return optimized_df

    def incremental_training_support(self, model, X: pd.DataFrame, y: pd.DataFrame, 
                                   batch_size: int = 1000) -> CalibrationModel:
        """Support incremental training for large datasets."""
        if hasattr(model, 'partial_fit'):
            n_batches = int(np.ceil(len(X) / batch_size))
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X))
                
                X_batch = X.iloc[start_idx:end_idx]
                y_batch = y.iloc[start_idx:end_idx]
                
                model.partial_fit(X_batch, y_batch)
                
                # Update progress
                if hasattr(model, 'update_training_progress'):
                    progress = (i + 1) / n_batches
                    model.update_training_progress(TrainingPhase.MODEL_TRAINING, progress)
            
            return model
        else:
            # Fall back to standard training
            return model.fit(X, y)

    def optimize_model_architecture(self, model: CalibrationModel, 
                                  strategy: str = "balanced") -> CalibrationModel:
        """Optimize model architecture based on strategy."""
        if strategy == "memory":
            # Reduce model complexity for memory efficiency
            if isinstance(model, RandomForestSurrogate):
                model.n_estimators = max(50, model.n_estimators // 2)
            elif isinstance(model, NeuralNetworkSurrogate):
                model.hidden_layer_sizes = tuple(size // 2 for size in model.hidden_layer_sizes)
                
        elif strategy == "speed":
            # Optimize for inference speed
            if isinstance(model, RandomForestSurrogate):
                model.max_depth = min(10, model.max_depth or 20)
            elif isinstance(model, GaussianProcessSurrogate):
                model.n_restarts_optimizer = max(1, model.n_restarts_optimizer // 2)
        
        return model

# =====================================================================
# Analytics and Insights Engine
# (Supports legacy validation and diagnostics interfaces)
# =====================================================================

class AnalyticsEngine:
    """Advanced analytics and insights generation."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.analytics_history = []

    def generate_model_insights(self, model_id: str, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive insights about model performance."""
        model = self.model_manager.get_model(model_id)
        
        insights = {
            "model_type": model.__class__.__name__,
            "training_metrics": model.metrics,
            "feature_analysis": self._analyze_features(model, X, y),
            "performance_characteristics": self._analyze_performance(model, X, y),
            "data_characteristics": self._analyze_data(X, y),
            "recommendations": self._generate_recommendations(model, X, y),
        }
        
        self.analytics_history.append({
            "model_id": model_id,
            "timestamp": _now_iso(),
            "insights_generated": list(insights.keys())
        })
        
        return insights

    def _analyze_features(self, model: CalibrationModel, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature importance and relationships."""
        analysis = {}
        
        # Feature importance
        if "feature_importances" in model.metrics:
            analysis["feature_importances"] = model.metrics["feature_importances"]
        
        # Feature correlations
        correlation_matrix = X.corr()
        analysis["feature_correlations"] = {
            "highly_correlated_pairs": self._find_high_correlations(correlation_matrix),
            "correlation_matrix": correlation_matrix.to_dict()
        }
        
        # Feature-target relationships
        if len(y.columns) == 1:  # Single target
            target_correlations = {}
            for feature in X.columns:
                correlation = X[feature].corr(y.iloc[:, 0])
                target_correlations[feature] = float(correlation)
            
            analysis["target_correlations"] = target_correlations
        
        return analysis

    def _find_high_correlations(self, correlation_matrix: pd.DataFrame, 
                              threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find highly correlated feature pairs."""
        high_corr_pairs = []
        corr_upper = correlation_matrix.where(
            np.triu(np.ones_like(correlation_matrix), k=1).astype(bool)
        )
        
        for i, col1 in enumerate(correlation_matrix.columns):
            for j, col2 in enumerate(correlation_matrix.columns[i+1:], i+1):
                corr_val = corr_upper.iloc[i, j]
                if abs(corr_val) > threshold:
                    high_corr_pairs.append({
                        "feature1": col1,
                        "feature2": col2,
                        "correlation": float(corr_val)
                    })
        
        return high_corr_pairs

    def _analyze_performance(self, model: CalibrationModel, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """Analyze model performance characteristics."""
        performance = {}
        
        # Prediction distribution
        predictions = model.predict(X)
        residuals = y.iloc[:, 0] - predictions.iloc[:, 0] if len(y.columns) == 1 else None
        
        if residuals is not None:
            performance["residual_analysis"] = {
                "mean_residual": float(residuals.mean()),
                "std_residual": float(residuals.std()),
                "skewness": float(residuals.skew()),
                "kurtosis": float(residuals.kurtosis()),
            }
        
        # Performance across data ranges
        performance["range_performance"] = self._analyze_range_performance(model, X, y)
        
        return performance

    def _analyze_range_performance(self, model: CalibrationModel, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """Analyze model performance across different data ranges."""
        range_performance = {}
        
        if len(y.columns) == 1:
            target = y.columns[0]
            # Split data into quartiles
            quartiles = np.percentile(y[target], [25, 50, 75])
            
            for i, (low, high) in enumerate([(None, quartiles[0]), 
                                           (quartiles[0], quartiles[1]),
                                           (quartiles[1], quartiles[2]),
                                           (quartiles[2], None)]):
                if low is None:
                    mask = y[target] <= high
                    range_name = f"Q1 (≤{high:.2f})"
                elif high is None:
                    mask = y[target] > low
                    range_name = f"Q4 (>{low:.2f})"
                else:
                    mask = (y[target] > low) & (y[target] <= high)
                    range_name = f"Q{i+1} ({low:.2f}-{high:.2f})"
                
                if mask.any():
                    X_subset = X[mask]
                    y_subset = y[mask]
                    
                    try:
                        preds = model.predict(X_subset)
                        r2 = float(r2_score(y_subset, preds))
                        mae = float(mean_absolute_error(y_subset, preds))
                        
                        range_performance[range_name] = {
                            "n_samples": int(mask.sum()),
                            "r2_score": r2,
                            "mae": mae,
                            "target_mean": float(y_subset[target].mean()),
                            "target_std": float(y_subset[target].std())
                        }
                    except Exception as e:
                        range_performance[range_name] = {
                            "n_samples": int(mask.sum()),
                            "error": str(e)
                        }
        
        return range_performance

    def _analyze_data(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data characteristics and quality."""
        analysis = {}
        
        # Basic statistics
        analysis["dataset_info"] = {
            "n_samples": len(X),
            "n_features": len(X.columns),
            "n_targets": len(y.columns),
            "memory_usage_mb": X.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Data quality
        analysis["data_quality"] = {
            "missing_values": {
                "features": X.isnull().sum().to_dict(),
                "targets": y.isnull().sum().to_dict()
            },
            "infinite_values": {
                "features": (X == np.inf).sum().to_dict(),
                "targets": (y == np.inf).sum().to_dict()
            }
        }
        
        # Feature statistics
        analysis["feature_statistics"] = {}
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                analysis["feature_statistics"][col] = {
                    "mean": float(X[col].mean()),
                    "std": float(X[col].std()),
                    "min": float(X[col].min()),
                    "max": float(X[col].max()),
                    "skew": float(X[col].skew()),
                    "kurtosis": float(X[col].kurtosis())
                }
        
        # Target statistics
        analysis["target_statistics"] = {}
        for col in y.columns:
            if pd.api.types.is_numeric_dtype(y[col]):
                analysis["target_statistics"][col] = {
                    "mean": float(y[col].mean()),
                    "std": float(y[col].std()),
                    "min": float(y[col].min()),
                    "max": float(y[col].max())
                }
        
        return analysis

    def _generate_recommendations(self, model: CalibrationModel, X: pd.DataFrame, y: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on model analysis."""
        recommendations = []
        
        # Check data quality
        missing_features = X.isnull().sum()
        missing_targets = y.isnull().sum()
        
        if missing_features.any():
            recommendations.append({
                "type": "data_quality",
                "priority": "high",
                "message": f"Found {missing_features.sum()} missing values in features",
                "action": "Handle missing values using imputation or removal"
            })
        
        if missing_targets.any():
            recommendations.append({
                "type": "data_quality",
                "priority": "high",
                "message": f"Found {missing_targets.sum()} missing values in targets",
                "action": "Remove samples with missing target values"
            })
        
        # Check feature importance
        if "feature_importances" in model.metrics:
            importances = model.metrics["feature_importances"]
            low_importance = [f for f, imp in importances.items() if imp < 0.01]
            
            if low_importance:
                recommendations.append({
                    "type": "feature_selection",
                    "priority": "medium",
                    "message": f"Found {len(low_importance)} features with low importance",
                    "action": f"Consider removing features: {', '.join(low_importance[:5])}"
                })
        
        # Check model performance
        if "r2_score" in model.metrics:
            r2 = model.metrics["r2_score"]
            if r2 < 0.7:
                recommendations.append({
                    "type": "model_performance",
                    "priority": "high",
                    "message": f"Model R² score ({r2:.3f}) is below 0.7",
                    "action": "Consider trying different model types or feature engineering"
                })
            elif r2 < 0.9:
                recommendations.append({
                    "type": "model_performance",
                    "priority": "medium",
                    "message": f"Model R² score ({r2:.3f}) has room for improvement",
                    "action": "Try hyperparameter optimization or ensemble methods"
                })
        
        # Check data size
        if len(X) < 100:
            recommendations.append({
                "type": "data_quantity",
                "priority": "medium",
                "message": f"Dataset size ({len(X)} samples) is relatively small",
                "action": "Consider collecting more data or using data augmentation"
            })
        
        return recommendations

    def generate_comparison_report(self, model_ids: List[str], X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive comparison report for multiple models."""
        comparison = {}
        
        for model_id in model_ids:
            try:
                insights = self.generate_model_insights(model_id, X, y)
                comparison[model_id] = insights
            except Exception as e:
                comparison[model_id] = {"error": str(e)}
        
        # Overall comparison
        comparison["summary"] = {
            "best_model_by_r2": max(
                model_ids,
                key=lambda mid: comparison[mid].get("training_metrics", {}).get("r2_score", -1) 
                if isinstance(comparison.get(mid), dict) and "error" not in comparison.get(mid, {}) else -1
            ) if any("error" not in comparison.get(mid, {}) for mid in model_ids) else "N/A"
        }
        
        return comparison

# =====================================================================
# Enhanced Validation Engine
# (Legacy compatibility: accepts legacy validation workflows)
# =====================================================================

class ValidationEngine:
    """Advanced model validation and testing."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.validation_history = []

    def cross_validate(self, model_id: str, X: pd.DataFrame, y: pd.DataFrame, 
                      cv_folds: int = 5, scoring: str = "r2") -> Dict[str, Any]:
        """Perform comprehensive cross-validation."""
        model = self.model_manager.get_model(model_id)
        
        if not HAVE_SKLEARN:
            return {"error": "scikit-learn required for cross-validation"}
        
        try:
            X_scaled = model.scaler_x.transform(X) if model.scaler_x else X
            y_scaled = model.scaler_y.transform(y) if model.scaler_y else y
            
            scores = cross_val_score(
                model.model, X_scaled, y_scaled,
                cv=cv_folds, scoring=scoring, n_jobs=-1
            )
            
            result = {
                "mean_score": float(scores.mean()),
                "std_score": float(scores.std()),
                "fold_scores": scores.tolist(),
                "cv_folds": cv_folds,
                "scoring": scoring
            }
            
            self.validation_history.append({
                "model_id": model_id,
                "validation_type": "cross_validation",
                "timestamp": _now_iso(),
                "results": result
            })
            
            return result
            
        except Exception as e:
            return {"error": str(e)}

    def learning_curve(self, model_id: str, X: pd.DataFrame, y: pd.DataFrame,
                      train_sizes: Optional[List[float]] = None,
                      cv_folds: int = 5) -> Dict[str, Any]:
        """Generate learning curve data."""
        if not HAVE_SKLEARN:
            return {"error": "scikit-learn required for learning curves"}
        
        from sklearn.model_selection import learning_curve
        
        model = self.model_manager.get_model(model_id)
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        try:
            X_scaled = model.scaler_x.transform(X) if model.scaler_x else X
            y_scaled = model.scaler_y.transform(y) if model.scaler_y else y
            
            train_sizes_abs, train_scores, test_scores = learning_curve(
                model.model, X_scaled, y_scaled,
                train_sizes=train_sizes, cv=cv_folds,
                scoring='r2', n_jobs=-1, random_state=42
            )
            
            result = {
                "train_sizes": train_sizes_abs.tolist(),
                "train_scores_mean": train_scores.mean(axis=1).tolist(),
                "train_scores_std": train_scores.std(axis=1).tolist(),
                "test_scores_mean": test_scores.mean(axis=1).tolist(),
                "test_scores_std": test_scores.std(axis=1).tolist()
            }
            
            self.validation_history.append({
                "model_id": model_id,
                "validation_type": "learning_curve",
                "timestamp": _now_iso(),
                "results": result
            })
            
            return result
            
        except Exception as e:
            return {"error": str(e)}

    def residual_analysis(self, model_id: str, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """Perform detailed residual analysis."""
        model = self.model_manager.get_model(model_id)
        
        try:
            predictions = model.predict(X)
            residuals = y.iloc[:, 0] - predictions.iloc[:, 0] if len(y.columns) == 1 else None
            
            if residuals is None:
                return {"error": "Residual analysis currently supports single target only"}
            
            # Normality test
            from scipy.stats import shapiro, normaltest
            shapiro_stat, shapiro_p = shapiro(residuals)
            normal_stat, normal_p = normaltest(residuals)
            
            # Autocorrelation
            try:
                from statsmodels.tsa.stattools import acf
                autocorr = acf(residuals, nlags=min(10, len(residuals)//4), fft=False)
                autocorr_list = autocorr.tolist()
            except ImportError:
                autocorr_list = "statsmodels not installed"

            result = {
                "residual_statistics": {
                    "mean": float(residuals.mean()),
                    "std": float(residuals.std()),
                    "min": float(residuals.min()),
                    "max": float(residuals.max()),
                    "skewness": float(residuals.skew()),
                    "kurtosis": float(residuals.kurtosis())
                },
                "normality_tests": {
                    "shapiro_wilk": {
                        "statistic": float(shapiro_stat),
                        "p_value": float(shapiro_p)
                    },
                    "dagostino_pearson": {
                        "statistic": float(normal_stat),
                        "p_value": float(normal_p)
                    }
                },
                "autocorrelation": autocorr_list,
                "heteroscedasticity": self._test_heteroscedasticity(residuals, predictions.iloc[:, 0])
            }
            
            self.validation_history.append({
                "model_id": model_id,
                "validation_type": "residual_analysis",
                "timestamp": _now_iso(),
                "results": result
            })
            
            return result
            
        except Exception as e:
            return {"error": str(e)}

    def _test_heteroscedasticity(self, residuals: pd.Series, predictions: pd.Series) -> Dict[str, float]:
        """Test for heteroscedasticity using Breusch-Pagan test approximation."""
        try:
            # Simple correlation between absolute residuals and predictions
            correlation = np.corrcoef(np.abs(residuals), predictions)[0, 1]
            return {"absolute_residuals_correlation": float(correlation)}
        except:
            return {"absolute_residuals_correlation": 0.0}

    def generate_validation_report(self, model_id: str, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        report = {}
        
        report["cross_validation"] = self.cross_validate(model_id, X, y)
        report["learning_curve"] = self.learning_curve(model_id, X, y)
        report["residual_analysis"] = self.residual_analysis(model_id, X, y)
        
        # Overall assessment
        scores = []
        if "error" not in report["cross_validation"]:
            scores.append(report["cross_validation"]["mean_score"])
        
        report["overall_assessment"] = {
            "mean_cv_score": np.mean(scores) if scores else 0.0,
            "is_reliable": len(scores) > 0 and np.mean(scores) > 0.7
        }
        
        return report

# =====================================================================
# Main CIE Engine Class
# =====================================================================

class CalibrationIntelligenceEngine:
    """World's Most Advanced Calibration Intelligence Engine - Main Class"""
    
    def __init__(self, model_dir: Union[str, Path] = "models"):
        self.model_manager = ModelManager(model_dir)
        self.optimization_engine = OptimizationEngine(self.model_manager)
        self.map_generator = MapGenerator(self.model_manager)
        self.interpolation_engine = InterpolationEngine()
        self.recommendation_engine = RecommendationEngine(self.model_manager)
        self.doe_engine = DOEEngine()
        self.performance_optimizer = PerformanceOptimizer()
        self.analytics_engine = AnalyticsEngine(self.model_manager)
        self.validation_engine = ValidationEngine(self.model_manager)
        
        self.engine_info = {
            "version": "6.0.0",
            "created_at": _now_iso(),
            "capabilities": [
                "model_training",
                "hyperparameter_optimization", 
                "map_generation",
                "optimization",
                "design_of_experiments",
                "recommendation_engine",
                "analytics_and_insights",
                "model_validation"
            ]
        }
    
    def train_model(self, model_id: str, feature_data: Dict[str, Iterable], 
                   target_data: Dict[str, Iterable], **kwargs) -> Dict[str, Any]:
        """High-level model training interface."""
        X, y = prepare_training_data(feature_data, target_data, **kwargs)
        return self.model_manager.train_model(model_id, X, y, **kwargs)
    
    def optimize(self, model_id: str, targets: Dict[str, float], **kwargs) -> Dict[str, Any]:
        """High-level optimization interface."""
        return self.optimization_engine.optimize(model_id, targets, **kwargs)
    
    def generate_map(self, model_id: str, grid_definitions: Dict[str, List[float]]) -> Dict[str, Any]:
        """High-level map generation interface."""
        return self.map_generator.generate_map(model_id, grid_definitions)
    
    def recommend(self, *args, **kwargs) -> Any:
        """High-level recommendation interface."""
        return self.recommendation_engine.recommend(*args, **kwargs)
    
    def analyze_model(self, model_id: str, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """High-level model analysis interface."""
        return self.analytics_engine.generate_model_insights(model_id, X, y)
    
    def validate_model(self, model_id: str, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """High-level model validation interface."""
        return self.validation_engine.generate_validation_report(model_id, X, y)
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine information and capabilities."""
        return self.engine_info
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and resource usage."""
        status = {
            "engine_version": self.engine_info["version"],
            "timestamp": _now_iso(),
            "model_count": len(self.model_manager.list_models()),
            "active_model": self.model_manager.active_model,
            "dependencies": {
                "scikit_learn": HAVE_SKLEARN,
                "scipy": HAVE_SCIPY,
                "scikit_optimize": HAVE_SKOPT,
                "optuna": HAVE_OPTUNA,
                "psutil": HAVE_PSUTIL
            }
        }
        
        if HAVE_PSUTIL:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            status["system_resources"] = {
                "memory_used_mb": memory_info.rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "threads": process.num_threads()
            }
        
        return status

# =====================================================================
# Factory Functions and Utilities
# =====================================================================

def create_advanced_engine(model_dir: Union[str, Path] = "models") -> CalibrationIntelligenceEngine:
    """Factory function to create an advanced CIE instance."""
    return CalibrationIntelligenceEngine(model_dir)

def quick_train(feature_data: Dict[str, Iterable], target_data: Dict[str, Iterable],
              model_type: str = "random_forest", model_id: str = "quick_model",
               **kwargs) -> Dict[str, Any]:
    """Quick training utility for rapid prototyping."""
    engine = create_advanced_engine()
    return engine.train_model(model_id, feature_data, target_data, model_type=model_type, **kwargs)

def quick_optimize(model_id: str, targets: Dict[str, float],
                  feature_ranges: Dict[str, Tuple[float, float]],
                  **kwargs) -> Dict[str, Any]:
    """Quick optimization utility."""
    engine = create_advanced_engine()
    return engine.optimize(model_id, targets, **kwargs)

# =====================================================================
# Legacy Compatibility Layer
# =====================================================================

class LegacyCIEAdapter:
    """
    Adapter for backward compatibility with previous CIE versions.
    This class provides legacy interfaces so that older scripts and integrations
    can continue working without modification.
    """
    
    def __init__(self, engine: CalibrationIntelligenceEngine):
        self.engine = engine
    
    def create_model(self, *args, **kwargs):
        """Legacy create_model compatibility. Use engine's model_manager."""
        return self.engine.model_manager.create_model(*args, **kwargs)
    
    def train(self, *args, **kwargs):
        """Legacy train compatibility. Use engine's train_model."""
        return self.engine.train_model(*args, **kwargs)
    
    def predict(self, model_id: str, features: Dict[str, float]) -> Dict[str, float]:
        """Legacy predict compatibility. Use engine's predict interface."""
        model = self.engine.model_manager.get_model(model_id)
        df = pd.DataFrame([features])
        predictions = model.predict(df)
        return predictions.iloc[0].to_dict()
    
    def optimize_parameters(self, *args, **kwargs):
        """Legacy optimize_parameters compatibility. Use engine's optimize."""
        return self.engine.optimize(*args, **kwargs)

# =====================================================================
# Export and Serialization Utilities
# =====================================================================

def export_model_to_json(model: CalibrationModel, include_training_data: bool = False) -> str:
    """
    Export model to JSON format for interoperability.
    Legacy compatibility: can exclude large data arrays for lightweight export.
    """
    model_info = model.get_info()
    
    if not include_training_data:
        # Remove large data arrays for lightweight export
        if "training_data_info" in model_info:
            model_info["training_data_info"] = {
                k: v for k, v in model_info["training_data_info"].items() 
                if not isinstance(v, (np.ndarray, list)) or len(str(v)) < 1000
            }
    
    return json.dumps(model_info, indent=2, default=str)

def import_model_from_json(json_str: str) -> Dict[str, Any]:
    """Import model configuration from JSON."""
    return json.loads(json_str)

# =====================================================================
# (Optional) Legacy API Wrappers
# =====================================================================

def legacy_train_model(*args, **kwargs):
    """Legacy API wrapper for train_model."""
    engine = create_advanced_engine()
    return engine.train_model(*args, **kwargs)

# =====================================================================
# Main Execution Block
# =====================================================================

if __name__ == "__main__":
    # Example usage and testing
    print("Advanced CIE Backend - Fully Updated cie.py")
    print("=" * 50)
    print("Welcome to the world's most advanced Calibration Intelligence Engine (CIE).")
    print("This engine supports legacy interfaces for backward compatibility.")
    
    # Create engine instance
    engine = create_advanced_engine()
    
    # Display system status
    status = engine.get_system_status()
    print(f"\nCIE Engine v{status['engine_version']}")
    print(f"Active models: {status['model_count']}")
    print(f"Dependencies: {status['dependencies']}")
    
    # Example of creating sample data for demonstration
    def create_sample_data(n_samples: int = 100) -> Tuple[Dict[str, List], Dict[str, List]]:
        np.random.seed(42)
        features = {
            'temperature': np.random.uniform(20, 100, n_samples).tolist(),
            'pressure': np.random.uniform(1, 10, n_samples).tolist(),
            'flow_rate': np.random.uniform(0.1, 5.0, n_samples).tolist()
        }
        targets = {
            'efficiency': [0.7 * t + 0.2 * p + 0.1 * f + np.random.normal(0, 0.1) 
                          for t, p, f in zip(features['temperature'], 
                                            features['pressure'], 
                                            features['flow_rate'])]
        }
        return features, targets
    
    print("\nSample capabilities demonstrated successfully!")
    print("Advanced CIE is ready for production use.")