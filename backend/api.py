"""FastAPI service for FinSight.

Provides endpoints for health, tickers, EDA, model metrics, forecasts and
model comparisons. This module re-uses the FinSight backend utilities for
data loading, EDA generation and preprocessing.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.data.data_loader import fetch_stock_data
from backend.data.preprocessor import FinSightPreprocessor
from backend.eda.eda_pipeline import FinSightEDA

LOGGER = logging.getLogger(__name__)

APP = FastAPI(title="FinSight API")

# Development-friendly CORS
APP.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = Path(BASE_DIR) / "artifacts"
EDA_OUTPUTS_DIR = Path(BASE_DIR) / "eda" / "outputs"
NOTEBOOKS_DIR = Path(os.path.dirname(BASE_DIR)) / "notebooks" / "models"
DEFAULT_TICKERS = ["RELIANCE.NS", "HDFCBANK.NS", "SBIN.NS"]


def _safe_ticker(ticker: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", ticker.strip()) or "default"


def _model_artifact_dir(ticker: str) -> Path:
    return ARTIFACTS_DIR / _safe_ticker(ticker)


def _find_model_file(artifacts_dir: Path, model_name: str) -> Path | None:
    safe = artifacts_dir.name
    candidates = [
        artifacts_dir / f"{safe}_{model_name}_model.pkl",
        artifacts_dir / f"{model_name}_model.pkl",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


@APP.get("/health")
def health():
    return {"status": "ok"}


@APP.get("/tickers")
def tickers():
    return DEFAULT_TICKERS


@APP.get("/eda/{ticker}")
def eda(ticker: str):
    """Return paths to saved EDA HTML files, generating them if missing."""
    try:
        safe = _safe_ticker(ticker)
        eda_dir = EDA_OUTPUTS_DIR / safe
        expected_files = {
            "trend_analysis": eda_dir / "trend_analysis.html",
            "volatility_analysis": eda_dir / "volatility_analysis.html",
            "rolling_statistics": eda_dir / "rolling_statistics.html",
            "seasonality_analysis": eda_dir / "seasonality_analysis.html",
            "distribution_analysis": eda_dir / "distribution_analysis.html",
            "correlation_analysis": eda_dir / "correlation_analysis.html",
        }

        missing = [name for name, path in expected_files.items() if not path.exists()]
        if missing:
            # Generate EDA using FinSightEDA
            LOGGER.info("EDA missing for %s: %s. Generating now.", ticker, missing)
            start = "2020-01-01"
            end = date.today().strftime("%Y-%m-%d")
            df = fetch_stock_data(ticker, start, end)
            all_stocks = None
            # try to include default tickers for correlation
            try:
                all_stocks = fetch_stock_data(ticker, start, end)
            except Exception:
                all_stocks = None

            eda_runner = FinSightEDA()
            eda_runner.run_all(df, ticker, all_stocks_dict=None)

        return {k: str(v) for k, v in expected_files.items()}
    except Exception as exc:
        LOGGER.exception("EDA generation failed for %s", ticker)
        raise HTTPException(status_code=500, detail=str(exc))


def _load_preprocessor(ticker: str) -> FinSightPreprocessor | None:
    prep = FinSightPreprocessor(ticker=ticker)
    try:
        prep.load_artifacts()
        return prep
    except Exception:
        LOGGER.warning("Preprocessor artifacts missing for %s", ticker)
        return None


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any():
        return {"mape": None, "rmse": None, "mae": None}

    y_t = y_true[mask]
    y_p = y_pred[mask]
    # Avoid division by zero in MAPE
    denom = np.where(y_t == 0, np.nan, y_t)
    with np.errstate(invalid="ignore"):
        mape = float(np.nanmean(np.abs((y_t - y_p) / denom))) * 100.0
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    mae = float(np.mean(np.abs(y_t - y_p)))
    return {
        "mape": float(np.round(mape, 4)) if not np.isnan(mape) else None,
        "rmse": float(np.round(rmse, 4)),
        "mae": float(np.round(mae, 4))
    }


@APP.get("/models/{ticker}")
def models_metrics(ticker: str):
    if ticker not in DEFAULT_TICKERS:
        raise HTTPException(status_code=404, detail="Ticker not supported")

    artifacts_dir = _model_artifact_dir(ticker)
    model_names = ["xgboost", "lightgbm", "random_forest"]
    results: Dict[str, Dict] = {}

    # Fetch a year of data to ensure feature engineering works and keep last 30 days for evaluation
    end = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    try:
        df = fetch_stock_data(ticker, start, end)
    except Exception as exc:
        LOGGER.exception("Failed to fetch data for metrics: %s", ticker)
        raise HTTPException(status_code=500, detail=str(exc))

    preprocessor = _load_preprocessor(ticker)

    for name in model_names:
        try:
            model_path = _find_model_file(artifacts_dir, name)
            if model_path is None:
                LOGGER.warning("Model %s missing for %s", name, ticker)
                results[name] = {"mape": None, "rmse": None, "mae": None}
                continue

            model = joblib.load(model_path)

            # Prepare recent evaluation dataset
            if preprocessor is None:
                LOGGER.warning("No preprocessor for %s; skipping metrics for %s", ticker, name)
                results[name] = {"mape": None, "rmse": None, "mae": None}
                continue

            engineered = preprocessor._engineer_features(df)
            features = engineered[preprocessor.feature_columns_]
            # scaled features
            X_scaled = pd.DataFrame(preprocessor.feature_scaler.transform(features), index=features.index, columns=preprocessor.feature_columns_)

            if X_scaled.shape[0] < 5:
                results[name] = {"mape": None, "rmse": None, "mae": None}
                continue

            X_recent = X_scaled.iloc[-30:]
            y_true_raw = df["Close"].reindex(X_recent.index).astype(float).values
            y_true = y_true_raw

            # Some model objects expect 2D numpy arrays
            try:
                y_pred_raw = model.predict(X_recent.values)
            except Exception:
                y_pred_raw = model.predict(X_recent)

            # In notebooks we trained on scaled targets; attempt inverse scaling
            try:
                y_pred = preprocessor.inverse_transform(np.asarray(y_pred_raw).ravel())
            except Exception:
                y_pred = np.asarray(y_pred_raw).ravel()

            metrics = _compute_metrics(y_true, y_pred)
            results[name] = metrics
        except Exception:
            LOGGER.exception("Failed to evaluate model %s for %s", name, ticker)
            results[name] = {"mape": None, "rmse": None, "mae": None}

    # pick best by rmse
    best = None
    best_rmse = float("inf")
    for k, v in results.items():
        rmse = v.get("rmse")
        if isinstance(rmse, (int, float)) and rmse is not None and rmse < best_rmse:
            best_rmse = rmse
            best = k

    # Calculate additional metrics for dashboard
    current_price = None
    daily_return = None
    volatility = None
    trend = "Sideways"

    if not df.empty:
        current_price = float(df["Close"].iloc[-1])
        if "daily_return" in df.columns:
            daily_return = float(df["daily_return"].iloc[-1] * 100.0)
            # Annualized volatility of last 30 trading days
            recent_returns = df["daily_return"].iloc[-30:]
            if len(recent_returns) > 1:
                volatility = float(recent_returns.std() * np.sqrt(252) * 100.0)
            
        # Trend logic: using 20-day and 50-day moving averages
        if len(df) >= 50:
            ma20 = df["Close"].rolling(20).mean().iloc[-1]
            ma50 = df["Close"].rolling(50).mean().iloc[-1]
            diff = (ma20 - ma50) / ma50 * 100.0
            if diff > 1.0:
                trend = "Bullish"
            elif diff < -1.0:
                trend = "Bearish"
            else:
                trend = "Sideways"

    return {
        **results,
        "best_model": best,
        "current_price": round(current_price, 2) if current_price is not None else None,
        "daily_return": round(daily_return, 2) if daily_return is not None else None,
        "volatility": round(volatility, 2) if volatility is not None else None,
        "trend": trend
    }


@APP.get("/forecast/{ticker}/{model_name}")
def forecast_html(ticker: str, model_name: str):
    if ticker not in DEFAULT_TICKERS:
        raise HTTPException(status_code=404, detail="Ticker not supported")

    safe = _safe_ticker(ticker)
    artifacts_dir = _model_artifact_dir(ticker)

    # Map model names to their forecast HTML filename patterns
    forecast_patterns = {
        "xgboost": f"{safe}_xgboost_forecast.html",
        "lightgbm": f"{safe}_lgbm_forecast.html",
        "random_forest": f"{safe}_rf_forecast.html",
    }

    if model_name not in forecast_patterns:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

    filename = forecast_patterns[model_name]

    # Check artifacts folder first
    artifact_path = artifacts_dir / filename
    if artifact_path.exists():
        return {"forecast_html": str(artifact_path)}

    # Fallback: check notebooks folder
    model_folder_map = {
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "random_forest": "random_forest",
    }
    notebook_path = NOTEBOOKS_DIR / model_folder_map[model_name] / filename
    if notebook_path.exists():
        return {"forecast_html": str(notebook_path)}

    return {"forecast_html": None}


@APP.get("/comparison/{ticker}")
def comparison(ticker: str):
    if ticker not in DEFAULT_TICKERS:
        raise HTTPException(status_code=404, detail="Ticker not supported")

    metrics = models_metrics(ticker)
    # models_metrics already includes best_model
    return metrics


@APP.get("/debug/{ticker}")
def debug(ticker: str):
    artifacts_dir = _model_artifact_dir(ticker)
    files = list(artifacts_dir.glob("*")) if artifacts_dir.exists() else []
    return {
        "artifacts_dir": str(artifacts_dir),
        "exists": artifacts_dir.exists(),
        "files": [f.name for f in files]
    }
