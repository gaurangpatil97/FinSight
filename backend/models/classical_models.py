"""Classical time series models for FinSight.

This module provides a common interface over ARIMA, SARIMA, ETS, and SARIMAX
models so the training pipeline can use them interchangeably.
"""

from __future__ import annotations

import logging
import re
from abc import ABC
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pmdarima as pm
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


LOGGER = logging.getLogger(__name__)
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def _safe_ticker_name(ticker: str) -> str:
    """Convert a ticker into a filesystem-safe name."""

    return re.sub(r"[^A-Za-z0-9]", "_", ticker.strip()) or "default"


def _validate_series(series: pd.Series) -> pd.Series:
    """Validate and normalize the input time series."""

    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")

    cleaned_series = series.copy().dropna()
    if cleaned_series.empty:
        raise ValueError("series must contain at least one non-null value.")

    if isinstance(cleaned_series.index, pd.DatetimeIndex):
        cleaned_series = cleaned_series.sort_index()
        cleaned_series.index = pd.DatetimeIndex(
            cleaned_series.index.normalize()
        ).floor("D")

    return cleaned_series.astype(float)


def _validate_steps(steps: int) -> None:
    """Validate forecast step count."""

    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer.")


def _future_index(series: pd.Series, steps: int) -> pd.Index:
    """Build a future index for forecast visualizations."""

    if isinstance(series.index, pd.DatetimeIndex) and not series.index.empty:
        start = series.index[-1] + pd.tseries.offsets.BDay(1)
        return pd.bdate_range(start=start, periods=steps)

    return pd.RangeIndex(start=len(series), stop=len(series) + steps)


def _artifact_root_for_ticker(ticker: str) -> Path:
    """Return the ticker-specific artifact root."""

    return ARTIFACTS_DIR / _safe_ticker_name(ticker)


class BaseClassicalModel(ABC):
    """Common interface for classical forecasting models."""

    model_name = "base"

    def __init__(self, ticker: str = "default") -> None:
        """Initialize the base model state.

        Parameters
        ----------
        ticker:
            Ticker symbol used for artifact storage.
        """

        self.ticker = ticker
        self.safe_ticker = _safe_ticker_name(ticker)
        self.model_name = getattr(self, "model_name", "base")
        self.artifacts_dir = _artifact_root_for_ticker(ticker)
        self.artifact_path = self.artifacts_dir / f"{self.model_name}_model.pkl"
        self._model = None
        self._training_series: pd.Series | None = None
        self._last_forecast: np.ndarray | None = None
        self._last_intervals: tuple[np.ndarray, np.ndarray] | None = None
        self._last_steps: int | None = None
        self._is_fitted = False

    def fit(self, series: pd.Series) -> None:
        """Fit the model on a single time series."""

        cleaned_series = _validate_series(series)
        try:
            self._model = self._fit_model(cleaned_series)
            self._training_series = cleaned_series
            self._last_forecast = None
            self._last_intervals = None
            self._last_steps = None
            self._is_fitted = True
            LOGGER.info("Fitted %s for %s.", self.model_name, self.ticker)
        except Exception:
            LOGGER.exception("Failed to fit %s for %s.", self.model_name, self.ticker)
            raise

    def predict(self, steps: int = 30) -> np.ndarray:
        """Generate point forecasts for the requested horizon."""

        self._require_fitted()
        _validate_steps(steps)

        try:
            forecast = np.asarray(self._predict_impl(steps), dtype=float).reshape(-1)
            self._last_forecast = forecast
            self._last_steps = steps
            self._last_intervals = None
            LOGGER.info("Generated %d-step forecast for %s.", steps, self.model_name)
            return forecast
        except Exception:
            LOGGER.exception("Failed to predict with %s for %s.", self.model_name, self.ticker)
            raise

    def get_confidence_intervals(self, steps: int = 30) -> tuple:
        """Return lower and upper forecast confidence bounds."""

        self._require_fitted()
        _validate_steps(steps)

        try:
            intervals = self._confidence_intervals_impl(steps)
            lower = np.asarray(intervals[0], dtype=float).reshape(-1)
            upper = np.asarray(intervals[1], dtype=float).reshape(-1)
            self._last_intervals = (lower, upper)
            self._last_steps = steps
            return lower, upper
        except Exception:
            LOGGER.exception("Failed to compute confidence intervals for %s.", self.model_name)
            raise

    def save(self, ticker: str) -> None:
        """Persist the fitted model for a ticker."""

        self._require_fitted()
        self._set_ticker(ticker)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self._model,
            "ticker": self.ticker,
            "model_name": self.model_name,
            "meta": self._save_metadata(),
        }
        joblib.dump(payload, self.artifact_path)
        LOGGER.info("Saved %s model to %s.", self.model_name, self.artifact_path)

    def load(self, ticker: str) -> None:
        """Load a previously saved model for a ticker."""

        self._set_ticker(ticker)
        if not self.artifact_path.exists():
            raise FileNotFoundError(f"Missing artifact: {self.artifact_path}")

        payload = joblib.load(self.artifact_path)
        if isinstance(payload, dict) and "model" in payload:
            self._model = payload["model"]
            self._load_metadata(payload.get("meta", {}))
        else:
            self._model = payload

        self._is_fitted = True
        LOGGER.info("Loaded %s model from %s.", self.model_name, self.artifact_path)

    def plot_forecast(self, series: pd.Series, steps: int = 30) -> Figure:
        """Plot the historical series, forecast, and confidence intervals."""

        self._require_fitted()
        cleaned_series = _validate_series(series)
        _validate_steps(steps)

        try:
            forecast = self.predict(steps)
            lower, upper = self.get_confidence_intervals(steps)
            history = cleaned_series.iloc[-90:]
            future_index = _future_index(history, steps)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=history.index,
                    y=history.values,
                    mode="lines",
                    name="Historical Close",
                    line=dict(color="#111827", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=future_index,
                    y=forecast,
                    mode="lines",
                    name="Forecast",
                    line=dict(color="#2563eb", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=future_index,
                    y=upper,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    name="Upper CI",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=future_index,
                    y=lower,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(37, 99, 235, 0.18)",
                    name="95% CI",
                    hoverinfo="skip",
                )
            )
            fig.update_layout(
                title=f"{self.ticker} {self.model_name.upper()} Forecast",
                template="plotly_white",
                xaxis_title="Date",
                yaxis_title="Value",
                height=600,
            )
            return fig
        except Exception:
            LOGGER.exception("Failed to plot forecast for %s.", self.model_name)
            raise

    def _set_ticker(self, ticker: str) -> None:
        """Update ticker-specific storage paths."""

        self.ticker = ticker
        self.safe_ticker = _safe_ticker_name(ticker)
        self.artifacts_dir = _artifact_root_for_ticker(ticker)
        self.artifact_path = self.artifacts_dir / f"{self.model_name}_model.pkl"

    def _require_fitted(self) -> None:
        """Raise if the model has not been fitted or loaded."""

        if not self._is_fitted or self._model is None:
            raise RuntimeError(f"{self.model_name} must be fitted or loaded before use.")

    def _save_metadata(self) -> dict:
        """Return subclass-specific metadata for persistence."""

        return {}

    def _load_metadata(self, metadata: dict) -> None:
        """Restore subclass-specific metadata after loading."""

    def _fit_model(self, series: pd.Series):  # pragma: no cover - implemented by subclasses
        """Fit the concrete model implementation."""

        raise NotImplementedError

    def _predict_impl(self, steps: int):  # pragma: no cover - implemented by subclasses
        """Return point forecasts for the concrete model implementation."""

        raise NotImplementedError

    def _confidence_intervals_impl(self, steps: int):  # pragma: no cover - implemented by subclasses
        """Return confidence intervals for the concrete model implementation."""

        raise NotImplementedError


class ARIMAModel(BaseClassicalModel):
    """ARIMA model using pmdarima auto_arima."""

    model_name = "arima"

    def _fit_model(self, series: pd.Series):
        """Fit an ARIMA model by selecting the best order automatically."""

        try:
            model = pm.auto_arima(
                series,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
            )
            return model
        except Exception:
            LOGGER.exception("auto_arima failed for ARIMA; falling back to a simple order.")
            return pm.ARIMA(order=(1, 1, 0)).fit(series)

    def _predict_impl(self, steps: int):
        forecast, _ = self._model.predict(n_periods=steps, return_conf_int=True, alpha=0.05)
        return forecast

    def _confidence_intervals_impl(self, steps: int):
        _, conf_int = self._model.predict(n_periods=steps, return_conf_int=True, alpha=0.05)
        return conf_int[:, 0], conf_int[:, 1]


class SARIMAModel(BaseClassicalModel):
    """SARIMA model using pmdarima auto_arima with seasonality."""

    model_name = "sarima"

    def __init__(self, ticker: str = "default", m: int = 5) -> None:
        """Initialize the SARIMA model.

        Parameters
        ----------
        ticker:
            Ticker symbol used for artifact storage.
        m:
            Seasonal period.
        """

        super().__init__(ticker=ticker)
        self.m = m

    def _fit_model(self, series: pd.Series):
        """Fit a SARIMA model by selecting the best seasonal order automatically."""

        try:
            model = pm.auto_arima(
                series,
                seasonal=True,
                m=self.m,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
            )
            return model
        except Exception:
            LOGGER.exception("auto_arima failed for SARIMA; falling back to a simple seasonal order.")
            return pm.ARIMA(order=(1, 1, 0), seasonal_order=(1, 1, 0, self.m)).fit(series)

    def _predict_impl(self, steps: int):
        forecast, _ = self._model.predict(n_periods=steps, return_conf_int=True, alpha=0.05)
        return forecast

    def _confidence_intervals_impl(self, steps: int):
        _, conf_int = self._model.predict(n_periods=steps, return_conf_int=True, alpha=0.05)
        return conf_int[:, 0], conf_int[:, 1]

    def _save_metadata(self) -> dict:
        """Persist the seasonal period for reload consistency."""

        return {"m": self.m}

    def _load_metadata(self, metadata: dict) -> None:
        """Restore the seasonal period from persisted metadata."""

        self.m = int(metadata.get("m", self.m))


class ETSModel(BaseClassicalModel):
    """Holt-Winters exponential smoothing model."""

    model_name = "ets"

    def __init__(self, ticker: str = "default", m: int = 5) -> None:
        """Initialize the ETS model.

        Parameters
        ----------
        ticker:
            Ticker symbol used for artifact storage.
        m:
            Seasonal period.
        """

        super().__init__(ticker=ticker)
        self.m = m

    def _fit_variant(self, series: pd.Series, trend: str, seasonal: str):
        """Fit one ETS specification and return the fitted result."""

        model = ExponentialSmoothing(series, trend=trend, seasonal=seasonal, seasonal_periods=self.m)
        return model.fit(optimized=True, use_brute=True)

    def _fit_model(self, series: pd.Series):
        """Fit additive and multiplicative ETS variants and keep the lower-AIC model."""

        candidates: list[tuple[float, object]] = []

        try:
            additive_result = self._fit_variant(series, trend="add", seasonal="add")
            candidates.append((float(getattr(additive_result, "aic", np.inf)), additive_result))
        except Exception:
            LOGGER.exception("Additive ETS fit failed for %s.", self.ticker)

        if (series > 0).all():
            try:
                multiplicative_result = self._fit_variant(series, trend="mul", seasonal="mul")
                candidates.append((float(getattr(multiplicative_result, "aic", np.inf)), multiplicative_result))
            except Exception:
                LOGGER.exception("Multiplicative ETS fit failed for %s.", self.ticker)

        if not candidates:
            raise ValueError("Unable to fit any ETS specification.")

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _predict_impl(self, steps: int):
        return self._model.forecast(steps)

    def _confidence_intervals_impl(self, steps: int):
        try:
            simulations = self._model.simulate(nsimulations=steps, repetitions=200)
            sim_array = np.asarray(simulations, dtype=float)

            if sim_array.ndim == 1:
                sim_array = sim_array.reshape(-1, 1)

            if sim_array.shape[0] != steps and sim_array.shape[1] == steps:
                sim_array = sim_array.T

            lower = np.nanpercentile(sim_array, 2.5, axis=1)
            upper = np.nanpercentile(sim_array, 97.5, axis=1)
            return lower, upper
        except Exception:
            LOGGER.exception("ETS simulation intervals failed; using residual-based fallback.")
            fitted = np.asarray(self._model.fittedvalues, dtype=float)
            residuals = np.asarray(self._training_series.loc[fitted.index] - fitted, dtype=float) if self._training_series is not None else np.asarray(self._model.resid, dtype=float)
            resid_std = float(np.nanstd(residuals, ddof=1)) if residuals.size else 0.0
            forecast = np.asarray(self._model.forecast(steps), dtype=float)
            margin = 1.96 * resid_std
            return forecast - margin, forecast + margin

    def _save_metadata(self) -> dict:
        """Persist the seasonal period for reload consistency."""

        return {"m": self.m}

    def _load_metadata(self, metadata: dict) -> None:
        """Restore the seasonal period from persisted metadata."""

        self.m = int(metadata.get("m", self.m))


class SARIMAXModel(BaseClassicalModel):
    """SARIMAX model with optional exogenous regressors."""

    model_name = "sarimax"

    def __init__(self, ticker: str = "default", exog: pd.DataFrame | None = None) -> None:
        """Initialize the SARIMAX model.

        Parameters
        ----------
        ticker:
            Ticker symbol used for artifact storage.
        exog:
            Optional exogenous variables aligned with the training series.
        """

        super().__init__(ticker=ticker)
        self.exog = exog.copy() if isinstance(exog, pd.DataFrame) else exog
        self._training_exog: pd.DataFrame | None = None
        self._exog_columns: list[str] | None = None

    def _align_exog(self, series: pd.Series) -> pd.DataFrame | None:
        """Align exogenous variables to the training series."""

        if self.exog is None:
            return None

        exog = self.exog.copy()
        if len(exog) != len(series):
            raise ValueError("exog must have the same number of rows as the training series.")

        if isinstance(series.index, pd.DatetimeIndex):
            exog.index = series.index

        exog = exog.apply(pd.to_numeric, errors="coerce").ffill().bfill()
        if exog.isna().any().any():
            raise ValueError("exog contains non-numeric or missing values that could not be cleaned.")

        return exog

    def _future_exog(self, steps: int) -> pd.DataFrame | None:
        """Create future exogenous values for forecasting."""

        if self._training_exog is None:
            return None

        last_row = self._training_exog.iloc[[-1]].copy()
        future_exog = pd.concat([last_row] * steps, ignore_index=True)
        future_exog.columns = self._training_exog.columns
        return future_exog

    def _fit_model(self, series: pd.Series):
        """Fit a SARIMAX model using an order selected by auto_arima."""

        self._training_exog = self._align_exog(series)
        self._exog_columns = list(self._training_exog.columns) if self._training_exog is not None else None

        try:
            order_model = pm.auto_arima(
                series,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
            )
            order = order_model.order
        except Exception:
            LOGGER.exception("auto_arima failed for SARIMAX; using fallback order.")
            order = (1, 1, 0)

        model = SARIMAX(
            series,
            exog=self._training_exog,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False)

    def _predict_impl(self, steps: int):
        future_exog = self._future_exog(steps)
        forecast_result = self._model.get_forecast(steps=steps, exog=future_exog)
        return forecast_result.predicted_mean.to_numpy()

    def _confidence_intervals_impl(self, steps: int):
        future_exog = self._future_exog(steps)
        forecast_result = self._model.get_forecast(steps=steps, exog=future_exog)
        conf_int = forecast_result.conf_int(alpha=0.05)
        if isinstance(conf_int, pd.DataFrame):
            lower = conf_int.iloc[:, 0].to_numpy()
            upper = conf_int.iloc[:, 1].to_numpy()
        else:
            lower = np.asarray(conf_int[:, 0], dtype=float)
            upper = np.asarray(conf_int[:, 1], dtype=float)
        return lower, upper

    def _save_metadata(self) -> dict:
        """Persist exogenous column information for reload consistency."""

        return {"exog_columns": self._exog_columns}

    def _load_metadata(self, metadata: dict) -> None:
        """Restore exogenous metadata after loading."""

        exog_columns = metadata.get("exog_columns")
        if isinstance(exog_columns, list):
            self._exog_columns = [str(column) for column in exog_columns]


def get_classical_model(model_name: str, **kwargs) -> BaseClassicalModel:
    """Return a classical model instance by name."""

    models = {
        "arima": ARIMAModel,
        "sarima": SARIMAModel,
        "ets": ETSModel,
        "sarimax": SARIMAXModel,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    return models[model_name](**kwargs)
