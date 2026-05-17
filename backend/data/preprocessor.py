"""Preprocessing utilities for FinSight model training and inference.

The preprocessor prepares cleaned OHLCV data for supervised forecasting by
adding lag and rolling features, scaling numerical columns, and persisting the
feature order and fitted scalers for later reuse.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger(__name__)
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def _validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that the input is a pandas DataFrame and return a copy.

    Parameters
    ----------
    df:
        Input DataFrame to validate.

    Returns
    -------
    pd.DataFrame
        A copy of the validated input.

    Raises
    ------
    TypeError
        If ``df`` is not a pandas DataFrame.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    return df.copy()


class FinSightPreprocessor:
    """Prepare OHLCV time series data for FinSight forecasting models."""

    def __init__(self, ticker: str = "default", target_column: str = "Close") -> None:
        """Initialize a new preprocessor instance.

        Parameters
        ----------
        ticker:
            Symbol used to scope saved artifacts.
        target_column:
            Name of the price column used for inverse scaling.
        """

        self.ticker = ticker
        self.target_column = target_column
        safe_ticker_name = re.sub(r"[^A-Za-z0-9]+", "_", ticker)
        self.artifacts_dir = ARTIFACTS_DIR / safe_ticker_name
        self.feature_columns_path = self.artifacts_dir / "feature_columns.json"
        self.scaler_path = self.artifacts_dir / "scaler.pkl"
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_columns_: list[str] = []
        self.fitted_ = False

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag and rolling features from the input data.

        Parameters
        ----------
        df:
            Clean OHLCV DataFrame from the loader.

        Returns
        -------
        pd.DataFrame
            Feature-engineered DataFrame with invalid rows removed.

        Raises
        ------
        ValueError
            If the target column is missing.
        """

        if self.target_column not in df.columns:
            raise ValueError(f"Missing required column: {self.target_column}")

        prepared_df = df.copy()
        prepared_df = prepared_df.dropna(subset=[self.target_column])

        prepared_df["Close_lag_1"] = prepared_df[self.target_column].shift(1)
        prepared_df["Close_lag_5"] = prepared_df[self.target_column].shift(5)
        prepared_df["Close_lag_10"] = prepared_df[self.target_column].shift(10)
        prepared_df["rolling_mean_20"] = prepared_df[self.target_column].rolling(window=20).mean()
        prepared_df["rolling_std_20"] = prepared_df[self.target_column].rolling(window=20).std()

        prepared_df = prepared_df.dropna()
        if prepared_df.empty:
            raise ValueError("Not enough data after feature engineering.")

        return prepared_df

    def fit(self, df: pd.DataFrame) -> FinSightPreprocessor:
        """Learn scaling parameters from training data.

        Parameters
        ----------
        df:
            Training DataFrame.

        Returns
        -------
        FinSightPreprocessor
            The fitted preprocessor instance.

        Raises
        ------
        Exception
            If feature engineering or scaling fails.
        """

        try:
            validated_df = _validate_dataframe(df)
            engineered_df = self._engineer_features(validated_df)

            numeric_frame = engineered_df.select_dtypes(include=[np.number])
            if numeric_frame.empty:
                raise ValueError("No numeric columns available for scaling.")

            if self.target_column not in numeric_frame.columns:
                raise ValueError(f"Target column '{self.target_column}' must be numeric.")

            self.feature_columns_ = list(numeric_frame.columns)
            self.feature_scaler.fit(numeric_frame[self.feature_columns_])
            self.target_scaler.fit(numeric_frame[[self.target_column]])
            self.fitted_ = True
            self.save_artifacts()
            LOGGER.info("Fitted FinSightPreprocessor with %d feature columns.", len(self.feature_columns_))
            return self
        except Exception:
            LOGGER.exception("Failed to fit FinSightPreprocessor.")
            raise

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned scaling parameters without refitting.

        Parameters
        ----------
        df:
            DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Scaled feature DataFrame.

        Raises
        ------
        RuntimeError
            If the preprocessor has not been fitted.
        ValueError
            If expected feature columns are missing.
        """

        self._require_fitted()
        validated_df = _validate_dataframe(df)
        engineered_df = self._engineer_features(validated_df)

        missing_columns = [column for column in self.feature_columns_ if column not in engineered_df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns required for transformation: {', '.join(missing_columns)}")

        ordered_df = engineered_df.reindex(columns=self.feature_columns_)
        scaled_values = self.feature_scaler.transform(ordered_df)
        transformed_df = pd.DataFrame(scaled_values, index=ordered_df.index, columns=self.feature_columns_)
        LOGGER.info("Transformed DataFrame with shape %s.", transformed_df.shape)
        return transformed_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the preprocessor and transform the input data in one step.

        Parameters
        ----------
        df:
            DataFrame to fit and transform.

        Returns
        -------
        pd.DataFrame
            Scaled feature DataFrame.
        """

        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, predictions: np.ndarray) -> np.ndarray:
        """Convert scaled predictions back to the original price scale.

        Parameters
        ----------
        predictions:
            Scaled model predictions.

        Returns
        -------
        np.ndarray
            Predictions converted back to the original scale.

        Raises
        ------
        RuntimeError
            If the preprocessor has not been fitted.
        ValueError
            If ``predictions`` does not have a single target dimension.
        """

        self._require_fitted()

        array = np.asarray(predictions)
        if array.ndim == 1:
            array = array.reshape(-1, 1)

        if array.ndim != 2 or array.shape[1] != 1:
            raise ValueError("predictions must be a one-dimensional array or a single-column 2D array.")

        inverse_values = self.target_scaler.inverse_transform(array)
        return inverse_values.ravel()

    def save_artifacts(self) -> None:
        """Persist the fitted scaler objects and feature column order.

        Raises
        ------
        RuntimeError
            If the preprocessor has not been fitted.
        """

        self._require_fitted()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        scaler_payload = {
            "feature_scaler": self.feature_scaler,
            "target_scaler": self.target_scaler,
            "target_column": self.target_column,
        }

        joblib.dump(scaler_payload, self.scaler_path)
        with self.feature_columns_path.open("w", encoding="utf-8") as file_handle:
            json.dump(self.feature_columns_, file_handle, indent=2)

        LOGGER.info("Saved scaler to %s and feature columns to %s.", self.scaler_path, self.feature_columns_path)

    def load_artifacts(self) -> FinSightPreprocessor:
        """Load persisted scaler objects and feature column order.

        Returns
        -------
        FinSightPreprocessor
            The preprocessor instance populated from saved artifacts.

        Raises
        ------
        FileNotFoundError
            If either artifact file is missing.
        ValueError
            If the saved artifact payload is invalid.
        """

        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Missing scaler artifact: {self.scaler_path}")

        if not self.feature_columns_path.exists():
            raise FileNotFoundError(f"Missing feature columns artifact: {self.feature_columns_path}")

        scaler_payload = joblib.load(self.scaler_path)
        if not isinstance(scaler_payload, dict):
            raise ValueError("Invalid scaler artifact format.")

        feature_scaler = scaler_payload.get("feature_scaler")
        target_scaler = scaler_payload.get("target_scaler")
        target_column = scaler_payload.get("target_column", self.target_column)

        if not isinstance(feature_scaler, StandardScaler) or not isinstance(target_scaler, StandardScaler):
            raise ValueError("Scaler artifact does not contain valid StandardScaler objects.")

        with self.feature_columns_path.open("r", encoding="utf-8") as file_handle:
            feature_columns = json.load(file_handle)

        if not isinstance(feature_columns, list) or not all(isinstance(column, str) for column in feature_columns):
            raise ValueError("Feature columns artifact is invalid.")

        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.target_column = target_column
        self.feature_columns_ = feature_columns
        self.fitted_ = True
        LOGGER.info("Loaded preprocessing artifacts from %s and %s.", self.scaler_path, self.feature_columns_path)
        return self

    def _require_fitted(self) -> None:
        """Ensure the preprocessor has been fitted or loaded."""

        if not self.fitted_:
            raise RuntimeError("FinSightPreprocessor must be fitted or loaded before use.")
