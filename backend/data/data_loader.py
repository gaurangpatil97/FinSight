"""Utilities for loading and caching stock market data for FinSight.

This module fetches OHLCV data from Yahoo Finance, caches the raw dataset to
Parquet, and returns a cleaned pandas DataFrame ready for downstream analysis.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


LOGGER = logging.getLogger(__name__)
DEFAULT_STOCKS = ["RELIANCE.NS", "HDFCBANK.NS", "SBIN.NS", "^NSEI"]
DATA_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = DATA_DIR / "raw"


def _ensure_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *frame* indexed by a sorted DatetimeIndex.

    Parameters
    ----------
    frame:
        Input DataFrame returned by Yahoo Finance or loaded from cache.

    Returns
    -------
    pd.DataFrame
        A copy of the input with a normalized DatetimeIndex.
    """

    cleaned_frame = frame.copy()

    if not isinstance(cleaned_frame.index, pd.DatetimeIndex):
        cleaned_frame.index = pd.to_datetime(cleaned_frame.index, errors="coerce")

    cleaned_frame = cleaned_frame.loc[~cleaned_frame.index.isna()]
    cleaned_frame = cleaned_frame.sort_index()
    cleaned_frame.index.name = "Date"
    return cleaned_frame


def _validate_date_range(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate and normalize the requested date range.

    Parameters
    ----------
    start:
        Inclusive start date in a format accepted by pandas.
    end:
        Inclusive end date in a format accepted by pandas.

    Returns
    -------
    tuple[pd.Timestamp, pd.Timestamp]
        Normalized timestamps for the requested range.

    Raises
    ------
    ValueError
        If the dates cannot be parsed or the range is invalid.
    """

    start_ts = pd.to_datetime(start, errors="coerce")
    end_ts = pd.to_datetime(end, errors="coerce")

    if pd.isna(start_ts) or pd.isna(end_ts):
        raise ValueError("Start and end dates must be valid date strings.")

    if start_ts > end_ts:
        raise ValueError("Start date must be before or equal to end date.")

    return start_ts.normalize(), end_ts.normalize()


def _cache_path(ticker: str) -> Path:
    """Build the cache path for a ticker symbol.

    Parameters
    ----------
    ticker:
        Stock ticker symbol.

    Returns
    -------
    pathlib.Path
        Absolute path to the cached Parquet file.
    """

    safe_ticker = re.sub(r"[^A-Za-z0-9._^-]+", "_", ticker.strip())
    return RAW_DATA_DIR / f"{safe_ticker}.parquet"


def _add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns used by FinSight analyses.

    Parameters
    ----------
    frame:
        Clean OHLCV data.

    Returns
    -------
    pd.DataFrame
        DataFrame with daily_return, log_return, and hl_spread added.
    """

    enriched_frame = frame.copy()
    enriched_frame["daily_return"] = enriched_frame["Close"].pct_change()
    enriched_frame["log_return"] = np.log(enriched_frame["Close"] / enriched_frame["Close"].shift(1))
    enriched_frame["hl_spread"] = enriched_frame["High"] - enriched_frame["Low"]
    return enriched_frame


def _clean_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean raw OHLCV data by filling missing values and removing leftovers.

    Parameters
    ----------
    frame:
        DataFrame to clean.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with missing values handled.
    """

    cleaned_frame = frame.copy()
    cleaned_frame = cleaned_frame.ffill().dropna()
    return cleaned_frame


def _prepare_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize, clean, and enrich a stock DataFrame.

    Parameters
    ----------
    frame:
        Raw or cached OHLCV data.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with derived columns and a DatetimeIndex.
    """

    prepared_frame = _ensure_datetime_index(frame)

    expected_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_columns = [column for column in expected_columns if column not in prepared_frame.columns]
    if missing_columns:
        raise ValueError(f"Missing expected OHLCV columns: {', '.join(missing_columns)}")

    selected_columns = expected_columns.copy()
    if "Adj Close" in prepared_frame.columns:
        selected_columns.append("Adj Close")

    prepared_frame = prepared_frame[selected_columns].copy()
    prepared_frame = _clean_ohlcv(prepared_frame)
    prepared_frame = _add_derived_columns(prepared_frame)
    prepared_frame = prepared_frame.replace([np.inf, -np.inf], np.nan).dropna()
    return prepared_frame


def _load_cached_data(cache_path: Path, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame | None:
    """Load cached data if it exists and covers the requested date range.

    Parameters
    ----------
    cache_path:
        Path to the cached Parquet file.
    start_ts:
        Requested start date.
    end_ts:
        Requested end date.

    Returns
    -------
    pd.DataFrame | None
        Cached data if available and usable; otherwise ``None``.
    """

    if not cache_path.exists():
        return None

    try:
        cached_frame = pd.read_parquet(cache_path)
    except Exception as exc:  # pragma: no cover - defensive cache recovery
        LOGGER.warning("Failed to read cached data from %s: %s", cache_path, exc)
        return None

    cached_frame = _ensure_datetime_index(cached_frame)
    if cached_frame.empty:
        return None

    cache_start = cached_frame.index.min()
    cache_end = cached_frame.index.max()
    if cache_start <= start_ts and cache_end >= end_ts:
        LOGGER.info("Loading %s from cache: %s", cache_path.stem, cache_path)
        return cached_frame.loc[start_ts:end_ts]

    return None


def _fetch_from_yfinance(ticker: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker:
        Stock ticker symbol.
    start_ts:
        Inclusive start date.
    end_ts:
        Inclusive end date.

    Returns
    -------
    pd.DataFrame
        Raw OHLCV data.

    Raises
    ------
    ValueError
        If Yahoo Finance returns no data for the requested ticker.
    """

    LOGGER.info("Fetching %s from Yahoo Finance", ticker)
    raw_frame = yf.download(
        ticker,
        start=start_ts.strftime("%Y-%m-%d"),
        end=(end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
        group_by="column",
    )

    if raw_frame.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    if isinstance(raw_frame.columns, pd.MultiIndex):
        raw_frame.columns = raw_frame.columns.get_level_values(0)

    return raw_frame


def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch, cache, and clean OHLCV data for a single ticker.

    Parameters
    ----------
    ticker:
        Stock ticker symbol such as ``RELIANCE.NS`` or ``^NSEI``.
    start:
        Inclusive start date.
    end:
        Inclusive end date.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame indexed by date.

    Raises
    ------
    ValueError
        If the date range is invalid or no data is available.
    """

    if not ticker or not ticker.strip():
        raise ValueError("Ticker must be a non-empty string.")

    start_ts, end_ts = _validate_date_range(start, end)
    cache_path = _cache_path(ticker)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cached_frame = _load_cached_data(cache_path, start_ts, end_ts)
    if cached_frame is not None:
        return _prepare_dataframe(cached_frame)

    try:
        raw_frame = _fetch_from_yfinance(ticker, start_ts, end_ts)
    except Exception as exc:
        LOGGER.exception("Failed to fetch data for %s", ticker)
        raise ValueError(f"Unable to fetch data for ticker '{ticker}': {exc}") from exc

    prepared_frame = _prepare_dataframe(raw_frame)

    prepared_frame.to_parquet(cache_path)
    LOGGER.info("Saved %s to %s", ticker, cache_path)

    return prepared_frame.loc[start_ts:end_ts]


def fetch_multiple_stocks(tickers: list, start: str, end: str) -> dict:
    """Fetch stock data for multiple tickers.

    Parameters
    ----------
    tickers:
        List of ticker symbols to fetch.
    start:
        Inclusive start date.
    end:
        Inclusive end date.

    Returns
    -------
    dict
        Dictionary mapping each ticker to its cleaned DataFrame.

    Notes
    -----
    Tickers that fail to load are skipped and logged, allowing EDA workflows to
    continue with the remaining symbols.
    """

    if not isinstance(tickers, list):
        raise TypeError("tickers must be provided as a list.")

    results: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        if not isinstance(ticker, str) or not ticker.strip():
            LOGGER.warning("Skipping invalid ticker value: %r", ticker)
            continue

        try:
            results[ticker] = fetch_stock_data(ticker, start, end)
        except Exception as exc:
            LOGGER.error("Skipping %s after load failure: %s", ticker, exc)

    return results