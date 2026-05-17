"""Automated financial time series EDA for FinSight.

This module builds a complete exploratory analysis pipeline for a cleaned stock
price DataFrame and saves every generated Plotly figure as HTML under a
ticker-specific output directory.
"""

from __future__ import annotations

import logging
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf


LOGGER = logging.getLogger(__name__)
EDA_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = EDA_DIR / "outputs"


def _validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that the input is a pandas DataFrame and return a copy.

    Parameters
    ----------
    df:
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        A copy of the validated DataFrame.

    Raises
    ------
    TypeError
        If ``df`` is not a pandas DataFrame.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    return df.copy()


def _safe_ticker_name(ticker: str) -> str:
    """Convert a ticker into a filesystem-safe name."""

    return re.sub(r"[^A-Za-z0-9]+", "_", ticker.strip()) or "default"


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame uses a sorted DatetimeIndex."""

    prepared_df = df.copy()
    if not isinstance(prepared_df.index, pd.DatetimeIndex):
        prepared_df.index = pd.to_datetime(prepared_df.index, errors="coerce")

    prepared_df = prepared_df.loc[~prepared_df.index.isna()].sort_index()
    prepared_df.index.name = "Date"
    return prepared_df


def _build_empty_figure(title: str, message: str) -> go.Figure:
    """Create a placeholder figure with a centered message."""

    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14),
            )
        ],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=450,
    )
    return fig


class FinSightEDA:
    """Automated exploratory data analysis for financial time series."""

    def __init__(self) -> None:
        """Initialize the EDA pipeline with default output context."""

        self.ticker = "default"
        self.safe_ticker = _safe_ticker_name(self.ticker)
        self.output_dir = OUTPUTS_DIR / self.safe_ticker

    def _set_context(self, ticker: str) -> None:
        """Set the active ticker and output directory for a run."""

        self.ticker = ticker
        self.safe_ticker = _safe_ticker_name(ticker)
        self.output_dir = OUTPUTS_DIR / self.safe_ticker
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_figure(self, fig: go.Figure, filename: str) -> str:
        """Save a Plotly figure as HTML in the ticker output directory."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        html_path = self.output_dir / filename
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        LOGGER.info("Saved figure to %s", html_path)
        return str(html_path)

    def _prepare_base_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize the base DataFrame used by all analyses."""

        prepared_df = _validate_dataframe(df)
        prepared_df = _ensure_datetime_index(prepared_df)

        if "Close" not in prepared_df.columns:
            raise ValueError("Input DataFrame must contain a 'Close' column.")

        prepared_df = prepared_df.dropna(subset=["Close"])
        if prepared_df.empty:
            raise ValueError("Input DataFrame does not contain enough non-null Close values.")

        return prepared_df

    def _get_returns(self, df: pd.DataFrame) -> pd.Series:
        """Compute daily log returns from the Close column."""

        returns = np.log(df["Close"] / df["Close"].shift(1))
        return returns.dropna()

    def _get_price_range(self, df: pd.DataFrame) -> tuple[float, float]:
        """Return a sensible plotting range for price-based charts."""

        values = pd.concat([df[c] for c in ["Close", "High", "Low"] if c in df.columns], axis=0).dropna()
        if values.empty:
            return 0.0, 1.0

        low = float(values.min())
        high = float(values.max())
        if low == high:
            high = low + 1.0
        return low, high

    def _save_and_wrap(self, fig: go.Figure, filename: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Attach the saved HTML path to a payload and return it."""

        payload = dict(payload)
        payload["fig"] = fig
        payload["html_path"] = self._save_figure(fig, filename)
        return payload

    def run_all(self, df: pd.DataFrame, ticker: str, all_stocks_dict: dict | None = None) -> dict:
        """Run the full EDA suite and save all figures.

        Parameters
        ----------
        df:
            Clean stock DataFrame.
        ticker:
            Ticker symbol used for output organization.
        all_stocks_dict:
            Optional dictionary of additional stock DataFrames for correlation
            analysis.

        Returns
        -------
        dict
            Dictionary containing all analysis results and Plotly figures.
        """

        self._set_context(ticker)
        prepared_df = self._prepare_base_frame(df)

        results: dict[str, Any] = {
            "ticker": ticker,
            "trend_analysis": self.trend_analysis(prepared_df),
            "volatility_analysis": self.volatility_analysis(prepared_df),
            "rolling_statistics": self.rolling_statistics(prepared_df),
            "seasonality_analysis": self.seasonality_analysis(prepared_df),
            "distribution_analysis": self.distribution_analysis(prepared_df),
        }

        correlation_input = all_stocks_dict or {}
        if ticker not in correlation_input:
            correlation_input = dict(correlation_input)
            correlation_input[ticker] = prepared_df

        results["correlation_analysis"] = self.correlation_analysis(prepared_df, correlation_input)
        return results

    def trend_analysis(self, df: pd.DataFrame) -> dict:
        """Analyze trend using moving averages and exponential moving averages."""

        try:
            prepared_df = self._prepare_base_frame(df)
            min_required = 20
            if len(prepared_df) < min_required:
                fig = _build_empty_figure("Trend Analysis", "Not enough data for moving averages.")
                return self._save_and_wrap(fig, "trend_analysis.html", {"trend_direction": "insufficient_data"})

            analysis_df = prepared_df.copy()
            analysis_df["SMA_20"] = analysis_df["Close"].rolling(window=20).mean()
            analysis_df["SMA_50"] = analysis_df["Close"].rolling(window=50).mean()
            analysis_df["SMA_200"] = analysis_df["Close"].rolling(window=200).mean()
            analysis_df["EMA_12"] = analysis_df["Close"].ewm(span=12, adjust=False).mean()
            analysis_df["EMA_26"] = analysis_df["Close"].ewm(span=26, adjust=False).mean()

            latest = analysis_df.dropna(subset=["SMA_20", "SMA_50", "EMA_12", "EMA_26"])
            if latest.empty:
                trend_direction = "insufficient_data"
            else:
                row = latest.iloc[-1]
                if row["Close"] > row["SMA_20"] > row["SMA_50"] and row["EMA_12"] > row["EMA_26"]:
                    trend_direction = "bullish"
                elif row["Close"] < row["SMA_20"] < row["SMA_50"] and row["EMA_12"] < row["EMA_26"]:
                    trend_direction = "bearish"
                else:
                    trend_direction = "sideways"

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=analysis_df.index, y=analysis_df["Close"], name="Close", line=dict(color="#111827", width=2)))
            fig.add_trace(go.Scatter(x=analysis_df.index, y=analysis_df["SMA_20"], name="SMA 20", line=dict(color="#2563eb")))
            fig.add_trace(go.Scatter(x=analysis_df.index, y=analysis_df["SMA_50"], name="SMA 50", line=dict(color="#f59e0b")))
            fig.add_trace(go.Scatter(x=analysis_df.index, y=analysis_df["SMA_200"], name="SMA 200", line=dict(color="#dc2626")))
            fig.add_trace(go.Scatter(x=analysis_df.index, y=analysis_df["EMA_12"], name="EMA 12", line=dict(color="#10b981", dash="dot")))
            fig.add_trace(go.Scatter(x=analysis_df.index, y=analysis_df["EMA_26"], name="EMA 26", line=dict(color="#8b5cf6", dash="dot")))
            fig.update_layout(
                title=f"{self.ticker} Trend Analysis",
                template="plotly_white",
                xaxis_title="Date",
                yaxis_title="Price",
                legend_title="Indicators",
                height=550,
            )

            summary = {
                "trend_direction": trend_direction,
                "latest_close": float(latest["Close"].iloc[-1]) if not latest.empty else None,
                "latest_sma_20": float(latest["SMA_20"].iloc[-1]) if not latest.empty else None,
                "latest_sma_50": float(latest["SMA_50"].iloc[-1]) if not latest.empty else None,
                "latest_ema_12": float(latest["EMA_12"].iloc[-1]) if not latest.empty else None,
                "latest_ema_26": float(latest["EMA_26"].iloc[-1]) if not latest.empty else None,
            }
            return self._save_and_wrap(fig, "trend_analysis.html", summary)
        except Exception:
            LOGGER.exception("Trend analysis failed for %s", self.ticker)
            fig = _build_empty_figure("Trend Analysis", "Trend analysis could not be completed.")
            return self._save_and_wrap(fig, "trend_analysis.html", {"trend_direction": "error"})

    def volatility_analysis(self, df: pd.DataFrame) -> dict:
        """Analyze volatility using rolling log-return volatility, Bollinger Bands, and ATR."""

        try:
            prepared_df = self._prepare_base_frame(df)
            if len(prepared_df) < 20:
                fig = _build_empty_figure("Volatility Analysis", "Not enough data for volatility indicators.")
                return self._save_and_wrap(fig, "volatility_analysis.html", {"volatility_level": "insufficient_data"})

            analysis_df = prepared_df.copy()
            returns = self._get_returns(analysis_df)
            analysis_df = analysis_df.loc[returns.index]
            analysis_df["log_return"] = returns
            analysis_df["rolling_volatility_20"] = returns.rolling(window=20).std()
            analysis_df["bb_mid"] = analysis_df["Close"].rolling(window=20).mean()
            analysis_df["bb_std"] = analysis_df["Close"].rolling(window=20).std()
            analysis_df["bb_upper"] = analysis_df["bb_mid"] + 2 * analysis_df["bb_std"]
            analysis_df["bb_lower"] = analysis_df["bb_mid"] - 2 * analysis_df["bb_std"]

            previous_close = analysis_df["Close"].shift(1)
            true_range = pd.concat(
                [
                    analysis_df["High"] - analysis_df["Low"],
                    (analysis_df["High"] - previous_close).abs(),
                    (analysis_df["Low"] - previous_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            analysis_df["atr_14"] = true_range.rolling(window=14).mean()

            rolling_vol = analysis_df["rolling_volatility_20"].dropna()
            if rolling_vol.empty:
                volatility_level = "insufficient_data"
            else:
                current_vol = float(rolling_vol.iloc[-1])
                low_threshold = float(rolling_vol.quantile(0.33))
                high_threshold = float(rolling_vol.quantile(0.67))
                if current_vol <= low_threshold:
                    volatility_level = "low"
                elif current_vol >= high_threshold:
                    volatility_level = "high"
                else:
                    volatility_level = "medium"

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                subplot_titles=("20-Day Rolling Log-Return Volatility", "Bollinger Bands (20, 2σ)", "ATR 14"))

            fig.add_trace(go.Scatter(x=analysis_df.index, y=analysis_df["rolling_volatility_20"], name="Rolling Volatility 20", line=dict(color="#2563eb")), row=1, col=1)

            fig.add_trace(go.Scatter(x=analysis_df.index, y=analysis_df["Close"], name="Close", line=dict(color="#111827")), row=2, col=1)
            fig.add_trace(go.Scatter(x=analysis_df.index, y=analysis_df["bb_upper"], name="Upper Band", line=dict(color="#ef4444", dash="dash")), row=2, col=1)
            fig.add_trace(go.Scatter(x=analysis_df.index, y=analysis_df["bb_mid"], name="Middle Band", line=dict(color="#64748b")), row=2, col=1)
            fig.add_trace(go.Scatter(x=analysis_df.index, y=analysis_df["bb_lower"], name="Lower Band", line=dict(color="#ef4444", dash="dash")), row=2, col=1)

            fig.add_trace(go.Scatter(x=analysis_df.index, y=analysis_df["atr_14"], name="ATR 14", line=dict(color="#10b981")), row=3, col=1)

            fig.update_layout(title=f"{self.ticker} Volatility Analysis", template="plotly_white", height=850, legend_title="Indicators")
            summary = {
                "volatility_level": volatility_level,
                "latest_rolling_volatility_20": float(rolling_vol.iloc[-1]) if not rolling_vol.empty else None,
                "latest_atr_14": float(analysis_df["atr_14"].dropna().iloc[-1]) if not analysis_df["atr_14"].dropna().empty else None,
            }
            return self._save_and_wrap(fig, "volatility_analysis.html", summary)
        except Exception:
            LOGGER.exception("Volatility analysis failed for %s", self.ticker)
            fig = _build_empty_figure("Volatility Analysis", "Volatility analysis could not be completed.")
            return self._save_and_wrap(fig, "volatility_analysis.html", {"volatility_level": "error"})

    def rolling_statistics(self, df: pd.DataFrame) -> dict:
        """Compute rolling statistics for log returns over a 30-day window."""

        try:
            prepared_df = self._prepare_base_frame(df)
            returns = self._get_returns(prepared_df)
            if len(returns) < 30:
                fig = _build_empty_figure("Rolling Statistics", "Not enough data for 30-day rolling statistics.")
                return self._save_and_wrap(fig, "rolling_statistics.html", {"latest_values": {}})

            stats_df = pd.DataFrame(index=returns.index)
            stats_df["rolling_mean_30"] = returns.rolling(window=30).mean()
            stats_df["rolling_std_30"] = returns.rolling(window=30).std()
            stats_df["rolling_skew_30"] = returns.rolling(window=30).skew()
            stats_df["rolling_kurtosis_30"] = returns.rolling(window=30).kurt()

            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                subplot_titles=("Rolling Mean", "Rolling Std", "Rolling Skewness", "Rolling Kurtosis"))
            fig.add_trace(go.Scatter(x=stats_df.index, y=stats_df["rolling_mean_30"], name="Rolling Mean", line=dict(color="#2563eb")), row=1, col=1)
            fig.add_trace(go.Scatter(x=stats_df.index, y=stats_df["rolling_std_30"], name="Rolling Std", line=dict(color="#f59e0b")), row=2, col=1)
            fig.add_trace(go.Scatter(x=stats_df.index, y=stats_df["rolling_skew_30"], name="Rolling Skewness", line=dict(color="#10b981")), row=3, col=1)
            fig.add_trace(go.Scatter(x=stats_df.index, y=stats_df["rolling_kurtosis_30"], name="Rolling Kurtosis", line=dict(color="#8b5cf6")), row=4, col=1)
            fig.update_layout(title=f"{self.ticker} Rolling Statistics", template="plotly_white", height=1000)

            latest_row = stats_df.dropna().iloc[-1] if not stats_df.dropna().empty else None
            latest_values = {
                "rolling_mean_30": float(latest_row["rolling_mean_30"]) if latest_row is not None else None,
                "rolling_std_30": float(latest_row["rolling_std_30"]) if latest_row is not None else None,
                "rolling_skew_30": float(latest_row["rolling_skew_30"]) if latest_row is not None else None,
                "rolling_kurtosis_30": float(latest_row["rolling_kurtosis_30"]) if latest_row is not None else None,
            }
            return self._save_and_wrap(fig, "rolling_statistics.html", {"latest_values": latest_values})
        except Exception:
            LOGGER.exception("Rolling statistics failed for %s", self.ticker)
            fig = _build_empty_figure("Rolling Statistics", "Rolling statistics could not be completed.")
            return self._save_and_wrap(fig, "rolling_statistics.html", {"latest_values": {}})

    def seasonality_analysis(self, df: pd.DataFrame) -> dict:
        """Analyze seasonal structure with STL decomposition, ACF/PACF, and monthly heatmap."""

        try:
            prepared_df = self._prepare_base_frame(df)
            series = prepared_df["Close"].dropna()
            if len(series) < 30:
                fig = _build_empty_figure("Seasonality Analysis", "Not enough data for seasonality analysis.")
                return self._save_and_wrap(fig, "seasonality_analysis.html", {"dominant_seasonal_period": None})

            returns = self._get_returns(prepared_df)
            seasonality_period = self._infer_seasonal_period(series)

            if seasonality_period is None:
                seasonality_period = min(30, max(7, len(series) // 10))

            seasonality_period = max(2, seasonality_period)

            stl_input = series.asfreq("B") if series.index.inferred_freq is None else series
            stl_input = stl_input.interpolate(method="time").dropna()
            if len(stl_input) <= seasonality_period * 2:
                fig = _build_empty_figure("Seasonality Analysis", "Not enough data for STL decomposition.")
                return self._save_and_wrap(fig, "seasonality_analysis.html", {"dominant_seasonal_period": seasonality_period})

            stl_result = STL(stl_input, period=seasonality_period, robust=True).fit()

            acf_lags = min(40, len(returns) - 1)
            acf_values = acf(returns, nlags=acf_lags, fft=True) if acf_lags >= 1 else np.array([1.0])
            pacf_values = pacf(returns, nlags=acf_lags, method="ywmle") if acf_lags >= 1 else np.array([1.0])

            monthly_returns = prepared_df["Close"].resample("ME").last().pct_change().dropna()
            monthly_heatmap = monthly_returns.to_frame(name="monthly_return")
            monthly_heatmap["year"] = monthly_heatmap.index.year
            monthly_heatmap["month"] = monthly_heatmap.index.month
            pivot = monthly_heatmap.pivot(index="year", columns="month", values="monthly_return").sort_index()
            pivot = pivot.reindex(columns=list(range(1, 13)))

            fig = make_subplots(
                rows=4,
                cols=1,
                shared_xaxes=False,
                vertical_spacing=0.08,
                subplot_titles=("STL Decomposition", "ACF (lags=40)", "PACF (lags=40)", "Monthly Return Heatmap"),
                specs=[[{}], [{}], [{}], [{"type": "heatmap"}]],
            )
            fig.add_trace(go.Scatter(x=stl_result.trend.index, y=stl_result.trend, name="Trend", line=dict(color="#2563eb")), row=1, col=1)
            fig.add_trace(go.Scatter(x=stl_result.seasonal.index, y=stl_result.seasonal, name="Seasonal", line=dict(color="#f59e0b")), row=1, col=1)
            fig.add_trace(go.Scatter(x=stl_result.resid.index, y=stl_result.resid, name="Residual", line=dict(color="#10b981")), row=1, col=1)

            fig.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values, name="ACF", marker_color="#2563eb"), row=2, col=1)
            fig.add_trace(go.Bar(x=list(range(len(pacf_values))), y=pacf_values, name="PACF", marker_color="#8b5cf6"), row=3, col=1)

            month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=month_labels,
                    y=pivot.index.astype(str),
                    colorscale="RdBu",
                    zmid=0,
                    colorbar=dict(title="Monthly Return"),
                ),
                row=4,
                col=1,
            )
            fig.update_layout(title=f"{self.ticker} Seasonality Analysis", template="plotly_white", height=1200)

            seasonal_component = stl_result.seasonal.dropna()
            dominant_period = None
            if len(acf_values) > 2:
                candidate_lags = np.arange(1, len(acf_values))
                candidate_values = np.abs(acf_values[1:])
                if candidate_values.size:
                    best_idx = int(np.argmax(candidate_values))
                    if candidate_values[best_idx] >= 0.2:
                        dominant_period = int(candidate_lags[best_idx])
            if dominant_period is None and not seasonal_component.empty:
                dominant_period = seasonality_period

            return self._save_and_wrap(fig, "seasonality_analysis.html", {"dominant_seasonal_period": dominant_period})
        except Exception:
            LOGGER.exception("Seasonality analysis failed for %s", self.ticker)
            fig = _build_empty_figure("Seasonality Analysis", "Seasonality analysis could not be completed.")
            return self._save_and_wrap(fig, "seasonality_analysis.html", {"dominant_seasonal_period": None})

    def _infer_seasonal_period(self, series: pd.Series) -> int | None:
        """Infer a likely seasonal period from the series autocorrelation."""

        clean_series = series.dropna()
        if len(clean_series) < 10:
            return None

        max_lag = min(40, len(clean_series) // 2)
        if max_lag < 2:
            return None

        series_acf = acf(clean_series, nlags=max_lag, fft=True)
        if len(series_acf) <= 2:
            return None

        candidate_lags = np.arange(1, len(series_acf))
        candidate_values = np.abs(series_acf[1:])
        if not candidate_values.size:
            return None

        best_idx = int(np.argmax(candidate_values))
        if candidate_values[best_idx] < 0.2:
            return None

        return int(candidate_lags[best_idx])

    def distribution_analysis(self, df: pd.DataFrame) -> dict:
        """Analyze the distribution of log returns and test normality."""

        try:
            prepared_df = self._prepare_base_frame(df)
            returns = self._get_returns(prepared_df)
            if len(returns) < 10:
                fig = _build_empty_figure("Distribution Analysis", "Not enough data for distribution analysis.")
                return self._save_and_wrap(fig, "distribution_analysis.html", {"normality_verdict": "insufficient_data"})

            mean = float(returns.mean())
            std = float(returns.std(ddof=1))
            skewness = float(stats.skew(returns, bias=False))
            kurtosis = float(stats.kurtosis(returns, fisher=True, bias=False))
            jb_stat, jb_pvalue = stats.jarque_bera(returns)

            if std == 0 or np.isnan(std):
                std = 1e-8

            hist_x = np.linspace(returns.min(), returns.max(), 200)
            normal_curve = stats.norm.pdf(hist_x, loc=mean, scale=std)
            qq_osm, qq_osr = stats.probplot(returns, dist="norm", fit=False)
            qq_slope, qq_intercept, _ = stats.probplot(returns, dist="norm", fit=True)[1]

            fig = make_subplots(rows=1, cols=2, subplot_titles=("Log Return Distribution", "QQ Plot"))
            fig.add_trace(go.Histogram(x=returns, histnorm="probability density", name="Log Returns", marker_color="#2563eb", opacity=0.7), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist_x, y=normal_curve, name="Normal Curve", line=dict(color="#ef4444", width=2)), row=1, col=1)

            fig.add_trace(go.Scatter(x=qq_osm, y=qq_osr, mode="markers", name="QQ Points", marker=dict(color="#2563eb")), row=1, col=2)
            qq_line_x = np.array([min(qq_osm), max(qq_osm)])
            qq_line_y = qq_slope * qq_line_x + qq_intercept
            fig.add_trace(go.Scatter(x=qq_line_x, y=qq_line_y, mode="lines", name="Reference Line", line=dict(color="#ef4444")), row=1, col=2)

            fig.update_layout(title=f"{self.ticker} Distribution Analysis", template="plotly_white", height=600, bargap=0.1)

            normality_verdict = "normal" if float(jb_pvalue) >= 0.05 else "non-normal"
            summary = {
                "mean": mean,
                "std": std,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "jarque_bera": {"statistic": float(jb_stat), "p_value": float(jb_pvalue)},
                "normality_verdict": normality_verdict,
            }
            return self._save_and_wrap(fig, "distribution_analysis.html", summary)
        except Exception:
            LOGGER.exception("Distribution analysis failed for %s", self.ticker)
            fig = _build_empty_figure("Distribution Analysis", "Distribution analysis could not be completed.")
            return self._save_and_wrap(fig, "distribution_analysis.html", {"normality_verdict": "error"})

    def correlation_analysis(self, df: pd.DataFrame, all_stocks_dict: dict | None) -> dict:
        """Analyze cross-stock correlations using rolling and static correlation views."""

        try:
            if not all_stocks_dict:
                fig = _build_empty_figure("Correlation Analysis", "Provide at least two stock DataFrames for correlation analysis.")
                return self._save_and_wrap(fig, "correlation_analysis.html", {
                    "strongest_correlation_pair": None,
                    "weakest_correlation_pair": None,
                })

            series_map: dict[str, pd.Series] = {}
            for ticker, stock_df in all_stocks_dict.items():
                if not isinstance(stock_df, pd.DataFrame):
                    LOGGER.warning("Skipping non-DataFrame value for ticker %s", ticker)
                    continue
                if "Close" not in stock_df.columns:
                    LOGGER.warning("Skipping %s because 'Close' is missing.", ticker)
                    continue

                prepared_stock_df = _ensure_datetime_index(stock_df)
                prepared_stock_df = prepared_stock_df.dropna(subset=["Close"])
                if len(prepared_stock_df) < 2:
                    continue

                series_map[str(ticker)] = np.log(prepared_stock_df["Close"] / prepared_stock_df["Close"].shift(1)).rename(str(ticker))

            if len(series_map) < 2:
                fig = _build_empty_figure("Correlation Analysis", "Not enough stocks with valid returns for correlation analysis.")
                return self._save_and_wrap(fig, "correlation_analysis.html", {
                    "strongest_correlation_pair": None,
                    "weakest_correlation_pair": None,
                })

            returns_df = pd.concat(series_map.values(), axis=1, join="inner").dropna(how="all")
            if returns_df.empty:
                fig = _build_empty_figure("Correlation Analysis", "No overlapping return history was found.")
                return self._save_and_wrap(fig, "correlation_analysis.html", {
                    "strongest_correlation_pair": None,
                    "weakest_correlation_pair": None,
                })

            correlation_matrix = returns_df.corr()
            upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            pair_values: list[tuple[tuple[str, str], float]] = []
            for left, right in combinations(correlation_matrix.columns, 2):
                coefficient = correlation_matrix.loc[left, right]
                if pd.notna(coefficient):
                    pair_values.append(((left, right), float(coefficient)))

            strongest_pair = None
            weakest_pair = None
            if pair_values:
                strongest_pair = max(pair_values, key=lambda item: abs(item[1]))
                weakest_pair = min(pair_values, key=lambda item: abs(item[1]))

            rolling_correlations: dict[str, pd.Series] = {}
            for left, right in combinations(returns_df.columns, 2):
                pair_series = returns_df[[left, right]].dropna()
                if len(pair_series) < 30:
                    continue
                rolling_correlations[f"{left} vs {right}"] = pair_series[left].rolling(window=30).corr(pair_series[right])

            fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.08,
                                subplot_titles=("Rolling 30-Day Correlations", "Return Correlation Heatmap"))

            if rolling_correlations:
                for name, corr_series in rolling_correlations.items():
                    fig.add_trace(go.Scatter(x=corr_series.index, y=corr_series, name=name, mode="lines"), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=[], y=[], name="Rolling Correlation", mode="lines"), row=1, col=1)

            fig.add_trace(
                go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns.tolist(),
                    y=correlation_matrix.index.tolist(),
                    zmin=-1,
                    zmax=1,
                    colorscale="RdBu",
                    colorbar=dict(title="Correlation"),
                ),
                row=2,
                col=1,
            )
            fig.update_layout(title=f"{self.ticker} Correlation Analysis", template="plotly_white", height=900)

            summary = {
                "strongest_correlation_pair": {
                    "pair": strongest_pair[0] if strongest_pair else None,
                    "correlation": strongest_pair[1] if strongest_pair else None,
                },
                "weakest_correlation_pair": {
                    "pair": weakest_pair[0] if weakest_pair else None,
                    "correlation": weakest_pair[1] if weakest_pair else None,
                },
            }
            return self._save_and_wrap(fig, "correlation_analysis.html", summary)
        except Exception:
            LOGGER.exception("Correlation analysis failed for %s", self.ticker)
            fig = _build_empty_figure("Correlation Analysis", "Correlation analysis could not be completed.")
            return self._save_and_wrap(fig, "correlation_analysis.html", {
                "strongest_correlation_pair": None,
                "weakest_correlation_pair": None,
            })
