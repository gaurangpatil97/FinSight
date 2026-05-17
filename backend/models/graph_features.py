"""Graph-based stock feature engineering for FinSight.

This module builds a correlation graph from multiple stock return series and
extracts graph-derived features that can be used to enrich downstream machine
learning models.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go


LOGGER = logging.getLogger(__name__)
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def _safe_ticker_name(ticker: str) -> str:
    """Convert a ticker into a filesystem-safe name."""

    return re.sub(r"[^A-Za-z0-9]", "_", ticker.strip()) or "default"


def _validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that the input is a pandas DataFrame and return a copy."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    return df.copy()


class StockGraphBuilder:
    """Build a stock correlation graph and derive graph-based features."""

    def __init__(self, tickers: list[str], window: int = 30) -> None:
        """Initialize the graph builder.

        Parameters
        ----------
        tickers:
            List of stock ticker symbols.
        window:
            Rolling window used for correlation calculations.
        """

        self.tickers = tickers
        self.window = window
        self.graph = nx.Graph()
        self.adjacency_matrix: pd.DataFrame | None = None
        self.rolling_correlations: dict[tuple[str, str], pd.Series] = {}

    def _get_log_returns(self, df: pd.DataFrame) -> pd.Series:
        """Compute log returns from a stock DataFrame."""

        if "Close" not in df.columns:
            raise ValueError("Each stock DataFrame must contain a 'Close' column.")

        close = _validate_dataframe(df)["Close"].astype(float)
        returns = np.log(close / close.shift(1))
        return returns.dropna()

    def build_graph(self, stocks_dict: dict) -> None:
        """Build the correlation graph from a dictionary of stock DataFrames.

        Parameters
        ----------
        stocks_dict:
            Dictionary mapping ticker symbols to cleaned stock DataFrames.
        """

        try:
            if not isinstance(stocks_dict, dict):
                raise TypeError("stocks_dict must be a dictionary.")

            valid_returns: dict[str, pd.Series] = {}
            for ticker in self.tickers:
                stock_df = stocks_dict.get(ticker)
                if not isinstance(stock_df, pd.DataFrame) or stock_df.empty:
                    LOGGER.warning("Skipping %s due to missing or empty data.", ticker)
                    continue

                try:
                    valid_returns[ticker] = self._get_log_returns(stock_df)
                except Exception as exc:
                    LOGGER.warning("Skipping %s due to invalid price data: %s", ticker, exc)

            if not valid_returns:
                self.adjacency_matrix = pd.DataFrame()
                self.graph = nx.Graph()
                self.rolling_correlations = {}
                return

            returns_df = pd.concat(valid_returns, axis=1, join="inner")
            if returns_df.empty:
                self.adjacency_matrix = pd.DataFrame(index=list(valid_returns.keys()), columns=list(valid_returns.keys()))
                self.adjacency_matrix[:] = 0.0
                self.graph = nx.Graph()
                self.graph.add_nodes_from(valid_returns.keys())
                self.rolling_correlations = {}
                return

            adjacency = pd.DataFrame(
                np.eye(len(returns_df.columns)),
                index=returns_df.columns,
                columns=returns_df.columns,
                dtype=float,
            )

            self.rolling_correlations = {}
            for left in returns_df.columns:
                for right in returns_df.columns:
                    if left == right:
                        continue

                    pair_key = tuple(sorted((str(left), str(right))))
                    pair_df = returns_df[[left, right]].dropna()
                    if pair_df.empty or len(pair_df) < max(self.window, 2):
                        self.rolling_correlations[pair_key] = pd.Series(dtype=float)
                        adjacency.loc[left, right] = 0.0
                        continue

                    rolling_corr = pair_df[left].rolling(window=self.window).corr(pair_df[right]).dropna()
                    self.rolling_correlations[pair_key] = rolling_corr
                    last_value = float(rolling_corr.iloc[-1]) if not rolling_corr.empty else 0.0
                    adjacency.loc[left, right] = last_value

            self.adjacency_matrix = adjacency.fillna(0.0)

            graph = nx.Graph()
            graph.add_nodes_from(self.adjacency_matrix.index.tolist())
            for left in self.adjacency_matrix.index:
                for right in self.adjacency_matrix.columns:
                    if left >= right:
                        continue
                    weight = float(self.adjacency_matrix.loc[left, right])
                    if np.isfinite(weight):
                        graph.add_edge(left, right, weight=weight)

            self.graph = graph
            LOGGER.info("Built stock correlation graph with %d nodes.", self.graph.number_of_nodes())
        except Exception:
            LOGGER.exception("Failed to build stock correlation graph.")
            raise

    def extract_graph_features(self, ticker: str) -> dict:
        """Extract graph-based features for a specific ticker.

        Parameters
        ----------
        ticker:
            Ticker symbol for which to compute graph features.

        Returns
        -------
        dict
            Dictionary of graph features.
        """

        if self.adjacency_matrix is None or self.adjacency_matrix.empty:
            return {
                "degree_centrality": 0.0,
                "avg_correlation": 0.0,
                "max_correlation": 0.0,
                "min_correlation": 0.0,
                "correlation_std": 0.0,
            }

        if ticker not in self.adjacency_matrix.index:
            return {
                "degree_centrality": 0.0,
                "avg_correlation": 0.0,
                "max_correlation": 0.0,
                "min_correlation": 0.0,
                "correlation_std": 0.0,
            }

        correlations = self.adjacency_matrix.loc[ticker].drop(labels=[ticker], errors="ignore").astype(float)
        if correlations.empty:
            return {
                "degree_centrality": 0.0,
                "avg_correlation": 0.0,
                "max_correlation": 0.0,
                "min_correlation": 0.0,
                "correlation_std": 0.0,
            }

        valid_correlations = correlations.replace([np.inf, -np.inf], np.nan).dropna()
        if valid_correlations.empty:
            valid_correlations = pd.Series([0.0])

        return {
            "degree_centrality": float(valid_correlations.abs().sum()),
            "avg_correlation": float(valid_correlations.mean()),
            "max_correlation": float(valid_correlations.max()),
            "min_correlation": float(valid_correlations.min()),
            "correlation_std": float(valid_correlations.std(ddof=0)),
        }

    def enrich_dataframe(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Add graph features as constant columns to a DataFrame.

        Parameters
        ----------
        df:
            Input DataFrame to enrich.
        ticker:
            Ticker symbol whose graph features should be attached.

        Returns
        -------
        pd.DataFrame
            Enriched DataFrame with graph feature columns.
        """

        prepared_df = _validate_dataframe(df)
        features = self.extract_graph_features(ticker)
        for feature_name, feature_value in features.items():
            prepared_df[feature_name] = feature_value
        return prepared_df

    def plot_correlation_heatmap(self) -> go.Figure:
        """Plot the correlation adjacency matrix as a heatmap."""

        if self.adjacency_matrix is None or self.adjacency_matrix.empty:
            fig = go.Figure()
            fig.update_layout(
                title="Correlation Heatmap",
                template="plotly_white",
                annotations=[dict(text="No correlation data available.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)],
            )
            return fig

        fig = go.Figure(
            go.Heatmap(
                z=self.adjacency_matrix.values,
                x=self.adjacency_matrix.columns.tolist(),
                y=self.adjacency_matrix.index.tolist(),
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Correlation"),
            )
        )
        fig.update_layout(
            title="Correlation Heatmap",
            template="plotly_white",
            height=650,
        )
        return fig

    def plot_correlation_network(self) -> go.Figure:
        """Plot the stock correlation graph as a network diagram."""

        if self.graph.number_of_nodes() == 0:
            fig = go.Figure()
            fig.update_layout(
                title="Correlation Network",
                template="plotly_white",
                annotations=[dict(text="No graph data available.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)],
            )
            return fig

        positions = nx.spring_layout(self.graph, seed=42)

        edge_x: list[float] = []
        edge_y: list[float] = []
        edge_widths: list[float] = []
        for left, right, data in self.graph.edges(data=True):
            weight = float(data.get("weight", 0.0))
            if abs(weight) <= 0.3:
                continue
            x0, y0 = positions[left]
            x1, y1 = positions[right]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_widths.append(max(1.0, abs(weight) * 6.0))

        degrees = dict(self.graph.degree(weight="weight"))
        node_x = [positions[node][0] for node in self.graph.nodes()]
        node_y = [positions[node][1] for node in self.graph.nodes()]
        node_sizes = [max(12.0, abs(degrees.get(node, 0.0)) * 20.0) for node in self.graph.nodes()]

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(color="rgba(99, 102, 241, 0.5)", width=2),
            hoverinfo="none",
            mode="lines",
            name="Edges",
        )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=list(self.graph.nodes()),
            textposition="top center",
            hovertemplate="%{text}<extra></extra>",
            marker=dict(
                size=node_sizes,
                color=node_sizes,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Degree Centrality"),
                line=dict(width=1, color="#111827"),
            ),
            name="Nodes",
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="Correlation Network",
            template="plotly_white",
            showlegend=False,
            height=700,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
        )
        return fig

    def save_graph_features(self, ticker: str) -> None:
        """Save graph features for a ticker to JSON."""

        safe_ticker = _safe_ticker_name(ticker)
        target_dir = ARTIFACTS_DIR / safe_ticker
        target_dir.mkdir(parents=True, exist_ok=True)

        features = self.extract_graph_features(ticker)
        output_path = target_dir / "graph_features.json"
        with output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(features, file_handle, indent=2)

        LOGGER.info("Saved graph features for %s to %s.", ticker, output_path)
