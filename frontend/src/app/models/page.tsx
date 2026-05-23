"use client";

import React, { useEffect, useState } from "react";
import { useStock } from "@/context/StockContext";
import { fetchModels, fetchForecast, ModelsResponse } from "@/lib/api";

type ModelKey = "xgboost" | "lightgbm" | "random_forest";

export default function ModelsPage() {
  const { selectedTicker } = useStock();
  const [data, setData] = useState<ModelsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Selected forecast model to show in the iframe below
  const [selectedModel, setSelectedModel] = useState<ModelKey | null>(null);
  const [forecastPath, setForecastPath] = useState<string | null>(null);
  const [forecastLoading, setForecastLoading] = useState<boolean>(false);
  const [forecastError, setForecastError] = useState<string | null>(null);

  const formatModelName = (name: string | null) => {
    if (!name) return "None";
    if (name === "xgboost") return "XGBoost";
    if (name === "lightgbm") return "LightGBM";
    if (name === "random_forest") return "Random Forest";
    return name;
  };

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const metricsData = await fetchModels(selectedTicker);
      setData(metricsData);
      
      // Auto-select best model on initial load
      if (metricsData.best_model) {
        setSelectedModel(metricsData.best_model as ModelKey);
      } else {
        setSelectedModel("xgboost");
      }
    } catch (err: any) {
      console.error("Failed to load models data:", err);
      setError(
        err.response?.data?.detail ||
          "Could not retrieve model metrics from backend. Ensure the server is online."
      );
    } finally {
      setLoading(false);
    }
  };

  const loadForecast = async (modelKey: ModelKey) => {
    setForecastLoading(true);
    setForecastError(null);
    setForecastPath(null);
    try {
      const forecastData = await fetchForecast(selectedTicker, modelKey);
      if (forecastData.forecast_html) {
        setForecastPath(forecastData.forecast_html);
      } else {
        setForecastError(
          `Forecast HTML report not found for ${formatModelName(modelKey)} on ${selectedTicker}. The model may still be training or the report file is missing on the local disk.`
        );
      }
    } catch (err: any) {
      console.error(`Failed to load forecast for ${modelKey}:`, err);
      setForecastError(
        err.response?.data?.detail ||
          `Forecast HTML plot not found. This model may still be training or the forecast file is missing on local disk.`
      );
    } finally {
      setForecastLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [selectedTicker]);

  useEffect(() => {
    if (selectedModel) {
      loadForecast(selectedModel);
    }
  }, [selectedModel, selectedTicker]);

  const models: { key: ModelKey; label: string; desc: string }[] = [
    {
      key: "xgboost",
      label: "XGBoost",
      desc: "Extreme Gradient Boosting regressor trained on engineered technical indicator indicators.",
    },
    {
      key: "lightgbm",
      label: "LightGBM",
      desc: "Highly efficient leaf-wise gradient boosting model specialized in capture market trends.",
    },
    {
      key: "random_forest",
      label: "Random Forest",
      desc: "Robust bootstrap aggregated ensemble of decision trees targeting mean variance returns.",
    },
  ];

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-4">
        <div className="w-12 h-12 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
        <p className="text-sm font-medium text-gray-500 animate-pulse">
          Fetching forecasting models performance data...
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-xl mx-auto my-12 p-8 bg-white border border-rose-100 rounded-2xl shadow-md text-center">
        <div className="w-16 h-16 bg-rose-50 border border-rose-100 rounded-full flex items-center justify-center mx-auto mb-4 text-rose-500">
          <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        <h3 className="text-lg font-bold text-gray-900 mb-2">Metrics Retrieval Failed</h3>
        <p className="text-sm text-gray-500 mb-6 leading-relaxed">{error}</p>
        <button
          onClick={loadData}
          className="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold rounded-lg shadow transition-colors"
        >
          Try Again
        </button>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Page Header */}
      <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h3 className="text-xl font-bold text-[#1a1a2e]">Model Forecasts & Metrics</h3>
          <p className="text-sm text-gray-400 mt-1">
            Compare model prediction scores and visualize historical testing forecasts projected against actual stock prices.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-emerald-700 bg-emerald-50 px-3 py-1 rounded-md border border-emerald-100">
            Selected Stock: {selectedTicker}
          </span>
        </div>
      </div>

      {/* Grid of 3 Model Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {models.map((model) => {
          const metrics = data[model.key];
          const isBest = data.best_model === model.key;
          const isSelected = selectedModel === model.key;

          return (
            <div
              key={model.key}
              className={`bg-white rounded-xl shadow-sm p-6 flex flex-col justify-between transition-all duration-300 relative border-2 ${
                isBest
                  ? "border-blue-600 ring-4 ring-blue-500/5 shadow-md shadow-blue-500/5"
                  : isSelected
                  ? "border-gray-300"
                  : "border-gray-200 hover:border-gray-300"
              }`}
            >
              {/* Recommended Badge for Best Model */}
              {isBest && (
                <span className="absolute -top-3.5 left-6 text-[10px] font-extrabold bg-blue-600 text-white px-2.5 py-1 rounded-md shadow-sm border border-blue-500 tracking-wider uppercase">
                  Best Model (Lowest RMSE)
                </span>
              )}

              <div className="space-y-4">
                <div>
                  <h4 className="text-lg font-bold text-[#1a1a2e] flex items-center justify-between">
                    {model.label}
                    {isSelected && (
                      <span className="w-2.5 h-2.5 rounded-full bg-blue-600 animate-pulse"></span>
                    )}
                  </h4>
                  <p className="text-xs text-gray-400 mt-1 leading-relaxed">{model.desc}</p>
                </div>

                {/* Score list grid */}
                <div className="grid grid-cols-3 gap-2 bg-gray-50 rounded-lg p-3 border border-gray-100">
                  <div className="text-center">
                    <span className="text-[10px] font-semibold text-gray-400 block uppercase">MAPE</span>
                    <span className="text-xs font-bold text-gray-800 mt-1 block">
                      {metrics?.mape !== null ? `${metrics.mape}%` : "N/A"}
                    </span>
                  </div>
                  <div className="text-center border-x border-gray-200">
                    <span className="text-[10px] font-semibold text-gray-400 block uppercase">RMSE</span>
                    <span className="text-xs font-bold text-gray-800 mt-1 block">
                      {metrics?.rmse !== null ? `₹${metrics.rmse}` : "N/A"}
                    </span>
                  </div>
                  <div className="text-center">
                    <span className="text-[10px] font-semibold text-gray-400 block uppercase">MAE</span>
                    <span className="text-xs font-bold text-gray-800 mt-1 block">
                      {metrics?.mae !== null ? `₹${metrics.mae}` : "N/A"}
                    </span>
                  </div>
                </div>
              </div>

              {/* View Forecast Trigger button */}
              <button
                onClick={() => setSelectedModel(model.key)}
                className={`w-full py-2.5 mt-6 text-xs font-bold rounded-lg border transition-all duration-200 cursor-pointer ${
                  isSelected
                    ? "bg-blue-600 border-blue-600 text-white shadow-md shadow-blue-500/10"
                    : isBest
                    ? "bg-white hover:bg-blue-50 border-blue-600 text-blue-600"
                    : "bg-white hover:bg-gray-50 border-gray-200 text-gray-600"
                }`}
              >
                {isSelected ? "Active Forecast" : "View Forecast"}
              </button>
            </div>
          );
        })}
      </div>

      {/* Dynamic Forecast Interactive Plot Section */}
      <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden flex flex-col">
        <div className="border-b border-gray-200 px-6 py-4 bg-gray-50 flex items-center justify-between">
          <div>
            <h3 className="font-bold text-[#1a1a2e] text-base">
              Interactive Time-Series Projection Plot
            </h3>
            <p className="text-xs text-gray-400 mt-0.5">
              Visualizing predictions versus real close prices for{" "}
              <strong className="text-gray-700">
                {formatModelName(selectedModel)} ({selectedTicker})
              </strong>
            </p>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>
            <span className="text-[10px] font-bold text-blue-700 uppercase tracking-wider">Plotly Engine</span>
          </div>
        </div>

        <div className="p-6 bg-white">
          {forecastLoading ? (
            <div className="flex flex-col items-center justify-center h-[550px] space-y-3">
              <div className="w-10 h-10 border-4 border-blue-100 border-t-blue-600 rounded-full animate-spin"></div>
              <p className="text-xs font-medium text-gray-500">Loading model target metrics and charts...</p>
            </div>
          ) : forecastError ? (
            <div className="flex flex-col items-center justify-center h-[550px] text-center max-w-md mx-auto space-y-4">
              <div className="w-12 h-12 bg-amber-50 border border-amber-100 rounded-full flex items-center justify-center text-amber-500">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <h4 className="text-sm font-bold text-gray-800">Forecast Plot Unavailable</h4>
              <p className="text-xs text-gray-400 leading-relaxed">{forecastError}</p>
            </div>
          ) : forecastPath ? (
            <div className="relative w-full h-[550px] border border-gray-100 rounded-lg overflow-hidden bg-gray-50 shadow-inner">
              <iframe
                src={`/api/eda-html?path=${encodeURIComponent(forecastPath)}`}
                title={`${selectedModel} forecast plot`}
                className="absolute inset-0 w-full h-full border-none"
                sandbox="allow-scripts allow-same-origin"
              />
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-[550px] text-center space-y-2 text-gray-400">
              <svg className="w-12 h-12 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              <h4 className="text-sm font-bold text-gray-800">No Forecast Selected</h4>
              <p className="text-xs">Click "View Forecast" on any card above to display details.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
