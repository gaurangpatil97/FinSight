"use client";

import React, { useEffect, useState, startTransition } from "react";
import { useStock } from "@/context/StockContext";
import { fetchModels, ModelsResponse } from "@/lib/api";
import Link from "next/link";

export default function DashboardPage() {
  const { selectedTicker } = useStock();
  const [data, setData] = useState<ModelsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadDashboardData = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchModels(selectedTicker);
      setData(result);
    } catch (err: any) {
      console.error("Error loading dashboard data:", err);
      setError(
        err.response?.data?.detail ||
          "Could not connect to the backend server. Please verify that the FastAPI backend is running on http://localhost:8000"
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDashboardData();
  }, [selectedTicker]);

  // Helper to determine volatility label
  const getVolatilityLabel = (vol: number | null) => {
    if (vol === null) return "N/A";
    if (vol < 15) return "Low";
    if (vol <= 25) return "Moderate";
    return "High";
  };

  // Helper to get volatility color class
  const getVolatilityClass = (vol: number | null) => {
    if (vol === null) return "text-gray-500 bg-gray-50 border-gray-200";
    if (vol < 15) return "text-emerald-700 bg-emerald-50 border-emerald-200";
    if (vol <= 25) return "text-amber-700 bg-amber-50 border-amber-200";
    return "text-rose-700 bg-rose-50 border-rose-200";
  };

  // Helper to get model display names
  const formatModelName = (name: string | null) => {
    if (!name) return "None";
    if (name === "xgboost") return "XGBoost";
    if (name === "lightgbm") return "LightGBM";
    if (name === "random_forest") return "Random Forest";
    return name;
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-4">
        <div className="w-12 h-12 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
        <p className="text-sm font-medium text-gray-500 animate-pulse">
          Fetching market data & evaluation metrics...
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
        <h3 className="text-lg font-bold text-gray-900 mb-2">Backend Connection Failed</h3>
        <p className="text-sm text-gray-500 mb-6 leading-relaxed">{error}</p>
        <button
          onClick={loadDashboardData}
          className="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold rounded-lg shadow transition-colors"
        >
          Retry Connection
        </button>
      </div>
    );
  }

  if (!data) return null;

  const isBullish = data.trend === "Bullish";
  const isBearish = data.trend === "Bearish";

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Welcome / Header Banner Section */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between p-6 bg-white border border-gray-200 rounded-xl shadow-sm gap-4">
        <div>
          <span className="text-xs font-semibold text-blue-600 bg-blue-50 px-2.5 py-1 rounded-md uppercase tracking-wider">
            Market Intelligence
          </span>
          <h2 className="text-2xl font-bold text-[#1a1a2e] mt-2 flex items-center gap-3">
            {selectedTicker} Overview
            <span
              className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border ${
                isBullish
                  ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                  : isBearish
                  ? "bg-rose-50 text-rose-700 border-rose-200"
                  : "bg-slate-50 text-slate-700 border-slate-200"
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  isBullish ? "bg-emerald-500" : isBearish ? "bg-rose-500" : "bg-slate-400"
                }`}
              ></span>
              {data.trend} Trend
            </span>
          </h2>
          <p className="text-sm text-gray-500 mt-1">
            Machine learning forecast performance metrics and analytical statistics.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Link
            href="/eda"
            className="px-4 py-2 text-xs font-bold text-gray-700 hover:text-blue-600 bg-gray-50 hover:bg-gray-100 border border-gray-200 rounded-lg transition-colors shadow-sm"
          >
            Explore Charts
          </Link>
          <Link
            href="/models"
            className="px-4 py-2 text-xs font-bold text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors shadow-sm shadow-blue-500/10"
          >
            Forecast Forecasts
          </Link>
        </div>
      </div>

      {/* Row of 4 Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Current Price */}
        <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm flex flex-col justify-between hover:shadow-md transition-shadow">
          <div>
            <span className="text-xs font-bold text-gray-400 uppercase tracking-wider block">
              Current Close Price
            </span>
            <span className="text-2xl font-extrabold text-[#1a1a2e] block mt-2">
              {data.current_price !== null ? `₹${data.current_price.toLocaleString("en-IN")}` : "N/A"}
            </span>
          </div>
          <div className="mt-4 pt-3 border-t border-gray-100 text-xs text-gray-400">
            Closing price for latest session
          </div>
        </div>

        {/* Daily Return % */}
        <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm flex flex-col justify-between hover:shadow-md transition-shadow">
          <div>
            <span className="text-xs font-bold text-gray-400 uppercase tracking-wider block">
              Daily Return %
            </span>
            <div className="flex items-center mt-2">
              <span
                className={`text-2xl font-extrabold block ${
                  data.daily_return && data.daily_return > 0
                    ? "text-emerald-600"
                    : data.daily_return && data.daily_return < 0
                    ? "text-rose-600"
                    : "text-gray-700"
                }`}
              >
                {data.daily_return !== null
                  ? `${data.daily_return > 0 ? "+" : ""}${data.daily_return}%`
                  : "N/A"}
              </span>
              {data.daily_return !== null && data.daily_return !== 0 && (
                <span className="ml-1">
                  {data.daily_return > 0 ? (
                    <svg className="w-5 h-5 text-emerald-500 inline" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M12 7a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0V8.414l-4.293 4.293a1 1 0 01-1.414 0L8 10.414l-4.293 4.293a1 1 0 01-1.414-1.414l5-5a1 1 0 011.414 0L11 10.586 14.586 7H12z" clipRule="evenodd" />
                    </svg>
                  ) : (
                    <svg className="w-5 h-5 text-rose-500 inline" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M12 13a1 1 0 100 2h5a1 1 0 001-1v-5a1 1 0 10-2 0v2.586l-4.293-4.293a1 1 0 00-1.414 0L8 9.586 3.707 5.293a1 1 0 00-1.414 1.414l5 5a1 1 0 001.414 0L11 9.414 14.586 13H12z" clipRule="evenodd" />
                    </svg>
                  )}
                </span>
              )}
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-gray-100 text-xs text-gray-400">
            Percentage change from previous session
          </div>
        </div>

        {/* Volatility Level */}
        <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm flex flex-col justify-between hover:shadow-md transition-shadow">
          <div>
            <span className="text-xs font-bold text-gray-400 uppercase tracking-wider block">
              Annualized Volatility
            </span>
            <div className="flex items-center gap-2 mt-2">
              <span className="text-2xl font-extrabold text-[#1a1a2e]">
                {data.volatility !== null ? `${data.volatility}%` : "N/A"}
              </span>
              {data.volatility !== null && (
                <span
                  className={`px-2 py-0.5 border rounded text-[10px] font-bold uppercase tracking-wider ${getVolatilityClass(
                    data.volatility
                  )}`}
                >
                  {getVolatilityLabel(data.volatility)}
                </span>
              )}
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-gray-100 text-xs text-gray-400">
            Computed on last 30 daily returns
          </div>
        </div>

        {/* Best Model */}
        <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm flex flex-col justify-between hover:shadow-md transition-shadow">
          <div>
            <span className="text-xs font-bold text-gray-400 uppercase tracking-wider block">
              Best Predictor Model
            </span>
            <span className="inline-flex items-center text-xl font-extrabold text-blue-600 bg-blue-50 border border-blue-100 rounded-lg px-3 py-1 mt-2">
              {formatModelName(data.best_model)}
            </span>
          </div>
          <div className="mt-4 pt-3 border-t border-gray-100 text-xs text-gray-400">
            Selected by lowest RMSE over test data
          </div>
        </div>
      </div>

      {/* Main Grid: Comparison Table and Model Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Model Metrics Table (takes 2 cols on lg screens) */}
        <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-6 lg:col-span-2 space-y-6">
          <div className="flex items-center justify-between border-b border-gray-100 pb-4">
            <div>
              <h3 className="font-bold text-[#1a1a2e] text-base">Model Forecast Accuracy</h3>
              <p className="text-xs text-gray-400">30-day recent target evaluation metrics</p>
            </div>
            <Link
              href="/comparison"
              className="text-xs font-bold text-blue-600 hover:text-blue-700 flex items-center gap-1"
            >
              Full Comparison
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
              </svg>
            </Link>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-gray-100 text-xs text-gray-400 font-bold uppercase tracking-wider">
                  <th className="py-3 px-4">Model Name</th>
                  <th className="py-3 px-4 text-right">MAPE (%)</th>
                  <th className="py-3 px-4 text-right">RMSE (₹)</th>
                  <th className="py-3 px-4 text-center">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50 text-sm">
                {(["xgboost", "lightgbm", "random_forest"] as const).map((modelKey) => {
                  const modelData = data[modelKey];
                  const isBest = data.best_model === modelKey;
                  return (
                    <tr
                      key={modelKey}
                      className={`transition-colors hover:bg-gray-50/50 ${
                        isBest ? "bg-blue-50/20 font-medium" : ""
                      }`}
                    >
                      <td className="py-3.5 px-4 text-[#1a1a2e] font-semibold flex items-center gap-2">
                        {formatModelName(modelKey)}
                        {isBest && (
                          <span className="text-[10px] bg-emerald-500 text-white font-bold rounded-full px-2 py-0.5 shadow-sm shadow-emerald-500/10">
                            WINNER
                          </span>
                        )}
                      </td>
                      <td className="py-3.5 px-4 text-right text-gray-700">
                        {modelData?.mape !== null ? `${modelData.mape}%` : "N/A"}
                      </td>
                      <td className="py-3.5 px-4 text-right text-gray-700">
                        {modelData?.rmse !== null ? `₹${modelData.rmse}` : "N/A"}
                      </td>
                      <td className="py-3.5 px-4 text-center">
                        <Link
                          href={`/models`}
                          onClick={() => {
                            startTransition(() => {
                              // We can direct them to models and highlight this specific one
                            });
                          }}
                          className={`text-xs font-bold px-3 py-1 rounded-md border transition-all ${
                            isBest
                              ? "bg-blue-600 border-blue-600 text-white shadow-sm"
                              : "bg-white hover:bg-gray-50 border-gray-200 text-gray-600"
                          }`}
                        >
                          View Forecast
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Action / Analytical Quick Insights */}
        <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-6 flex flex-col justify-between space-y-6">
          <div className="space-y-4">
            <h3 className="font-bold text-[#1a1a2e] text-base border-b border-gray-100 pb-4">
              FinSight AI Analyzer
            </h3>
            <p className="text-xs text-gray-500 leading-relaxed">
              Based on the latest mathematical evaluation, the best performing forecasting model for{" "}
              <strong className="text-[#1a1a2e]">{selectedTicker}</strong> is{" "}
              <strong className="text-blue-600">{formatModelName(data.best_model)}</strong>.
            </p>
            <div className="bg-blue-50/50 border border-blue-100 rounded-lg p-4 space-y-2">
              <span className="text-[10px] font-extrabold uppercase tracking-wider text-blue-700 block">
                Quick Interpretation
              </span>
              <p className="text-xs text-blue-900 leading-relaxed font-medium">
                {data.best_model === "xgboost" &&
                  "XGBoost excels in identifying localized trends and complex interaction patterns within our technical indictor features."}
                {data.best_model === "lightgbm" &&
                  "LightGBM provides extreme predictive speed and shows high stability when predicting multi-step forward volatility changes."}
                {data.best_model === "random_forest" &&
                  "Random Forest displays robust resistance to overfitting, making it excellent for long-term baseline price levels."}
                {!data.best_model && "Insufficient model artifacts compiled for this stock ticker."}
              </p>
            </div>
          </div>

          <div className="pt-4 border-t border-gray-100 space-y-3">
            <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider">Useful Resources</h4>
            <div className="grid grid-cols-2 gap-2 text-center text-xs">
              <Link
                href="/eda"
                className="py-2.5 border border-gray-200 bg-gray-50 hover:bg-gray-100 font-semibold text-gray-700 rounded-lg transition-colors"
              >
                Volatility EDA
              </Link>
              <Link
                href="/comparison"
                className="py-2.5 border border-gray-200 bg-gray-50 hover:bg-gray-100 font-semibold text-gray-700 rounded-lg transition-colors"
              >
                Performance Chart
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
