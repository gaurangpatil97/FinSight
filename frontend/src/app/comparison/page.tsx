"use client";

import React, { useEffect, useState } from "react";
import { useStock } from "@/context/StockContext";
import { fetchComparison, ModelsResponse } from "@/lib/api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from "recharts";

type ModelKey = "xgboost" | "lightgbm" | "random_forest";

export default function ComparisonPage() {
  const { selectedTicker } = useStock();
  const [data, setData] = useState<ModelsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [mounted, setMounted] = useState<boolean>(false);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const compData = await fetchComparison(selectedTicker);
      setData(compData);
    } catch (err: any) {
      console.error("Failed to load comparison data:", err);
      setError(
        err.response?.data?.detail ||
          "Could not retrieve comparison metrics from the backend. Ensure the server is online."
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setMounted(true);
    loadData();
  }, [selectedTicker]);

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
          Fetching and compiling model statistics comparison...
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
        <h3 className="text-lg font-bold text-gray-900 mb-2">Comparison Compilation Failed</h3>
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

  const winnerModel = data.best_model as ModelKey | null;
  const winnerMetrics = winnerModel ? data[winnerModel] : null;

  // Compile Recharts Bar Data
  const chartData = [
    {
      name: "XGBoost",
      mape: data.xgboost?.mape || 0,
      rmse: data.xgboost?.rmse || 0,
      mae: data.xgboost?.mae || 0,
      isWinner: winnerModel === "xgboost",
    },
    {
      name: "LightGBM",
      mape: data.lightgbm?.mape || 0,
      rmse: data.lightgbm?.rmse || 0,
      mae: data.lightgbm?.mae || 0,
      isWinner: winnerModel === "lightgbm",
    },
    {
      name: "Random Forest",
      mape: data.random_forest?.mape || 0,
      rmse: data.random_forest?.rmse || 0,
      mae: data.random_forest?.mae || 0,
      isWinner: winnerModel === "random_forest",
    },
  ];

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Winner Announcement Section at Top */}
      {winnerModel && winnerMetrics ? (
        <div className="bg-gradient-to-r from-blue-900 via-[#1a1a2e] to-[#1e1e3f] text-white rounded-xl shadow-md p-6 md:p-8 flex flex-col md:flex-row items-center justify-between gap-6 border border-[#2b2b52]">
          <div className="space-y-3 max-w-2xl text-center md:text-left">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-bold bg-blue-500/20 text-blue-300 border border-blue-500/30 uppercase tracking-wider">
              Winner Model Announced
            </span>
            <h2 className="text-2xl md:text-3xl font-extrabold tracking-tight">
              {formatModelName(winnerModel)} Wins for {selectedTicker}!
            </h2>
            <p className="text-sm text-gray-300 leading-relaxed font-medium">
              After evaluating predictions against target close values over the last 30 trading sessions,{" "}
              <strong className="text-white">{formatModelName(winnerModel)}</strong> demonstrated the highest
              predictive precision with an MAPE of{" "}
              <strong className="text-emerald-400">{winnerMetrics.mape}%</strong> and an RMSE of{" "}
              <strong className="text-emerald-400">₹{winnerMetrics.rmse}</strong>.
            </p>
          </div>
          <div className="flex-shrink-0 bg-blue-600 shadow-lg shadow-blue-500/20 rounded-2xl w-24 h-24 flex flex-col items-center justify-center border border-blue-400 text-center p-3 animate-pulse">
            <span className="text-[10px] font-extrabold tracking-wider uppercase text-blue-200 block">
              MIN MAPE
            </span>
            <span className="text-xl font-black block mt-0.5">{winnerMetrics.mape}%</span>
          </div>
        </div>
      ) : (
        <div className="bg-white border border-gray-200 rounded-xl p-6 text-center text-gray-400">
          No metrics computed.
        </div>
      )}

      {/* Grid: Charts and Tables */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Recharts Bar Chart (Comparing MAPE) */}
        <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-6 flex flex-col justify-between space-y-4">
          <div>
            <h3 className="font-bold text-[#1a1a2e] text-base">Model MAPE % Comparison</h3>
            <p className="text-xs text-gray-400 mt-0.5">Lower MAPE percentage represents higher forecasting accuracy</p>
          </div>

          <div className="h-72 w-full">
            {mounted ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={chartData}
                  margin={{ top: 20, right: 30, left: 10, bottom: 5 }}
                  barSize={40}
                >
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: "#6b7280", fontSize: 11, fontWeight: 600 }}
                    axisLine={{ stroke: "#e2e8f0" }}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fill: "#6b7280", fontSize: 11 }}
                    axisLine={{ stroke: "#e2e8f0" }}
                    tickLine={false}
                    unit="%"
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#ffffff",
                      border: "1px solid #e5e7eb",
                      borderRadius: "8px",
                      boxShadow: "0 2px 8px rgba(0, 0, 0, 0.05)",
                    }}
                    labelStyle={{ fontWeight: 700, color: "#1a1a2e" }}
                    cursor={{ fill: "#f8f9fa" }}
                  />
                  <Bar dataKey="mape" radius={[4, 4, 0, 0]} name="MAPE (%)">
                    {chartData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.isWinner ? "#2563eb" : "#94a3b8"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="w-full h-full bg-gray-50 flex items-center justify-center rounded-lg border border-gray-100">
                <span className="text-xs text-gray-400">Loading chart graphics...</span>
              </div>
            )}
          </div>

          <div className="flex items-center justify-center gap-4 text-xs font-semibold">
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded bg-blue-600"></span>
              <span className="text-gray-600">Winning Model</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded bg-gray-400"></span>
              <span className="text-gray-600">Alternative Models</span>
            </div>
          </div>
        </div>

        {/* Detailed Metrics Table */}
        <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-6 flex flex-col justify-between">
          <div>
            <h3 className="font-bold text-[#1a1a2e] text-base border-b border-gray-100 pb-4">
              Comprehensive Metrics Matrix
            </h3>
            <div className="overflow-x-auto mt-4">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-gray-100 text-xs font-bold text-gray-400 uppercase tracking-wider">
                    <th className="py-3.5 px-4">Model Algorithm</th>
                    <th className="py-3.5 px-4 text-right">MAPE (%)</th>
                    <th className="py-3.5 px-4 text-right">RMSE (₹)</th>
                    <th className="py-3.5 px-4 text-right">MAE (₹)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-50 text-sm">
                  {chartData.map((model) => {
                    return (
                      <tr
                        key={model.name}
                        className={`transition-colors ${
                          model.isWinner
                            ? "bg-blue-50/20 font-medium"
                            : "hover:bg-gray-50/30 text-gray-600"
                        }`}
                      >
                        <td className="py-4 px-4 text-[#1a1a2e] font-semibold flex items-center gap-2">
                          {model.name}
                          {model.isWinner && (
                            <span className="text-[9px] bg-emerald-500 text-white font-extrabold rounded-full px-2 py-0.5 uppercase tracking-wider shadow-sm">
                              WINNER
                            </span>
                          )}
                        </td>
                        <td className="py-4 px-4 text-right">
                          {model.mape !== 0 ? `${model.mape}%` : "N/A"}
                        </td>
                        <td className="py-4 px-4 text-right">
                          {model.rmse !== 0 ? `₹${model.rmse}` : "N/A"}
                        </td>
                        <td className="py-4 px-4 text-right">
                          {model.mae !== 0 ? `₹${model.mae}` : "N/A"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          <div className="pt-4 border-t border-gray-100 text-xs text-gray-400 leading-relaxed bg-gray-50/50 -mx-6 -mb-6 p-6 rounded-b-xl border-t">
            <span className="font-bold text-gray-600 block mb-1">Metrics Definition Reference:</span>
            <ul className="list-disc pl-4 space-y-1">
              <li><strong>MAPE:</strong> Mean Absolute Percentage Error measures average size of forecasting errors in percentage relative to target value.</li>
              <li><strong>RMSE:</strong> Root Mean Squared Error gives high penalty to large outliers. Highly useful to track volatility bursts.</li>
              <li><strong>MAE:</strong> Mean Absolute Error shows absolute scale of daily price misses in Indian Rupees.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
