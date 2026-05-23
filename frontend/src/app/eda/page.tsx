"use client";

import React, { useEffect, useState } from "react";
import { useStock } from "@/context/StockContext";
import { fetchEDA, EDAResponse } from "@/lib/api";

type TabId =
  | "trend_analysis"
  | "volatility_analysis"
  | "rolling_statistics"
  | "seasonality_analysis"
  | "distribution_analysis"
  | "correlation_analysis";

export default function EDAPage() {
  const { selectedTicker } = useStock();
  const [edaPaths, setEdaPaths] = useState<EDAResponse | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("trend_analysis");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const tabs: { id: TabId; label: string; desc: string }[] = [
    {
      id: "trend_analysis",
      label: "Trend",
      desc: "Historical stock close price with key exponential moving averages (50, 100, 200 EMA) to capture baseline trend structure.",
    },
    {
      id: "volatility_analysis",
      label: "Volatility",
      desc: "Daily log returns and rolling volatility spreads, displaying period-by-period variance clusters and risk profiles.",
    },
    {
      id: "rolling_statistics",
      label: "Rolling Stats",
      desc: "Rolling mean standard deviations across 30, 60, and 90-day intervals, tracking dynamic structural breaks.",
    },
    {
      id: "seasonality_analysis",
      label: "Seasonality",
      desc: "Decomposed seasonality components showcasing monthly, weekly, and daily structural return cycles.",
    },
    {
      id: "distribution_analysis",
      label: "Distribution",
      desc: "Distribution histogram of daily returns overlaid with normal distribution curve, highlighting fat-tails (kurtosis).",
    },
    {
      id: "correlation_analysis",
      label: "Correlation",
      desc: "Correlation heatmaps comparing technical indicator features, volume spikes, and relative returns across tickers.",
    },
  ];

  const loadEDA = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchEDA(selectedTicker);
      setEdaPaths(data);
    } catch (err: any) {
      console.error("Failed to load EDA paths:", err);
      setError(
        err.response?.data?.detail ||
          "Failed to load or generate EDA metrics. Please verify the backend is online and running."
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadEDA();
  }, [selectedTicker]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-4">
        <div className="w-12 h-12 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
        <p className="text-sm font-medium text-gray-500 animate-pulse">
          Generating and rendering Plotly analysis files... (This can take 5-10s if computing for the first time)
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
        <h3 className="text-lg font-bold text-gray-900 mb-2">EDA Loading Failed</h3>
        <p className="text-sm text-gray-500 mb-6 leading-relaxed">{error}</p>
        <button
          onClick={loadEDA}
          className="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold rounded-lg shadow transition-colors"
        >
          Try Again
        </button>
      </div>
    );
  }

  if (!edaPaths) return null;

  const currentFilePath = edaPaths[activeTab];
  const iframeSrc = `/api/eda-html?path=${encodeURIComponent(currentFilePath)}`;
  const currentTabDesc = tabs.find((t) => t.id === activeTab)?.desc;

  return (
    <div className="space-y-6">
      {/* Header Info */}
      <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
        <h3 className="text-xl font-bold text-[#1a1a2e]">Exploratory Data Analysis (EDA)</h3>
        <p className="text-sm text-gray-400 mt-1">
          Explore interactive time-series plots and quantitative profiles generated dynamically from backend stock records.
        </p>
      </div>

      {/* Tabs Menu Section */}
      <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden flex flex-col">
        {/* Navigation Tabs */}
        <div className="border-b border-gray-200 overflow-x-auto">
          <nav className="flex px-4 -mb-px" aria-label="Tabs">
            {tabs.map((tab) => {
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`py-4 px-6 text-sm font-bold border-b-2 whitespace-nowrap transition-colors duration-200 cursor-pointer ${
                    isActive
                      ? "border-blue-600 text-blue-600"
                      : "border-transparent text-gray-400 hover:text-gray-700 hover:border-gray-300"
                  }`}
                >
                  {tab.label}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Tab Metadata description */}
        <div className="p-4 bg-gray-50 border-b border-gray-200">
          <p className="text-xs font-semibold text-gray-600 flex items-center gap-2">
            <svg className="w-4 h-4 text-blue-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            {currentTabDesc}
          </p>
        </div>

        {/* Plotly Chart Iframe Canvas */}
        <div className="p-6 bg-white flex flex-col items-stretch">
          <div className="relative w-full h-[650px] border border-gray-100 rounded-lg overflow-hidden bg-gray-50 shadow-inner">
            <iframe
              src={iframeSrc}
              title={`${activeTab} plot`}
              className="absolute inset-0 w-full h-full border-none"
              sandbox="allow-scripts allow-same-origin"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
