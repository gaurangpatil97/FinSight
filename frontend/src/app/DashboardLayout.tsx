"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useStock } from "@/context/StockContext";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const { selectedTicker, setSelectedTicker, tickers } = useStock();
  const pathname = usePathname();

  const navLinks = [
    { name: "Dashboard", href: "/" },
    { name: "EDA Analysis", href: "/eda" },
    { name: "Model Forecasts", href: "/models" },
    { name: "Comparison", href: "/comparison" },
  ];

  // Helper to check if a route is active
  const isActive = (href: string) => {
    if (href === "/") {
      return pathname === "/";
    }
    return pathname.startsWith(href);
  };

  return (
    <div className="flex min-h-screen">
      {/* Sidebar - Dark Navy (#1a1a2e) */}
      <aside className="w-[240px] bg-[#1a1a2e] text-white flex flex-col flex-shrink-0 z-20 shadow-xl border-r border-[#262646]">
        {/* Logo Section */}
        <div className="h-16 flex items-center px-6 border-b border-[#262646] gap-3">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center font-bold text-lg text-white shadow-md shadow-blue-500/20">
            F
          </div>
          <div>
            <h1 className="font-extrabold text-xl tracking-tight bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
              FinSight
            </h1>
            <p className="text-[10px] text-gray-400 font-semibold uppercase tracking-wider -mt-1">
              Analytics Hub
            </p>
          </div>
        </div>

        {/* Navigation Section */}
        <nav className="flex-1 px-4 py-6 space-y-2">
          {navLinks.map((link) => {
            const active = isActive(link.href);
            return (
              <Link
                key={link.name}
                href={link.href}
                className={`flex items-center px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 group ${
                  active
                    ? "bg-blue-600 text-white shadow-md shadow-blue-600/10"
                    : "text-gray-400 hover:bg-[#252542] hover:text-white"
                }`}
              >
                {link.name === "Dashboard" && (
                  <svg
                    className={`w-5 h-5 mr-3 transition-colors ${active ? "text-white" : "text-gray-400 group-hover:text-white"}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v4a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v4a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v4a2 2 0 01-2 2H6a2 2 0 01-2-2v-4zM14 16a2 2 0 012-2h2a2 2 0 012 2v4a2 2 0 01-2 2h-2a2 2 0 01-2-2v-4z" />
                  </svg>
                )}
                {link.name === "EDA Analysis" && (
                  <svg
                    className={`w-5 h-5 mr-3 transition-colors ${active ? "text-white" : "text-gray-400 group-hover:text-white"}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                )}
                {link.name === "Model Forecasts" && (
                  <svg
                    className={`w-5 h-5 mr-3 transition-colors ${active ? "text-white" : "text-gray-400 group-hover:text-white"}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 12l3-3 3 3 4-4M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
                  </svg>
                )}
                {link.name === "Comparison" && (
                  <svg
                    className={`w-5 h-5 mr-3 transition-colors ${active ? "text-white" : "text-gray-400 group-hover:text-white"}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19M12 5v14m0-14L9 8m3-3l3 3m-3 11H9m3 0h3" />
                  </svg>
                )}
                {link.name}
              </Link>
            );
          })}
        </nav>

        {/* Footer info in Sidebar */}
        <div className="p-4 border-t border-[#262646] text-xs text-gray-500">
          <div className="flex items-center justify-between">
            <span>Server Status</span>
            <div className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span className="text-[10px] text-gray-400">Online</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 bg-[#f8f9fa]">
        {/* Top Header */}
        <header className="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-8 z-10 shadow-sm">
          <div>
            <h2 className="text-lg font-bold text-[#1a1a2e]">
              {pathname === "/" && "Overview Dashboard"}
              {pathname === "/eda" && "Exploratory Data Analysis"}
              {pathname === "/models" && "Model Forecasts"}
              {pathname === "/comparison" && "Model Comparison"}
            </h2>
            <p className="text-xs text-gray-400 font-medium">
              Real-time analytics & prediction engine
            </p>
          </div>

          {/* Stock Ticker Selector */}
          <div className="flex items-center gap-3">
            <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Selected Stock
            </span>
            <div className="relative">
              <select
                value={selectedTicker}
                onChange={(e) => setSelectedTicker(e.target.value)}
                className="appearance-none bg-gray-50 border border-gray-200 text-[#1a1a2e] text-sm font-semibold rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-44 pl-4 pr-10 py-2.5 shadow-sm transition-colors duration-200 hover:bg-gray-100 cursor-pointer"
              >
                {tickers.map((ticker) => (
                  <option key={ticker} value={ticker}>
                    {ticker}
                  </option>
                ))}
              </select>
              {/* Custom Selector Arrow */}
              <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-gray-500">
                <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                  <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
                </svg>
              </div>
            </div>
          </div>
        </header>

        {/* Page Content Container */}
        <main className="flex-1 overflow-y-auto p-8 max-w-7xl w-full mx-auto">
          {children}
        </main>
      </div>
    </div>
  );
}
