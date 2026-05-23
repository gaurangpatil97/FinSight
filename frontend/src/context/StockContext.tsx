"use client";

import React, { createContext, useContext, useState, useEffect } from "react";

type StockContextType = {
  selectedTicker: string;
  setSelectedTicker: (ticker: string) => void;
  tickers: string[];
};

const StockContext = createContext<StockContextType | undefined>(undefined);

export const StockProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [selectedTicker, setSelectedTickerState] = useState<string>("RELIANCE.NS");

  // Keep stock selection in localStorage so it persists on page refreshes
  useEffect(() => {
    const saved = localStorage.getItem("selectedTicker");
    if (saved && ["RELIANCE.NS", "HDFCBANK.NS", "SBIN.NS"].includes(saved)) {
      setSelectedTickerState(saved);
    }
  }, []);

  const setSelectedTicker = (ticker: string) => {
    setSelectedTickerState(ticker);
    localStorage.setItem("selectedTicker", ticker);
  };

  const tickers = ["RELIANCE.NS", "HDFCBANK.NS", "SBIN.NS"];

  return (
    <StockContext.Provider value={{ selectedTicker, setSelectedTicker, tickers }}>
      {children}
    </StockContext.Provider>
  );
};

export const useStock = () => {
  const context = useContext(StockContext);
  if (!context) {
    throw new Error("useStock must be used within a StockProvider");
  }
  return context;
};
