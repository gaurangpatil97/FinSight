import axios from "axios";

const API_BASE = "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    "Content-Type": "application/json",
  },
});

export interface ModelMetrics {
  mape: number | null;
  rmse: number | null;
  mae: number | null;
}

export interface ModelsResponse {
  xgboost: ModelMetrics;
  lightgbm: ModelMetrics;
  random_forest: ModelMetrics;
  best_model: string | null;
  current_price: number | null;
  daily_return: number | null;
  volatility: number | null;
  trend: string;
}

export interface EDAResponse {
  trend_analysis: string;
  volatility_analysis: string;
  rolling_statistics: string;
  seasonality_analysis: string;
  distribution_analysis: string;
  correlation_analysis: string;
}

export interface ForecastResponse {
  forecast_html: string | null;
}

export const fetchEDA = async (ticker: string): Promise<EDAResponse> => {
  const response = await api.get<EDAResponse>(`/eda/${ticker}`);
  return response.data;
};

export const fetchModels = async (ticker: string): Promise<ModelsResponse> => {
  const response = await api.get<ModelsResponse>(`/models/${ticker}`);
  return response.data;
};

export const fetchForecast = async (ticker: string, modelName: string): Promise<ForecastResponse> => {
  const response = await api.get<ForecastResponse>(`/forecast/${ticker}/${modelName}`);
  return response.data;
};

export const fetchComparison = async (ticker: string): Promise<ModelsResponse> => {
  const response = await api.get<ModelsResponse>(`/comparison/${ticker}`);
  return response.data;
};
