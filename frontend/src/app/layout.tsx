import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { StockProvider } from "@/context/StockContext";
import DashboardLayout from "./DashboardLayout";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "FinSight - Stock Performance Analytics & Forecasting",
  description: "Advanced analytics, automated EDA, and state-of-the-art machine learning forecasting models for Indian equities.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col font-sans">
        <StockProvider>
          <DashboardLayout>{children}</DashboardLayout>
        </StockProvider>
      </body>
    </html>
  );
}
