"use client";

import React, { useState } from "react";
import {
  Upload,
  Activity,
  Heart,
  AlertCircle,
  CheckCircle,
  Loader2,
} from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Tabs } from "@/components/ui/Tabs";
import { SignalChart } from "@/components/charts/SignalChart";
import { SpectrogramViewer } from "@/components/charts/SpectrogramViewer";
import { RiskBarChart } from "@/components/charts/RiskBarChart";
import { uploadFiles, PredictionResponse } from "@/services/api";

export default function Home() {
  // State Management
  const [ecgFile, setEcgFile] = useState<File | null>(null);
  const [pcgFile, setPcgFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<number>(0);

  // Handle file selection
  const handleEcgChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setEcgFile(e.target.files[0]);
      setError(null);
    }
  };

  const handlePcgChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setPcgFile(e.target.files[0]);
      setError(null);
    }
  };

  // Handle analysis
  const handleAnalyze = async () => {
    if (!ecgFile && !pcgFile) {
      setError("Please upload at least one file (ECG or PCG)");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await uploadFiles(ecgFile, pcgFile);
      setResult(response);
      setActiveTab(0); // Reset to first tab
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "An error occurred during analysis",
      );
      console.error("Analysis error:", err);
    } finally {
      setLoading(false);
    }
  };

  // Risk color helper
  const getRiskColor = (risk: number | null): string => {
    if (risk === null) return "text-gray-400";
    return risk > 0.5 ? "text-red-400" : "text-emerald-400";
  };

  const getRiskLabel = (risk: number | null): string => {
    if (risk === null) return "N/A";
    return risk > 0.5 ? "HIGH RISK" : "LOW RISK";
  };

  // Format ECG data for chart
  const getEcgChartData = () => {
    if (!result?.ecg_plot_data) return [];
    return result.ecg_plot_data.map(([x, y]) => ({ x, y }));
  };

  // Format PCG waveform data for chart
  const getPcgChartData = () => {
    if (!result?.pcg_waveform_data) return [];
    return result.pcg_waveform_data.map((y, x) => ({ x, y }));
  };

  return (
    <div className="min-h-screen p-6 md:p-8 lg:p-12">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-3">
          <div className="flex items-center justify-center gap-3">
            <Heart className="w-10 h-10 text-emerald-500 animate-pulse" />
            <h1 className="text-4xl md:text-5xl font-bold bg-linear-to-r from-emerald-400 to-blue-500 bg-clip-text text-transparent">
              SmartHeart
            </h1>
          </div>
          <p className="text-gray-400 text-lg">
            Bimodal Deep Learning for ECG and PCG–Based Cardiac Monitoring
          </p>
        </div>

        {/* Upload Section */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* ECG Upload */}
          <Card
            title="ECG Signal Upload"
            className="hover:border-blue-500/50 transition-colors"
          >
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-sm text-gray-400">
                <Activity className="w-4 h-4" />
                <span>Electrocardiogram (.csv format)</span>
              </div>

              <label className="block">
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleEcgChange}
                  className="hidden"
                  disabled={loading}
                />
                <div className="border-2 border-dashed border-gray-700 rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 hover:bg-gray-800/30 transition-all">
                  <Upload className="w-8 h-8 mx-auto mb-3 text-gray-500" />
                  <p className="text-sm text-gray-400">
                    {ecgFile ? (
                      <span className="text-blue-400 font-medium">
                        {ecgFile.name}
                      </span>
                    ) : (
                      "Click to upload ECG file"
                    )}
                  </p>
                  <p className="text-xs text-gray-600 mt-1">
                    187 sample points required
                  </p>
                </div>
              </label>
            </div>
          </Card>

          {/* PCG Upload */}
          <Card
            title="PCG Audio Upload"
            className="hover:border-emerald-500/50 transition-colors"
          >
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-sm text-gray-400">
                <Heart className="w-4 h-4" />
                <span>Phonocardiogram (.wav format)</span>
              </div>

              <label className="block">
                <input
                  type="file"
                  accept=".wav"
                  onChange={handlePcgChange}
                  className="hidden"
                  disabled={loading}
                />
                <div className="border-2 border-dashed border-gray-700 rounded-lg p-8 text-center cursor-pointer hover:border-emerald-500 hover:bg-gray-800/30 transition-all">
                  <Upload className="w-8 h-8 mx-auto mb-3 text-gray-500" />
                  <p className="text-sm text-gray-400">
                    {pcgFile ? (
                      <span className="text-emerald-400 font-medium">
                        {pcgFile.name}
                      </span>
                    ) : (
                      "Click to upload PCG file"
                    )}
                  </p>
                  <p className="text-xs text-gray-600 mt-1">
                    5-second heart sound recording
                  </p>
                </div>
              </label>
            </div>
          </Card>
        </div>

        {/* Analyze Button */}
        <div className="flex flex-col items-center gap-4">
          <button
            onClick={handleAnalyze}
            disabled={loading || (!ecgFile && !pcgFile)}
            className="px-8 py-4 bg-linear-to-r from-emerald-500 to-blue-500 rounded-lg font-semibold text-white text-lg
                     hover:from-emerald-600 hover:to-blue-600 disabled:from-gray-700 disabled:to-gray-700 
                     disabled:cursor-not-allowed transition-all duration-300 shadow-lg hover:shadow-emerald-500/25
                     flex items-center gap-3"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Activity className="w-5 h-5" />
                Analyze Heart Data
              </>
            )}
          </button>

          {error && (
            <div className="flex items-center gap-2 text-red-400 bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-2">
              <AlertCircle className="w-4 h-4" />
              <span className="text-sm">{error}</span>
            </div>
          )}
        </div>

        {/* Results Section */}
        {result && (
          <div className="space-y-8 animate-in fade-in duration-500">
            {/* Risk Cards */}
            <div className="grid md:grid-cols-3 gap-6">
              {/* ECG Risk */}
              <Card className="text-center">
                <div className="space-y-3">
                  <div className="flex items-center justify-center gap-2 text-blue-400">
                    <Activity className="w-5 h-5" />
                    <h3 className="font-semibold">ECG Risk</h3>
                  </div>
                  <div
                    className={`text-4xl font-bold ${getRiskColor(result.ecg_risk)}`}
                  >
                    {result.ecg_risk !== null
                      ? (result.ecg_risk * 100).toFixed(1) + "%"
                      : "N/A"}
                  </div>
                  <div
                    className={`text-sm font-medium ${getRiskColor(result.ecg_risk)}`}
                  >
                    {getRiskLabel(result.ecg_risk)}
                  </div>
                </div>
              </Card>

              {/* PCG Risk */}
              <Card className="text-center">
                <div className="space-y-3">
                  <div className="flex items-center justify-center gap-2 text-emerald-400">
                    <Heart className="w-5 h-5" />
                    <h3 className="font-semibold">PCG Risk</h3>
                  </div>
                  <div
                    className={`text-4xl font-bold ${getRiskColor(result.pcg_risk)}`}
                  >
                    {result.pcg_risk !== null
                      ? (result.pcg_risk * 100).toFixed(1) + "%"
                      : "N/A"}
                  </div>
                  <div
                    className={`text-sm font-medium ${getRiskColor(result.pcg_risk)}`}
                  >
                    {getRiskLabel(result.pcg_risk)}
                  </div>
                </div>
              </Card>

              {/* Combined Risk */}
              <Card className="text-center border-emerald-500/30 bg-emerald-500/5">
                <div className="space-y-3">
                  <div className="flex items-center justify-center gap-2 text-emerald-400">
                    {result.combined_risk !== null &&
                    result.combined_risk > 0.5 ? (
                      <AlertCircle className="w-5 h-5" />
                    ) : (
                      <CheckCircle className="w-5 h-5" />
                    )}
                    <h3 className="font-semibold">Combined Risk</h3>
                  </div>
                  <div
                    className={`text-5xl font-bold ${getRiskColor(result.combined_risk)}`}
                  >
                    {result.combined_risk !== null
                      ? (result.combined_risk * 100).toFixed(1) + "%"
                      : "N/A"}
                  </div>
                  <div
                    className={`text-sm font-medium ${getRiskColor(result.combined_risk)}`}
                  >
                    {getRiskLabel(result.combined_risk)}
                  </div>
                </div>
              </Card>
            </div>

            {/* Risk Comparison Bar Chart */}
            <Card title="Risk Assessment Overview">
              <div className="h-80">
                <RiskBarChart
                  ecgRisk={result.ecg_risk}
                  pcgRisk={result.pcg_risk}
                  combinedRisk={result.combined_risk}
                />
              </div>
            </Card>

            {/* Visuals Container */}
            <Card title="Signal Visualization">
              <div className="space-y-6">
                {/* Tabs */}
                <Tabs
                  labels={["ECG Signal", "PCG Waveform", "Spectrogram"]}
                  onChange={setActiveTab}
                  defaultActiveIndex={0}
                />

                {/* Chart Container */}
                <div className="h-100 bg-gray-950/50 rounded-lg p-4 border border-gray-800">
                  {activeTab === 0 && result.ecg_plot_data && (
                    <SignalChart
                      data={getEcgChartData()}
                      heatmapData={result.ecg_heatmap || undefined}
                      riskScore={
                        result.ecg_risk !== null
                          ? result.ecg_risk * 100
                          : undefined
                      }
                      color="#3b82f6"
                      title="ECG Waveform with Risk Heatmap"
                      xLabel="Sample"
                      yLabel="Normalized Amplitude"
                    />
                  )}

                  {activeTab === 1 && result.pcg_waveform_data && (
                    <SignalChart
                      data={getPcgChartData()}
                      heatmapData={result.pcg_heatmap || undefined}
                      riskScore={
                        result.pcg_risk !== null
                          ? result.pcg_risk * 100
                          : undefined
                      }
                      color="#10b981"
                      title="PCG Audio Waveform with Risk Heatmap"
                      xLabel="Sample"
                      yLabel="Amplitude"
                    />
                  )}

                  {activeTab === 2 && (
                    <SpectrogramViewer
                      base64Image={result.pcg_spectrogram}
                      title="PCG Frequency Spectrogram"
                    />
                  )}

                  {/* Empty state for missing data */}
                  {((activeTab === 0 && !result.ecg_plot_data) ||
                    (activeTab === 1 && !result.pcg_waveform_data) ||
                    (activeTab === 2 && !result.pcg_spectrogram)) && (
                    <div className="flex items-center justify-center h-full">
                      <div className="text-center text-gray-500">
                        <AlertCircle className="w-12 h-12 mx-auto mb-3 opacity-50" />
                        <p>No data available for this visualization</p>
                        <p className="text-sm text-gray-600 mt-1">
                          Upload the corresponding file to view
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </Card>
          </div>
        )}

        {/* Footer */}
        <div className="text-center text-sm text-gray-600 pt-8 border-t border-gray-800">
          <p>AI-powered multimodal cardiovascular risk assessment system</p>
          <p className="mt-3">For research and educational purposes only</p>
        </div>
      </div>
    </div>
  );
}
