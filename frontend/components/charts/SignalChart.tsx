"use client";

import React, { useState, useEffect } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";
import { Eye, EyeOff } from "lucide-react";

interface SignalChartProps {
  data: Array<{ x: number; y: number }>;
  heatmapData?: number[]; // Optional: array of values from 0 to 1
  riskScore?: number; // Risk score from 0-100 (for smart visibility logic)
  color?: string;
  title?: string;
  xLabel?: string;
  yLabel?: string;
}

export const SignalChart: React.FC<SignalChartProps> = ({
  data,
  heatmapData,
  riskScore,
  color = "#10b981", // Emerald-500 default
  title,
  xLabel = "Sample",
  yLabel = "Amplitude",
}) => {
  // Smart heatmap visibility state
  const [showHeatmap, setShowHeatmap] = useState(false);

  // Smart logic: Auto-show heatmap for high-risk patients (>50%)
  useEffect(() => {
    if (riskScore !== undefined && riskScore !== null) {
      setShowHeatmap(riskScore > 50);
    } else {
      setShowHeatmap(false);
    }
  }, [riskScore]);

  // Merge signal data with heatmap data
  const chartData = data.map((point, index) => ({
    ...point,
    risk: heatmapData ? heatmapData[index] || 0 : 0,
  }));

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const linePayload = payload.find((p: any) => p.dataKey === "y");
      const areaPayload = payload.find((p: any) => p.dataKey === "risk");

      return (
        <div
          style={{
            backgroundColor: "#1f2937",
            border: "1px solid #374151",
            borderRadius: "6px",
            color: "#fff",
            fontSize: "12px",
            padding: "8px",
          }}
        >
          <p style={{ color: "#9ca3af" }}>{`${xLabel}: ${label}`}</p>
          {linePayload && (
            <p
              style={{ color: linePayload.stroke }}
            >{`${yLabel}: ${linePayload.value.toFixed(4)}`}</p>
          )}
          {areaPayload && heatmapData && showHeatmap && (
            <p
              style={{ color: "#ff0000" }}
            >{`Risk: ${areaPayload.value.toFixed(4)}`}</p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full h-full">
      {/* Header with title and toggle button */}
      <div className="flex items-center justify-between mb-3 px-2">
        {title && (
          <h4 className="text-sm font-medium text-gray-300">{title}</h4>
        )}

        {/* Heatmap Toggle Button - only show if heatmap data exists */}
        {heatmapData && (
          <button
            onClick={() => setShowHeatmap(!showHeatmap)}
            className={`
              flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium
              transition-all duration-200 border
              ${
                showHeatmap
                  ? "bg-red-500/20 border-red-500/50 text-red-400 hover:bg-red-500/30"
                  : "bg-gray-800/50 border-gray-700 text-gray-400 hover:bg-gray-800 hover:border-gray-600"
              }
            `}
            title={
              showHeatmap
                ? "Hide explainability heatmap"
                : "Show explainability heatmap"
            }
          >
            {showHeatmap ? (
              <>
                <Eye className="w-3.5 h-3.5" />
                <span>Hide Heatmap</span>
              </>
            ) : (
              <>
                <EyeOff className="w-3.5 h-3.5" />
                <span>Show Heatmap</span>
              </>
            )}
          </button>
        )}
      </div>

      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={chartData}
          margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
        >
          {/* Grid for medical precision */}
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />

          {/* X Axis */}
          <XAxis
            dataKey="x"
            stroke="#9ca3af"
            tick={{ fill: "#9ca3af", fontSize: 12 }}
            label={{
              value: xLabel,
              position: "insideBottom",
              offset: -5,
              fill: "#9ca3af",
              fontSize: 12,
            }}
          />

          {/* Y Axis for Signal Line */}
          <YAxis
            yAxisId="left"
            dataKey="y"
            stroke="#9ca3af"
            tick={{ fill: "#9ca3af", fontSize: 12 }}
            label={{
              value: yLabel,
              angle: -90,
              position: "insideLeft",
              fill: "#9ca3af",
              fontSize: 12,
            }}
          />

          {/* Hidden Y-Axis for Heatmap Area */}
          <YAxis yAxisId="right" hide={true} domain={[0, 1]} />

          {/* Tooltip */}
          <Tooltip content={<CustomTooltip />} />

          {/* Heatmap Area - Conditional rendering based on toggle state */}
          {showHeatmap && heatmapData && (
            <Area
              type="monotone"
              dataKey="risk"
              stroke="none"
              fill="#ff0000"
              fillOpacity={0.3}
              yAxisId="right"
              isAnimationActive={false}
            />
          )}

          {/* Signal Line */}
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="y"
            stroke={color}
            strokeWidth={1.5}
            dot={false} // Remove dots for performance
            isAnimationActive={false} // Disable animation for large datasets
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

export default SignalChart;
