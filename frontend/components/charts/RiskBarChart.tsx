"use client";

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Cell,
  ResponsiveContainer,
} from "recharts";

interface RiskBarChartProps {
  ecgRisk: number | null;
  pcgRisk: number | null;
  combinedRisk: number | null;
}

interface ChartData {
  name: string;
  value: number;
  label: string;
}

export const RiskBarChart: React.FC<RiskBarChartProps> = ({
  ecgRisk,
  pcgRisk,
  combinedRisk,
}) => {
  // Transform data: Convert to percentages and handle nulls
  const chartData: ChartData[] = [
    {
      name: "ECG",
      value: ecgRisk !== null ? ecgRisk * 100 : 0,
      label: "ECG Risk",
    },
    {
      name: "PCG",
      value: pcgRisk !== null ? pcgRisk * 100 : 0,
      label: "PCG Risk",
    },
    {
      name: "Combined",
      value: combinedRisk !== null ? combinedRisk * 100 : 0,
      label: "Combined Risk",
    },
  ];

  // Dynamic color based on risk threshold
  const getBarColor = (value: number): string => {
    return value > 50 ? "#ef4444" : "#10b981"; // Red if >50%, Green otherwise
  };

  // Custom Tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 shadow-xl">
          <p className="text-white font-semibold mb-1">{data.label}</p>
          <p
            className={`text-lg font-bold ${data.value > 50 ? "text-red-400" : "text-emerald-400"}`}
          >
            {data.value.toFixed(1)}%
          </p>
          <p className="text-xs text-gray-400 mt-1">
            {data.value > 50 ? "High Risk" : "Low Risk"}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full h-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >
          {/* Grid */}
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />

          {/* X Axis */}
          <XAxis
            dataKey="name"
            stroke="#9ca3af"
            tick={{ fill: "#9ca3af", fontSize: 14, fontWeight: 500 }}
            axisLine={{ stroke: "#4b5563" }}
          />

          {/* Y Axis */}
          <YAxis
            stroke="#9ca3af"
            tick={{ fill: "#9ca3af", fontSize: 12 }}
            label={{
              value: "Risk Percentage (%)",
              angle: -90,
              position: "insideLeft",
              fill: "#9ca3af",
              fontSize: 12,
            }}
            domain={[0, 100]}
            axisLine={{ stroke: "#4b5563" }}
          />

          {/* Tooltip */}
          <Tooltip
            content={<CustomTooltip />}
            cursor={{ fill: "#374151", opacity: 0.2 }}
          />

          {/* Reference Line at 50% threshold */}
          <ReferenceLine
            y={50}
            stroke="#ef4444"
            strokeDasharray="5 5"
            strokeWidth={2}
            label={{
              value: "Risk Threshold (50%)",
              position: "right",
              fill: "#ef4444",
              fontSize: 11,
              fontWeight: 600,
            }}
          />

          {/* Bar with dynamic colors */}
          <Bar dataKey="value" radius={[8, 8, 0, 0]} animationDuration={800}>
            {chartData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={getBarColor(entry.value)}
                opacity={entry.value === 0 ? 0.3 : 1}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default RiskBarChart;
