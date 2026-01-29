"use client";

import React from "react";
import { cn } from "@/lib/utils";

interface SpectrogramViewerProps {
  base64Image: string | null;
  title?: string;
  className?: string;
}

export const SpectrogramViewer: React.FC<SpectrogramViewerProps> = ({
  base64Image,
  title = "PCG Spectrogram",
  className,
}) => {
  if (!base64Image) {
    return (
      <div
        className={cn(
          "flex items-center justify-center h-full min-h-75",
          "rounded-lg border-2 border-dashed border-gray-700",
          "bg-gray-900/30",
          className,
        )}
      >
        <div className="text-center">
          <div className="text-gray-500 text-sm">
            No spectrogram data available
          </div>
          <div className="text-gray-600 text-xs mt-1">
            Upload a PCG file to view
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={cn("w-full h-full", className)}>
      {title && (
        <h4 className="text-sm font-medium text-gray-300 mb-3 px-2">{title}</h4>
      )}

      <div
        className={cn(
          "relative rounded-lg overflow-hidden",
          "border-2 border-emerald-500/30",
          "bg-gray-950",
          "shadow-lg shadow-emerald-500/10",
          "transition-all duration-300 hover:border-emerald-500/50",
        )}
      >
        {/* Technical instrument overlay effect */}
        <div className="absolute inset-0 pointer-events-none">
          {/* Corner brackets for technical look */}
          <div className="absolute top-2 left-2 w-4 h-4 border-l-2 border-t-2 border-emerald-500/50" />
          <div className="absolute top-2 right-2 w-4 h-4 border-r-2 border-t-2 border-emerald-500/50" />
          <div className="absolute bottom-2 left-2 w-4 h-4 border-l-2 border-b-2 border-emerald-500/50" />
          <div className="absolute bottom-2 right-2 w-4 h-4 border-r-2 border-b-2 border-emerald-500/50" />

          {/* Subtle scan line effect */}
          <div className="absolute inset-0 bg-linear-to-b from-transparent via-emerald-500/5 to-transparent animate-pulse" />
        </div>

        {/* Spectrogram Image */}
        <img
          src={base64Image}
          alt="PCG Spectrogram"
          className={cn(
            "w-full h-auto object-contain",
            "min-h-62.5 max-h-[400px]",
            "bg-gray-950",
          )}
        />

        {/* Technical info overlay */}
        <div className="absolute bottom-0 left-0 right-0 bg-linear-to-t from-gray-950/90 to-transparent p-2">
          <div className="flex items-center justify-between text-xs text-emerald-400/80">
            <span className="font-mono">FREQ</span>
            <span className="font-mono">TIME →</span>
          </div>
        </div>
      </div>

      {/* Technical specs display */}
      <div className="mt-2 flex justify-between text-xs text-gray-500 font-mono px-2">
        <span>SR: 1000 Hz</span>
        <span>RANGE: 20-400 Hz</span>
        <span>WIN: 5.0s</span>
      </div>
    </div>
  );
};

export default SpectrogramViewer;
