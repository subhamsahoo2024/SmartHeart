"use client";

import React, { useState } from "react";
import { cn } from "@/lib/utils";

interface TabsProps {
  labels: string[];
  onChange?: (index: number) => void;
  defaultActiveIndex?: number;
  className?: string;
}

export const Tabs: React.FC<TabsProps> = ({
  labels,
  onChange,
  defaultActiveIndex = 0,
  className,
}) => {
  const [activeIndex, setActiveIndex] = useState(defaultActiveIndex);

  const handleTabClick = (index: number) => {
    setActiveIndex(index);
    onChange?.(index);
  };

  return (
    <div
      className={cn(
        "flex space-x-1 rounded-lg bg-gray-900/80 p-1 border border-gray-800",
        className,
      )}
    >
      {labels.map((label, index) => (
        <button
          key={index}
          onClick={() => handleTabClick(index)}
          className={cn(
            "flex-1 rounded-md px-4 py-2.5 text-sm font-medium transition-all duration-200",
            "focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:ring-offset-2 focus:ring-offset-gray-900",
            activeIndex === index
              ? "bg-emerald-500 text-white shadow-lg shadow-emerald-500/25"
              : "text-gray-400 hover:text-gray-200 hover:bg-gray-800/50",
          )}
        >
          {label}
        </button>
      ))}
    </div>
  );
};

export default Tabs;
