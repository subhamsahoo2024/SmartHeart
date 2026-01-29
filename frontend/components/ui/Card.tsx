import React from "react";
import { cn } from "@/lib/utils";

interface CardProps {
  children: React.ReactNode;
  title?: string;
  className?: string;
}

export const Card: React.FC<CardProps> = ({ children, title, className }) => {
  return (
    <div
      className={cn(
        "rounded-lg border border-gray-800 bg-gray-900/50 backdrop-blur-md",
        "shadow-lg shadow-black/20",
        "transition-all duration-300 hover:border-gray-700",
        className,
      )}
    >
      {title && (
        <div className="border-b border-gray-800 px-6 py-4">
          <h3 className="text-lg font-semibold text-white">{title}</h3>
        </div>
      )}
      <div className="p-6">{children}</div>
    </div>
  );
};

export default Card;
