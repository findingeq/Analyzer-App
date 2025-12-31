/**
 * DashboardShell Component
 * Main layout with resizable sidebar and chart area
 */

import type { ReactNode } from "react";

interface DashboardShellProps {
  sidebar: ReactNode;
  chart: ReactNode;
  footer?: ReactNode;
}

export function DashboardShell({ sidebar, chart, footer }: DashboardShellProps) {
  return (
    <div className="flex h-screen flex-col bg-background">
      {/* Header */}
      <header className="flex h-14 items-center justify-between border-b border-border px-4">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-semibold text-foreground">
            VT Threshold Analyzer
          </h1>
        </div>
        <div className="flex items-center gap-2">
          {/* Header actions can go here */}
        </div>
      </header>

      {/* Main Content - Simplified layout for debugging */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <div className="w-80 border-r border-border overflow-y-auto p-4">
          {sidebar}
        </div>

        {/* Chart */}
        <div className="flex-1 flex flex-col">
          <div className="flex-1 overflow-hidden">{chart}</div>
          {footer && (
            <div className="border-t border-border">{footer}</div>
          )}
        </div>
      </div>
    </div>
  );
}

export default DashboardShell;
