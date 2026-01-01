/**
 * DashboardShell Component
 * Main layout with resizable sidebar and chart area
 */

import type { ReactNode } from "react";

interface DashboardShellProps {
  sidebar: ReactNode;
  chart: ReactNode;
  footer?: ReactNode;
  headerMetrics?: ReactNode;
}

export function DashboardShell({ sidebar, chart, footer, headerMetrics }: DashboardShellProps) {
  return (
    <div className="flex h-screen flex-col bg-background">
      {/* Header */}
      <header className="flex h-14 items-center justify-between border-b border-border px-4">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
            VT Check
          </h1>
        </div>
        <div className="flex-1 flex justify-center">
          {headerMetrics}
        </div>
      </header>

      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <div className="w-80 border-r border-border overflow-y-auto p-4">
          {sidebar}
        </div>

        {/* Chart + Footer */}
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
