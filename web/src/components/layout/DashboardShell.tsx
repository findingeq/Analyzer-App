/**
 * DashboardShell Component
 * Main layout with resizable sidebar and chart area
 */

import type { ReactNode } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useRunStore } from "@/store/use-run-store";

interface DashboardShellProps {
  sidebar: ReactNode;
  chart: ReactNode;
  footer?: ReactNode;
  headerMetrics?: ReactNode;
}

export function DashboardShell({ sidebar, chart, footer, headerMetrics }: DashboardShellProps) {
  const { sidebarCollapsed, toggleSidebar } = useRunStore();

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
        <div
          className={`relative border-r border-border overflow-y-auto transition-all duration-300 ${
            sidebarCollapsed ? "w-14" : "w-80"
          }`}
        >
          <div className={sidebarCollapsed ? "p-2" : "p-4"}>
            {sidebar}
          </div>

          {/* Collapse Toggle Button */}
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleSidebar}
            className="absolute -right-3 top-4 z-10 h-6 w-6 rounded-full border border-border bg-background shadow-sm hover:bg-muted"
          >
            {sidebarCollapsed ? (
              <ChevronRight className="h-3 w-3" />
            ) : (
              <ChevronLeft className="h-3 w-3" />
            )}
          </Button>
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
