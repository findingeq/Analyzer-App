/**
 * DashboardShell Component
 * Main layout with resizable sidebar and chart area
 */

import type { ReactNode } from "react";
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from "@/components/ui/resizable";
import { ScrollArea } from "@/components/ui/scroll-area";

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

      {/* Main Content */}
      <div className="flex-1 overflow-hidden">
        <ResizablePanelGroup orientation="horizontal" className="h-full">
          {/* Sidebar Panel */}
          <ResizablePanel
            defaultSize={25}
            minSize={15}
            maxSize={40}
            className="border-r border-border"
          >
            <ScrollArea className="h-full">
              <div className="p-4">{sidebar}</div>
            </ScrollArea>
          </ResizablePanel>

          {/* Resize Handle */}
          <ResizableHandle withHandle />

          {/* Chart Panel */}
          <ResizablePanel defaultSize={75} minSize={50}>
            <div className="flex h-full flex-col">
              <div className="flex-1 overflow-hidden">{chart}</div>
              {footer && (
                <div className="border-t border-border">{footer}</div>
              )}
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>
    </div>
  );
}

export default DashboardShell;
