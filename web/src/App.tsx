/**
 * VT Threshold Analyzer - Main App Component
 */

import { DashboardShell } from "@/components/layout/DashboardShell";
import { Sidebar } from "@/components/layout/Sidebar";
import { MainChart } from "@/components/charts/MainChart";
import { IntervalMetrics } from "@/components/charts/IntervalMetrics";
import { HeaderMetrics } from "@/components/layout/HeaderMetrics";

function App() {
  return (
    <DashboardShell
      sidebar={<Sidebar />}
      chart={<MainChart />}
      footer={<IntervalMetrics />}
      headerMetrics={<HeaderMetrics />}
    />
  );
}

export default App;
