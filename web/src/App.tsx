/**
 * VT Threshold Analyzer - Main App Component
 */

import { DashboardShell } from "@/components/layout/DashboardShell";
import { Sidebar } from "@/components/layout/Sidebar";
import { MainChart } from "@/components/charts/MainChart";

function App() {
  return (
    <DashboardShell
      sidebar={<Sidebar />}
      chart={<MainChart />}
    />
  );
}

export default App;
