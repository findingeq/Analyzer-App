/**
 * VT Threshold Analyzer - Main App Component
 */

import { DashboardShell } from "@/components/layout/DashboardShell";
import { Sidebar } from "@/components/layout/Sidebar";
import { MainChart } from "@/components/charts/MainChart";
import { IntervalMetrics } from "@/components/charts/IntervalMetrics";
import { HeaderMetrics } from "@/components/layout/HeaderMetrics";
import { VEApprovalDialog } from "@/components/calibration/VEApprovalDialog";
import { useRunStore } from "@/store/use-run-store";

function App() {
  const { pendingVEPrompt, clearVEPrompt, setVt1Ceiling, setVt2Ceiling } = useRunStore();

  const handleVEApproved = (threshold: "vt1" | "vt2", newValue: number) => {
    if (threshold === "vt1") {
      setVt1Ceiling(newValue);
    } else {
      setVt2Ceiling(newValue);
    }
  };

  return (
    <>
      <DashboardShell
        sidebar={<Sidebar />}
        chart={<MainChart />}
        footer={<IntervalMetrics />}
        headerMetrics={<HeaderMetrics />}
      />
      <VEApprovalDialog
        prompt={pendingVEPrompt}
        onClose={clearVEPrompt}
        onApproved={handleVEApproved}
      />
    </>
  );
}

export default App;
