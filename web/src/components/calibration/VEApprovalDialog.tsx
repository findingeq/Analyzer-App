/**
 * VEApprovalDialog Component
 * Shows when calibration suggests a VE threshold change >= 1 L/min
 */

import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { approveVEThreshold } from "@/lib/client";

interface VEPrompt {
  threshold: "vt1" | "vt2";
  current_value: number;
  proposed_value: number;
  pending_delta: number;
}

interface VEApprovalDialogProps {
  prompt: VEPrompt | null;
  onClose: () => void;
  onApproved?: (threshold: "vt1" | "vt2", newValue: number) => void;
}

export function VEApprovalDialog({
  prompt,
  onClose,
  onApproved,
}: VEApprovalDialogProps) {
  const [isSubmitting, setIsSubmitting] = useState(false);

  if (!prompt) return null;

  const thresholdName = prompt.threshold === "vt1" ? "VT1" : "VT2";
  const direction = prompt.pending_delta > 0 ? "increase" : "decrease";
  const absChange = Math.abs(prompt.pending_delta).toFixed(1);

  const handleApprove = async () => {
    setIsSubmitting(true);
    try {
      const result = await approveVEThreshold(prompt.threshold, true);
      onApproved?.(prompt.threshold, result.new_value);
      onClose();
    } catch (error) {
      console.error("Failed to approve VE threshold:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleReject = async () => {
    setIsSubmitting(true);
    try {
      await approveVEThreshold(prompt.threshold, false);
      onClose();
    } catch (error) {
      console.error("Failed to reject VE threshold:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Dialog open={!!prompt} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Update {thresholdName} Threshold?</DialogTitle>
          <DialogDescription>
            Based on your recent runs, calibration suggests adjusting your{" "}
            {thresholdName} VE threshold.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Current Value</span>
            <span className="text-sm font-medium">
              {prompt.current_value.toFixed(1)} L/min
            </span>
          </div>

          <div className="flex items-center justify-center">
            <div className="flex items-center gap-2">
              <span className="text-2xl font-bold">
                {prompt.current_value.toFixed(1)}
              </span>
              <span className="text-muted-foreground">→</span>
              <span
                className={`text-2xl font-bold ${
                  direction === "increase" ? "text-yellow-400" : "text-emerald-400"
                }`}
              >
                {prompt.proposed_value.toFixed(1)}
              </span>
              <span className="text-sm text-muted-foreground">L/min</span>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Change</span>
            <span
              className={`text-sm font-medium ${
                direction === "increase" ? "text-yellow-400" : "text-emerald-400"
              }`}
            >
              {direction === "increase" ? "+" : "-"}
              {absChange} L/min
            </span>
          </div>

          <p className="text-xs text-muted-foreground">
            This will also sync to your iPhone app on next launch.
          </p>
        </div>

        <DialogFooter className="flex gap-2 sm:gap-0">
          <Button
            variant="outline"
            onClick={handleReject}
            disabled={isSubmitting}
          >
            Keep Current
          </Button>
          <Button onClick={handleApprove} disabled={isSubmitting}>
            {isSubmitting ? "Updating..." : "Apply Change"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export default VEApprovalDialog;
