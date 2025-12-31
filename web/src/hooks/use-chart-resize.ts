/**
 * useChartResize Hook
 * Handles chart resize when container dimensions change
 */

import { useEffect, useRef, useCallback, useState } from "react";
import type { EChartsType } from "echarts";

interface UseChartResizeOptions {
  /**
   * Debounce delay in milliseconds
   * @default 100
   */
  debounceMs?: number;
}

/**
 * Hook to handle ECharts resize on container size changes
 * Uses ResizeObserver for efficient resize detection
 */
export function useChartResize(
  options: UseChartResizeOptions = {}
): {
  containerRef: React.RefObject<HTMLDivElement | null>;
  chartRef: React.MutableRefObject<EChartsType | null>;
  resize: () => void;
} {
  const { debounceMs = 100 } = options;

  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<EChartsType | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const resize = useCallback(() => {
    if (chartRef.current) {
      chartRef.current.resize();
    }
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver(() => {
      // Debounce resize calls
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      timeoutRef.current = setTimeout(() => {
        resize();
      }, debounceMs);
    });

    observer.observe(container);

    return () => {
      observer.disconnect();
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [debounceMs, resize]);

  // Also resize on window resize for good measure
  useEffect(() => {
    const handleWindowResize = () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      timeoutRef.current = setTimeout(() => {
        resize();
      }, debounceMs);
    };

    window.addEventListener("resize", handleWindowResize);
    return () => {
      window.removeEventListener("resize", handleWindowResize);
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [debounceMs, resize]);

  return { containerRef, chartRef, resize };
}

/**
 * Simple hook for tracking element dimensions
 */
export function useElementSize(): {
  ref: React.RefObject<HTMLDivElement | null>;
  width: number;
  height: number;
} {
  const ref = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setSize({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    });

    observer.observe(element);

    // Set initial size
    setSize({
      width: element.offsetWidth,
      height: element.offsetHeight,
    });

    return () => observer.disconnect();
  }, []);

  return { ref, ...size };
}
