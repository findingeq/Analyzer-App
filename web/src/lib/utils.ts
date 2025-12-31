import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Format seconds to mm:ss display
 */
export function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

/**
 * Format a number with specified decimal places
 */
export function formatNumber(value: number, decimals: number = 1): string {
  return value.toFixed(decimals)
}

/**
 * Get status color class based on interval status
 */
export function getStatusColor(status: string): string {
  switch (status) {
    case 'BELOW_THRESHOLD':
      return 'text-status-good'
    case 'BORDERLINE':
      return 'text-status-warn'
    case 'ABOVE_THRESHOLD':
      return 'text-status-bad'
    default:
      return 'text-muted-foreground'
  }
}

/**
 * Get status background color class
 */
export function getStatusBgColor(status: string): string {
  switch (status) {
    case 'BELOW_THRESHOLD':
      return 'bg-status-good/20'
    case 'BORDERLINE':
      return 'bg-status-warn/20'
    case 'ABOVE_THRESHOLD':
      return 'bg-status-bad/20'
    default:
      return 'bg-muted'
  }
}
