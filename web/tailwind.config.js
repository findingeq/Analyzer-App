/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#09090B",      // Deep Zinc Black (App BG)
        foreground: "#FAFAFA",      // Zinc 50 (Primary text)
        card: {
          DEFAULT: "#18181B",       // Lighter Zinc (Card/Sidebar BG)
          foreground: "#FAFAFA",
        },
        popover: {
          DEFAULT: "#18181B",
          foreground: "#FAFAFA",
        },

        // Brand Accents
        primary: {
          DEFAULT: "#6366F1",       // Indigo 500 (Main Action)
          foreground: "#FFFFFF",
        },
        secondary: {
          DEFAULT: "#A855F7",       // Purple 500 (Highlights)
          foreground: "#FFFFFF",
        },

        muted: {
          DEFAULT: "#27272A",       // Zinc 800
          foreground: "#A1A1AA",    // Zinc 400
        },

        accent: {
          DEFAULT: "#27272A",
          foreground: "#FAFAFA",
        },

        destructive: {
          DEFAULT: "#EF4444",
          foreground: "#FFFFFF",
        },

        border: "#27272A",          // Zinc 800
        input: "#27272A",
        ring: "#6366F1",            // Indigo 500

        // Chart Specific Colors
        "chart-ve": "#60A5FA",      // Blue 400 (VE Line)
        "chart-breath": "#60A5FA",  // Blue 400 (Breath dots)
        "chart-slope": "#C084FC",   // Purple 400 (Slopes)
        "chart-cusum-ok": "#34D399",    // Emerald 400
        "chart-cusum-alarm": "#F87171", // Red 400
        "chart-grid": "#27272A",    // Zinc 800 (Grid lines)
        "chart-ceiling": "#FB923C", // Orange 400

        // Threshold Status (Traffic Lights)
        "status-good": "#34D399",   // Emerald 400 (Below Threshold)
        "status-warn": "#FBBF24",   // Amber 400 (Borderline)
        "status-bad": "#F87171",    // Red 400 (Above Threshold)
        "recovery-zone": "#27272A", // Zinc 800 (Background shading)
        "ramp-zone": "rgba(251, 191, 36, 0.15)", // Amber with low opacity
      },
      borderRadius: {
        lg: "0.5rem",
        md: "calc(0.5rem - 2px)",
        sm: "calc(0.5rem - 4px)",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
}
