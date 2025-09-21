/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Dark theme color palette - sophisticated grays (Dashboard v3)
        slate: {
          25: '#0a0e13',
          50: '#0f1419',
          100: '#16202b',
          150: '#1a252f',
          200: '#202c39',
          250: '#283542',
          300: '#334155',
          350: '#3f4c5c',
          400: '#4a5568',
          450: '#5a6575',
          500: '#64748b',
          600: '#94a3b8',
          700: '#cbd5e1',
          750: '#d1d9e1',
          800: '#e2e8f0',
          850: '#f1f5f9',
          900: '#f8fafc'
        },
        // Discord-inspired accents
        discord: {
          50: '#f0f4ff',
          100: '#e0e7ff',
          200: '#c7d2fe',
          300: '#a5b4fc',
          400: '#818cf8',
          500: '#6366f1', // Primary discord-like purple
          600: '#5145cd',
          700: '#4338ca',
          800: '#3730a3',
          900: '#312e81'
        },
        // Notion-inspired greens
        notion: {
          50: '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#22c55e', // Notion green
          600: '#16a34a',
          700: '#15803d',
          800: '#166534',
          900: '#14532d'
        },
        // Spotify-inspired accent
        spotify: {
          50: '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#1db954',
          600: '#1ed760'
        },
        // Legacy colors for backward compatibility
        primary: {
          50: '#eff6ff',
          100: '#dbeafe',
          500: '#6366f1', // Updated to match discord-500
          600: '#5145cd', // Updated to match discord-600
          700: '#4338ca', // Updated to match discord-700
          900: '#312e81', // Updated to match discord-900
        },
        gray: {
          50: '#0f1419', // Updated to match slate-50
          100: '#16202b', // Updated to match slate-100
          200: '#202c39', // Updated to match slate-200
          300: '#334155', // Updated to match slate-300
          400: '#4a5568', // Updated to match slate-400
          500: '#64748b', // Updated to match slate-500
          600: '#94a3b8', // Updated to match slate-600
          700: '#cbd5e1', // Updated to match slate-700
          800: '#e2e8f0', // Updated to match slate-800
          900: '#f8fafc', // Updated to match slate-900
        },
        success: '#22c55e', // Updated to match notion-500
        warning: '#f59e0b',
        danger: '#ef4444',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'pulse-soft': 'pulseSoft 2s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        pulseSoft: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.7' },
        },
      },
    },
  },
  plugins: [],
}