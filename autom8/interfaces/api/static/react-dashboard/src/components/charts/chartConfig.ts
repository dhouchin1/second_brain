import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  TimeScale,
  ChartOptions,
} from 'chart.js';
import 'chartjs-adapter-date-fns';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  TimeScale
);

// Common color palette - Dashboard v3 theme
export const colors = {
  primary: '#6366f1', // discord-500
  secondary: '#ec4899', // pink-500
  success: '#22c55e', // notion-500
  warning: '#f59e0b',
  danger: '#ef4444',
  info: '#06b6d4',
  gray: '#64748b', // slate-500
  dark: '#0f1419', // slate-50
  light: '#f8fafc', // slate-900
} as const;

export const chartColors = [
  colors.primary,
  colors.secondary,
  colors.success,
  colors.warning,
  colors.danger,
  colors.info,
  '#f472b6', // pink
  '#a78bfa', // purple
  '#34d399', // emerald
  '#fbbf24', // amber
];

// Base chart options with dark theme
export const baseChartOptions: ChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: {
    intersect: false,
    mode: 'index',
  },
  elements: {
    point: {
      radius: 4,
      hoverRadius: 6,
    },
    line: {
      tension: 0.3,
    },
  },
  plugins: {
    legend: {
      display: true,
      position: 'top',
      labels: {
        usePointStyle: true,
        color: '#e5e7eb',
        font: {
          size: 12,
        },
        padding: 20,
      },
    },
    tooltip: {
      backgroundColor: '#1f2937',
      titleColor: '#f9fafb',
      bodyColor: '#e5e7eb',
      borderColor: '#374151',
      borderWidth: 1,
      cornerRadius: 8,
      displayColors: true,
      usePointStyle: true,
    },
  },
  scales: {
    x: {
      grid: {
        color: '#334155', // slate-300
        drawBorder: false,
      },
      ticks: {
        color: '#64748b', // slate-500
        font: {
          size: 11,
        },
      },
    },
    y: {
      grid: {
        color: '#334155', // slate-300
        drawBorder: false,
      },
      ticks: {
        color: '#64748b', // slate-500
        font: {
          size: 11,
        },
      },
    },
  },
};

// Line chart specific options
export const lineChartOptions: ChartOptions<'line'> = {
  ...baseChartOptions,
  plugins: {
    ...baseChartOptions.plugins,
    filler: {
      propagate: false,
    },
  },
};

// Bar chart specific options
export const barChartOptions: ChartOptions<'bar'> = {
  ...baseChartOptions,
  scales: {
    ...baseChartOptions.scales,
    y: {
      ...baseChartOptions.scales?.y,
      beginAtZero: true,
    },
  },
};

// Doughnut chart specific options
export const doughnutChartOptions: ChartOptions<'doughnut'> = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      display: true,
      position: 'right',
      labels: {
        usePointStyle: true,
        color: '#e5e7eb',
        font: {
          size: 12,
        },
        padding: 15,
        generateLabels: (chart) => {
          const data = chart.data;
          if (data.labels && data.datasets.length) {
            return data.labels.map((label, i) => ({
              text: String(label),
              fillStyle: Array.isArray(data.datasets[0].backgroundColor) 
                ? data.datasets[0].backgroundColor[i] 
                : data.datasets[0].backgroundColor,
              strokeStyle: Array.isArray(data.datasets[0].borderColor)
                ? data.datasets[0].borderColor[i]
                : data.datasets[0].borderColor,
              lineWidth: 2,
              pointStyle: 'circle',
              hidden: false,
              index: i,
            }));
          }
          return [];
        },
      },
    },
    tooltip: {
      backgroundColor: '#1f2937',
      titleColor: '#f9fafb',
      bodyColor: '#e5e7eb',
      borderColor: '#374151',
      borderWidth: 1,
      cornerRadius: 8,
      displayColors: true,
      usePointStyle: true,
      callbacks: {
        label: (context) => {
          const label = context.label || '';
          const value = context.parsed;
          const total = context.dataset.data.reduce((a: number, b: number) => a + b, 0);
          const percentage = ((value / total) * 100).toFixed(1);
          return `${label}: ${value} (${percentage}%)`;
        },
      },
    },
  },
  cutout: '60%',
};

// Time series options
export const timeSeriesOptions: ChartOptions<'line'> = {
  ...lineChartOptions,
  scales: {
    x: {
      type: 'time',
      time: {
        displayFormats: {
          minute: 'HH:mm',
          hour: 'HH:mm',
          day: 'MMM dd',
        },
      },
      grid: {
        color: '#334155', // slate-300
        drawBorder: false,
      },
      ticks: {
        color: '#64748b', // slate-500
        font: {
          size: 11,
        },
        maxTicksLimit: 8,
      },
    },
    y: {
      grid: {
        color: '#334155', // slate-300
        drawBorder: false,
      },
      ticks: {
        color: '#64748b', // slate-500
        font: {
          size: 11,
        },
      },
    },
  },
};

// Real-time chart options with animation disabled for performance
export const realtimeChartOptions: ChartOptions<'line'> = {
  ...timeSeriesOptions,
  animation: false,
  parsing: false,
  normalized: true,
  spanGaps: 1000 * 60 * 5, // 5 minutes
  plugins: {
    ...timeSeriesOptions.plugins,
    legend: {
      ...timeSeriesOptions.plugins?.legend,
      display: true,
    },
  },
};

// Utility function to create gradient
export function createGradient(
  ctx: CanvasRenderingContext2D,
  chartArea: { top: number; bottom: number; left: number; right: number },
  colorStart: string,
  colorEnd: string = colorStart + '10'
): CanvasGradient {
  const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
  gradient.addColorStop(0, colorStart);
  gradient.addColorStop(1, colorEnd);
  return gradient;
}

// Utility function to generate dataset with consistent styling
export function createDataset(
  label: string,
  data: any[],
  colorIndex: number = 0,
  type: 'line' | 'bar' | 'area' = 'line'
) {
  const color = chartColors[colorIndex % chartColors.length];
  
  const baseDataset = {
    label,
    data,
    borderColor: color,
    backgroundColor: type === 'area' ? color + '20' : color,
    borderWidth: 2,
  };

  if (type === 'area') {
    return {
      ...baseDataset,
      fill: true,
      backgroundColor: color + '20',
    };
  }

  if (type === 'bar') {
    return {
      ...baseDataset,
      backgroundColor: color,
      borderColor: color,
      borderWidth: 1,
    };
  }

  return baseDataset;
}