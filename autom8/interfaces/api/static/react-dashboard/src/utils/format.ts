import { formatDistanceToNow, format as dateFnsFormat } from 'date-fns';

/**
 * Format duration from seconds to human readable string
 */
export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  } else if (seconds < 3600) {
    return `${Math.round(seconds / 60)}m`;
  } else if (seconds < 86400) {
    return `${Math.round(seconds / 3600)}h`;
  } else {
    return `${Math.round(seconds / 86400)}d`;
  }
}

/**
 * Format bytes to human readable string
 */
export function formatBytes(bytes: number, decimals: number = 2): string {
  if (bytes === 0) return '0 B';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

/**
 * Format number with appropriate suffix (K, M, B)
 */
export function formatNumber(num: number, decimals: number = 1): string {
  if (num === 0) return '0';

  const k = 1000;
  const sizes = ['', 'K', 'M', 'B', 'T'];
  const i = Math.floor(Math.log(Math.abs(num)) / Math.log(k));

  if (i === 0) {
    return num.toString();
  }

  return parseFloat((num / Math.pow(k, i)).toFixed(decimals)) + sizes[i];
}

/**
 * Format percentage with proper rounding
 */
export function formatPercentage(value: number, decimals: number = 1): string {
  return `${value.toFixed(decimals)}%`;
}

/**
 * Format currency (assumes USD)
 */
export function formatCurrency(amount: number, decimals: number = 2): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(amount);
}

/**
 * Format date to relative time (e.g., "2 hours ago")
 */
export function formatRelativeTime(date: string | Date): string {
  try {
    const dateObj = typeof date === 'string' ? new Date(date) : date;
    return formatDistanceToNow(dateObj, { addSuffix: true });
  } catch {
    return 'Unknown';
  }
}

/**
 * Format date to absolute time
 */
export function formatAbsoluteTime(date: string | Date, formatStr: string = 'MMM dd, yyyy HH:mm:ss'): string {
  try {
    const dateObj = typeof date === 'string' ? new Date(date) : date;
    return dateFnsFormat(dateObj, formatStr);
  } catch {
    return 'Invalid date';
  }
}

/**
 * Format response time in appropriate units
 */
export function formatResponseTime(ms: number): string {
  if (ms < 1) {
    return `${(ms * 1000).toFixed(0)}μs`;
  } else if (ms < 1000) {
    return `${ms.toFixed(1)}ms`;
  } else {
    return `${(ms / 1000).toFixed(2)}s`;
  }
}

/**
 * Format a decimal to a percentage string
 */
export function decimalToPercentage(decimal: number, decimals: number = 1): string {
  return `${(decimal * 100).toFixed(decimals)}%`;
}

/**
 * Truncate text with ellipsis
 */
export function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength - 3) + '...';
}

/**
 * Get status color class based on value ranges
 */
export function getStatusColor(value: number, thresholds: { good: number; warning: number }): string {
  if (value >= thresholds.good) return 'text-green-500';
  if (value >= thresholds.warning) return 'text-yellow-500';
  return 'text-red-500';
}

/**
 * Get status color for success rates
 */
export function getSuccessRateColor(rate: number): string {
  return getStatusColor(rate, { good: 95, warning: 85 });
}

/**
 * Get status color for response times (lower is better)
 */
export function getResponseTimeColor(timeMs: number): string {
  if (timeMs <= 1000) return 'text-green-500';
  if (timeMs <= 3000) return 'text-yellow-500';
  return 'text-red-500';
}

/**
 * Capitalize first letter of string
 */
export function capitalize(str: string): string {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * Convert snake_case to Title Case
 */
export function snakeToTitle(str: string): string {
  return str
    .split('_')
    .map(word => capitalize(word))
    .join(' ');
}

/**
 * Generate color from string hash (for consistent coloring)
 */
export function stringToColor(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 70%, 60%)`;
}

/**
 * Format complexity level
 */
export function formatComplexityLevel(level: string): { label: string; color: string } {
  const levels = {
    trivial: { label: 'Trivial', color: 'text-green-500' },
    simple: { label: 'Simple', color: 'text-blue-500' },
    moderate: { label: 'Moderate', color: 'text-yellow-500' },
    complex: { label: 'Complex', color: 'text-orange-500' },
    frontier: { label: 'Frontier', color: 'text-red-500' },
  };
  
  return levels[level as keyof typeof levels] || { label: capitalize(level), color: 'text-gray-500' };
}