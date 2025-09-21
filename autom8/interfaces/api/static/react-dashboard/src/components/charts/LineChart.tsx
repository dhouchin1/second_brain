import React from 'react';
import { Line } from 'react-chartjs-2';
import { ChartData, ChartOptions } from 'chart.js';
import { lineChartOptions, realtimeChartOptions } from './chartConfig';
import { cn } from '@/utils/cn';

interface LineChartProps {
  data: ChartData<'line'>;
  options?: ChartOptions<'line'>;
  height?: number;
  className?: string;
  realtime?: boolean;
  title?: string;
}

export function LineChart({ 
  data, 
  options, 
  height = 300, 
  className,
  realtime = false,
  title
}: LineChartProps) {
  const chartOptions = {
    ...(realtime ? realtimeChartOptions : lineChartOptions),
    ...options,
  };

  return (
    <div className={cn('w-full', className)}>
      {title && (
        <h3 className="text-lg font-semibold text-gray-200 mb-4">{title}</h3>
      )}
      <div style={{ height }}>
        <Line data={data} options={chartOptions} />
      </div>
    </div>
  );
}