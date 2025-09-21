import React from 'react';
import { Bar } from 'react-chartjs-2';
import { ChartData, ChartOptions } from 'chart.js';
import { barChartOptions } from './chartConfig';
import { cn } from '@/utils/cn';

interface BarChartProps {
  data: ChartData<'bar'>;
  options?: ChartOptions<'bar'>;
  height?: number;
  className?: string;
  title?: string;
  horizontal?: boolean;
}

export function BarChart({ 
  data, 
  options, 
  height = 300, 
  className,
  title,
  horizontal = false
}: BarChartProps) {
  const chartOptions: ChartOptions<'bar'> = {
    ...barChartOptions,
    indexAxis: horizontal ? 'y' : 'x',
    ...options,
  };

  return (
    <div className={cn('w-full', className)}>
      {title && (
        <h3 className="text-lg font-semibold text-gray-200 mb-4">{title}</h3>
      )}
      <div style={{ height }}>
        <Bar data={data} options={chartOptions} />
      </div>
    </div>
  );
}