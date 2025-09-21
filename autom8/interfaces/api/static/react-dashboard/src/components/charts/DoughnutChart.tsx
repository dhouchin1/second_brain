import React from 'react';
import { Doughnut } from 'react-chartjs-2';
import { ChartData, ChartOptions } from 'chart.js';
import { doughnutChartOptions } from './chartConfig';
import { cn } from '@/utils/cn';

interface DoughnutChartProps {
  data: ChartData<'doughnut'>;
  options?: ChartOptions<'doughnut'>;
  height?: number;
  className?: string;
  title?: string;
  centerText?: string;
  centerValue?: string;
}

export function DoughnutChart({ 
  data, 
  options, 
  height = 300, 
  className,
  title,
  centerText,
  centerValue
}: DoughnutChartProps) {
  const chartOptions: ChartOptions<'doughnut'> = {
    ...doughnutChartOptions,
    ...options,
  };

  return (
    <div className={cn('w-full', className)}>
      {title && (
        <h3 className="text-lg font-semibold text-gray-200 mb-4">{title}</h3>
      )}
      <div className="relative" style={{ height }}>
        <Doughnut data={data} options={chartOptions} />
        
        {/* Center text overlay */}
        {(centerText || centerValue) && (
          <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
            {centerValue && (
              <div className="text-2xl font-bold text-gray-200">
                {centerValue}
              </div>
            )}
            {centerText && (
              <div className="text-sm text-gray-400 mt-1">
                {centerText}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}