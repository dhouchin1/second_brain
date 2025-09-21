import React from 'react';
import { LucideIcon } from 'lucide-react';
import { cn } from '@/utils/cn';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: LucideIcon;
  trend?: {
    value: number;
    direction: 'up' | 'down' | 'neutral';
    label?: string;
  };
  color?: 'blue' | 'green' | 'yellow' | 'red' | 'purple' | 'gray';
  className?: string;
  onClick?: () => void;
}

const colorVariants = {
  blue: {
    bg: 'bg-discord-500/10',
    text: 'text-discord-400',
    border: 'border-discord-500/20',
    icon: 'text-discord-400',
  },
  green: {
    bg: 'bg-notion-500/10',
    text: 'text-notion-400',
    border: 'border-notion-500/20',
    icon: 'text-notion-400',
  },
  yellow: {
    bg: 'bg-yellow-500/10',
    text: 'text-yellow-400',
    border: 'border-yellow-500/20',
    icon: 'text-yellow-400',
  },
  red: {
    bg: 'bg-red-500/10',
    text: 'text-red-400',
    border: 'border-red-500/20',
    icon: 'text-red-400',
  },
  purple: {
    bg: 'bg-discord-500/10',
    text: 'text-discord-400',
    border: 'border-discord-500/20',
    icon: 'text-discord-400',
  },
  gray: {
    bg: 'bg-slate-300/10',
    text: 'text-slate-500',
    border: 'border-slate-300/20',
    icon: 'text-slate-500',
  },
};

export function MetricCard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  color = 'blue',
  className,
  onClick,
}: MetricCardProps) {
  const colors = colorVariants[color];

  const getTrendColor = (direction: string) => {
    switch (direction) {
      case 'up':
        return 'text-notion-500';
      case 'down':
        return 'text-red-400';
      default:
        return 'text-slate-500';
    }
  };

  const getTrendIcon = (direction: string) => {
    switch (direction) {
      case 'up':
        return '↗';
      case 'down':
        return '↘';
      default:
        return '→';
    }
  };

  return (
    <div
      className={cn(
        'rounded-lg border p-6 transition-all duration-200',
        'bg-slate-100/80 backdrop-blur-sm',
        colors.border,
        onClick && 'cursor-pointer hover:bg-slate-100/90',
        className
      )}
      onClick={onClick}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-slate-500">{title}</p>
          <div className="mt-2 flex items-baseline">
            <p className={cn('text-3xl font-semibold', colors.text)}>
              {value}
            </p>
            {trend && (
              <div className={cn('ml-2 text-sm', getTrendColor(trend.direction))}>
                <span className="inline-flex items-center">
                  <span className="mr-1">{getTrendIcon(trend.direction)}</span>
                  {Math.abs(trend.value)}%
                  {trend.label && <span className="ml-1">{trend.label}</span>}
                </span>
              </div>
            )}
          </div>
          {subtitle && (
            <p className="mt-1 text-sm text-slate-400">{subtitle}</p>
          )}
        </div>
        
        {Icon && (
          <div className={cn('rounded-full p-3', colors.bg)}>
            <Icon className={cn('h-6 w-6', colors.icon)} />
          </div>
        )}
      </div>
    </div>
  );
}