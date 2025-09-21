import React, { useState } from 'react';
import {
  Cpu,
  TrendingUp,
  DollarSign,
  Clock,
  CheckCircle,
  AlertTriangle,
  Zap,
  Cloud,
  Monitor,
  MoreVertical
} from 'lucide-react';
import { MetricCard } from '@/components/charts/MetricCard';
import { BarChart } from '@/components/charts/BarChart';
import { LineChart } from '@/components/charts/LineChart';
import { DoughnutChart } from '@/components/charts/DoughnutChart';
import { useModelsStatus } from '@/hooks/useApi';
import { ModelInfo } from '@/types/api';
import { 
  formatCurrency, 
  formatResponseTime, 
  formatPercentage, 
  formatRelativeTime,
  getSuccessRateColor,
  getResponseTimeColor 
} from '@/utils/format';
import { chartColors, createDataset } from '@/components/charts/chartConfig';
import { cn } from '@/utils/cn';

export function ModelsDashboard() {
  const { data: modelsStatus, isLoading, refetch } = useModelsStatus();
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const getProviderIcon = (provider: string) => {
    switch (provider.toLowerCase()) {
      case 'ollama': return Monitor;
      case 'openai': return Cloud;
      case 'anthropic': return Cloud;
      default: return Cpu;
    }
  };

  const getProviderColor = (provider: string) => {
    switch (provider.toLowerCase()) {
      case 'ollama': return 'text-blue-400';
      case 'openai': return 'text-green-400';
      case 'anthropic': return 'text-purple-400';
      default: return 'text-gray-400';
    }
  };

  // Prepare charts data
  const requestsChartData = modelsStatus?.models ? {
    labels: modelsStatus.models.map(model => model.name),
    datasets: [
      createDataset(
        'Requests Count',
        modelsStatus.models.map(model => model.requests_count),
        0,
        'bar'
      ),
    ],
  } : { labels: [], datasets: [] };

  const responseTimeChartData = modelsStatus?.models ? {
    labels: modelsStatus.models.map(model => model.name),
    datasets: [
      createDataset(
        'Response Time (ms)',
        modelsStatus.models.map(model => model.avg_response_time * 1000),
        1,
        'bar'
      ),
    ],
  } : { labels: [], datasets: [] };

  const successRateChartData = modelsStatus?.models ? {
    labels: modelsStatus.models.map(model => model.name),
    datasets: [
      createDataset(
        'Success Rate (%)',
        modelsStatus.models.map(model => model.success_rate),
        2,
        'bar'
      ),
    ],
  } : { labels: [], datasets: [] };

  // Provider distribution
  const providerDistribution = modelsStatus?.models.reduce((acc, model) => {
    acc[model.provider] = (acc[model.provider] || 0) + 1;
    return acc;
  }, {} as Record<string, number>) || {};

  const providerChartData = {
    labels: Object.keys(providerDistribution),
    datasets: [{
      data: Object.values(providerDistribution),
      backgroundColor: Object.keys(providerDistribution).map((_, index) => 
        chartColors[index % chartColors.length]
      ),
      borderColor: Object.keys(providerDistribution).map((_, index) => 
        chartColors[index % chartColors.length]
      ),
      borderWidth: 2,
    }],
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="flex items-center space-x-3">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="text-gray-400">Loading models data...</span>
        </div>
      </div>
    );
  }

  if (!modelsStatus || modelsStatus.models.length === 0) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <Cpu className="h-16 w-16 mx-auto text-gray-500 mb-4" />
          <h3 className="text-lg font-medium text-gray-200 mb-2">No Models Available</h3>
          <p className="text-gray-400">No model performance data found</p>
          <button
            onClick={refetch}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const availableModels = modelsStatus.models.filter(model => model.available);
  const avgSuccessRate = modelsStatus.models.reduce((sum, model) => sum + model.success_rate, 0) / modelsStatus.models.length;
  const avgResponseTime = modelsStatus.models.reduce((sum, model) => sum + model.avg_response_time, 0) / modelsStatus.models.length;

  return (
    <div className="space-y-6">
      {/* Dashboard Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Models Dashboard</h1>
          <p className="text-gray-400 mt-1">Monitor model performance and usage</p>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={refetch}
            className="px-4 py-2 bg-gray-700 text-gray-200 rounded-lg hover:bg-gray-600 transition-colors"
          >
            Refresh
          </button>
          {modelsStatus.preferred_model && (
            <div className="px-3 py-2 bg-blue-600/20 text-blue-400 rounded-lg border border-blue-500/30">
              <div className="flex items-center space-x-2">
                <Zap className="h-4 w-4" />
                <span className="text-sm font-medium">
                  Preferred: {modelsStatus.preferred_model}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Available Models"
          value={availableModels.length}
          subtitle={`${modelsStatus.models.length} total models`}
          icon={CheckCircle}
          color="green"
        />
        
        <MetricCard
          title="Total Requests"
          value={modelsStatus.total_requests.toLocaleString()}
          icon={TrendingUp}
          color="blue"
        />
        
        <MetricCard
          title="Total Cost"
          value={formatCurrency(modelsStatus.total_cost)}
          icon={DollarSign}
          color="yellow"
        />
        
        <MetricCard
          title="Avg Success Rate"
          value={formatPercentage(avgSuccessRate)}
          subtitle={`${formatResponseTime(avgResponseTime * 1000)} avg response`}
          icon={CheckCircle}
          color={avgSuccessRate >= 95 ? 'green' : avgSuccessRate >= 90 ? 'yellow' : 'red'}
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Requests by Model */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-200 mb-4">Requests by Model</h3>
          <BarChart data={requestsChartData} height={300} />
        </div>

        {/* Provider Distribution */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-200 mb-4">Provider Distribution</h3>
          <DoughnutChart
            data={providerChartData}
            height={300}
            centerText="Total Models"
            centerValue={modelsStatus.models.length.toString()}
          />
        </div>
      </div>

      {/* Response Time and Success Rate Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-200 mb-4">Average Response Time</h3>
          <BarChart 
            data={responseTimeChartData} 
            height={300}
            options={{
              scales: {
                y: {
                  beginAtZero: true,
                  ticks: {
                    callback: (value) => formatResponseTime(Number(value)),
                  },
                },
              },
            }}
          />
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-200 mb-4">Success Rates</h3>
          <BarChart 
            data={successRateChartData} 
            height={300}
            options={{
              scales: {
                y: {
                  beginAtZero: true,
                  max: 100,
                  ticks: {
                    callback: (value) => `${value}%`,
                  },
                },
              },
            }}
          />
        </div>
      </div>

      {/* Model Details Table */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-200">Model Details</h3>
          <span className="text-sm text-gray-400">
            {modelsStatus.models.length} models
          </span>
        </div>

        <div className="space-y-4">
          {modelsStatus.models.map((model) => {
            const ProviderIcon = getProviderIcon(model.provider);
            return (
              <div
                key={model.name}
                className={cn(
                  'p-4 rounded-lg border transition-all duration-200',
                  'bg-gray-700/30 border-gray-600 hover:bg-gray-700/50',
                  selectedModel === model.name && 'ring-2 ring-blue-500 bg-gray-700/50'
                )}
                onClick={() => setSelectedModel(selectedModel === model.name ? null : model.name)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-4">
                    {/* Model Icon and Basic Info */}
                    <div className="flex-shrink-0">
                      <div className={cn(
                        'w-10 h-10 rounded-lg flex items-center justify-center',
                        model.available ? 'bg-green-500/20' : 'bg-red-500/20'
                      )}>
                        <ProviderIcon className={cn(
                          'h-5 w-5',
                          model.available ? 'text-green-400' : 'text-red-400'
                        )} />
                      </div>
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-3 mb-2">
                        <h4 className="text-sm font-medium text-gray-200">
                          {model.name}
                        </h4>
                        <span className={cn(
                          'px-2 py-1 rounded-full text-xs font-medium',
                          getProviderColor(model.provider),
                          'bg-gray-600/30'
                        )}>
                          {model.provider}
                        </span>
                        <span className={cn(
                          'px-2 py-1 rounded-full text-xs font-medium',
                          model.available 
                            ? 'bg-green-500/20 text-green-400'
                            : 'bg-red-500/20 text-red-400'
                        )}>
                          {model.available ? 'Available' : 'Unavailable'}
                        </span>
                      </div>

                      {/* Model Metrics Grid */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                        <div>
                          <span className="text-gray-400">Requests</span>
                          <div className="text-gray-200 font-medium mt-1">
                            {model.requests_count.toLocaleString()}
                          </div>
                        </div>
                        <div>
                          <span className="text-gray-400">Success Rate</span>
                          <div className={cn('font-medium mt-1', getSuccessRateColor(model.success_rate))}>
                            {formatPercentage(model.success_rate)}
                          </div>
                        </div>
                        <div>
                          <span className="text-gray-400">Avg Response</span>
                          <div className={cn('font-medium mt-1', getResponseTimeColor(model.avg_response_time * 1000))}>
                            {formatResponseTime(model.avg_response_time * 1000)}
                          </div>
                        </div>
                        <div>
                          <span className="text-gray-400">Cost per Request</span>
                          <div className="text-gray-200 font-medium mt-1">
                            {model.cost_per_request > 0 
                              ? formatCurrency(model.cost_per_request)
                              : 'Free'}
                          </div>
                        </div>
                      </div>

                      {model.last_used && (
                        <p className="text-xs text-gray-500 mt-2">
                          Last used: {formatRelativeTime(model.last_used)}
                        </p>
                      )}
                    </div>
                  </div>

                  <button className="text-gray-400 hover:text-gray-200">
                    <MoreVertical className="h-4 w-4" />
                  </button>
                </div>

                {/* Expanded Details */}
                {selectedModel === model.name && (
                  <div className="mt-4 pt-4 border-t border-gray-600">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                      <div>
                        <h5 className="text-gray-400 font-medium mb-2">Performance</h5>
                        <div className="space-y-1">
                          <div className="flex justify-between">
                            <span className="text-gray-300">Success Rate:</span>
                            <span className={getSuccessRateColor(model.success_rate)}>
                              {formatPercentage(model.success_rate)}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-300">Response Time:</span>
                            <span className={getResponseTimeColor(model.avg_response_time * 1000)}>
                              {formatResponseTime(model.avg_response_time * 1000)}
                            </span>
                          </div>
                        </div>
                      </div>
                      
                      <div>
                        <h5 className="text-gray-400 font-medium mb-2">Usage</h5>
                        <div className="space-y-1">
                          <div className="flex justify-between">
                            <span className="text-gray-300">Total Requests:</span>
                            <span className="text-gray-200">{model.requests_count}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-300">Total Cost:</span>
                            <span className="text-gray-200">
                              {formatCurrency(model.requests_count * model.cost_per_request)}
                            </span>
                          </div>
                        </div>
                      </div>
                      
                      <div>
                        <h5 className="text-gray-400 font-medium mb-2">Status</h5>
                        <div className="space-y-1">
                          <div className="flex justify-between">
                            <span className="text-gray-300">Provider:</span>
                            <span className={getProviderColor(model.provider)}>
                              {model.provider}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-300">Available:</span>
                            <span className={model.available ? 'text-green-400' : 'text-red-400'}>
                              {model.available ? 'Yes' : 'No'}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}