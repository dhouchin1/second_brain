import React from 'react';
import {
  HardDrive,
  Cpu,
  MemoryStick,
  TrendingUp,
  Target,
  Zap,
  Archive,
  BarChart3,
  PieChart
} from 'lucide-react';
import { MetricCard } from '@/components/charts/MetricCard';
import { LineChart } from '@/components/charts/LineChart';
import { BarChart } from '@/components/charts/BarChart';
import { DoughnutChart } from '@/components/charts/DoughnutChart';
import { useComplexityStats, useRoutingStats, useContextStats } from '@/hooks/useApi';
import { 
  formatNumber, 
  formatBytes, 
  formatPercentage, 
  formatComplexityLevel,
  decimalToPercentage 
} from '@/utils/format';
import { chartColors, createDataset } from '@/components/charts/chartConfig';

export function ResourcesDashboard() {
  const { data: complexityStats, isLoading: complexityLoading } = useComplexityStats();
  const { data: routingStats, isLoading: routingLoading } = useRoutingStats();
  const { data: contextStats, isLoading: contextLoading } = useContextStats();

  const loading = complexityLoading || routingLoading || contextLoading;

  // Complexity distribution chart
  const complexityDistributionData = complexityStats ? {
    labels: Object.keys(complexityStats.distribution).map(level => 
      formatComplexityLevel(level).label
    ),
    datasets: [{
      data: Object.values(complexityStats.distribution),
      backgroundColor: Object.keys(complexityStats.distribution).map((level, index) => {
        const { color } = formatComplexityLevel(level);
        return chartColors[index % chartColors.length];
      }),
      borderColor: Object.keys(complexityStats.distribution).map((level, index) => 
        chartColors[index % chartColors.length]
      ),
      borderWidth: 2,
    }],
  } : { labels: [], datasets: [] };

  // Routing distribution chart
  const routingDistributionData = routingStats ? {
    labels: ['Local Routes', 'Cloud Routes'],
    datasets: [{
      data: [routingStats.local_routes, routingStats.cloud_routes],
      backgroundColor: [chartColors[0], chartColors[1]],
      borderColor: [chartColors[0], chartColors[1]],
      borderWidth: 2,
    }],
  } : { labels: [], datasets: [] };

  // Model routing distribution
  const modelRoutingData = routingStats ? {
    labels: Object.keys(routingStats.model_distribution),
    datasets: [
      createDataset(
        'Routes',
        Object.values(routingStats.model_distribution),
        0,
        'bar'
      ),
    ],
  } : { labels: [], datasets: [] };

  // Context size distribution
  const contextSizeData = contextStats ? {
    labels: Object.keys(contextStats.size_distribution || {}),
    datasets: [{
      data: Object.values(contextStats.size_distribution || {}),
      backgroundColor: [chartColors[2], chartColors[3], chartColors[4]],
      borderColor: [chartColors[2], chartColors[3], chartColors[4]],
      borderWidth: 2,
    }],
  } : { labels: [], datasets: [] };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="flex items-center space-x-3">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="text-gray-400">Loading resources data...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Dashboard Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Resources Dashboard</h1>
          <p className="text-gray-400 mt-1">Monitor resource utilization and optimization</p>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Complexity Analyses"
          value={formatNumber(complexityStats?.total_analyses || 0)}
          subtitle={`Avg score: ${(complexityStats?.avg_complexity_score || 0).toFixed(2)}`}
          icon={BarChart3}
          color="blue"
        />
        
        <MetricCard
          title="Routing Accuracy"
          value={formatPercentage(routingStats?.routing_accuracy || 0)}
          subtitle={`${formatNumber(routingStats?.total_routes || 0)} total routes`}
          icon={Target}
          color="green"
        />
        
        <MetricCard
          title="Context Cache Hit Rate"
          value={formatPercentage(contextStats?.cache_hit_rate || 0)}
          subtitle="Memory optimization"
          icon={Zap}
          color="purple"
        />
        
        <MetricCard
          title="Optimization Savings"
          value={formatBytes(contextStats?.optimization_savings || 0)}
          subtitle="Tokens saved"
          icon={TrendingUp}
          color="yellow"
        />
      </div>

      {/* Complexity Analysis Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Complexity Distribution */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-200">Complexity Distribution</h3>
            <BarChart3 className="h-5 w-5 text-gray-400" />
          </div>
          {complexityStats && Object.keys(complexityStats.distribution).length > 0 ? (
            <DoughnutChart
              data={complexityDistributionData}
              height={300}
              centerText="Total Analyses"
              centerValue={formatNumber(complexityStats.total_analyses)}
            />
          ) : (
            <div className="h-[300px] flex items-center justify-center text-gray-500">
              <div className="text-center">
                <PieChart className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No complexity data available</p>
              </div>
            </div>
          )}
        </div>

        {/* Complexity Stats */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-200">Complexity Analysis Stats</h3>
            <Cpu className="h-5 w-5 text-gray-400" />
          </div>
          <div className="space-y-4">
            {complexityStats && Object.entries(complexityStats.distribution).map(([level, count]) => {
              const { label, color } = formatComplexityLevel(level);
              const percentage = (count / complexityStats.total_analyses) * 100;
              return (
                <div key={level} className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className={`w-3 h-3 rounded-full bg-current ${color}`} />
                    <span className="text-gray-300">{label}</span>
                  </div>
                  <div className="text-right">
                    <div className="text-gray-200 font-medium">{count}</div>
                    <div className="text-xs text-gray-400">{percentage.toFixed(1)}%</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Routing Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Routing Distribution */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-200">Routing Distribution</h3>
            <Target className="h-5 w-5 text-gray-400" />
          </div>
          {routingStats ? (
            <DoughnutChart
              data={routingDistributionData}
              height={300}
              centerText="Total Routes"
              centerValue={formatNumber(routingStats.total_routes)}
            />
          ) : (
            <div className="h-[300px] flex items-center justify-center text-gray-500">
              <div className="text-center">
                <Target className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No routing data available</p>
              </div>
            </div>
          )}
        </div>

        {/* Model Routing */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-200 mb-4">Routes by Model</h3>
          {routingStats && Object.keys(routingStats.model_distribution).length > 0 ? (
            <BarChart data={modelRoutingData} height={300} />
          ) : (
            <div className="h-[300px] flex items-center justify-center text-gray-500">
              <div className="text-center">
                <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No model routing data available</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Context Management Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Context Size Distribution */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-200">Context Size Distribution</h3>
            <Archive className="h-5 w-5 text-gray-400" />
          </div>
          {contextStats && contextStats.size_distribution && Object.keys(contextStats.size_distribution).length > 0 ? (
            <DoughnutChart
              data={contextSizeData}
              height={300}
              centerText="Total Contexts"
              centerValue={formatNumber(contextStats.total_contexts)}
            />
          ) : (
            <div className="h-[300px] flex items-center justify-center text-gray-500">
              <div className="text-center">
                <Archive className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No context size data available</p>
              </div>
            </div>
          )}
        </div>

        {/* Context Statistics */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-200">Context Optimization</h3>
            <MemoryStick className="h-5 w-5 text-gray-400" />
          </div>
          <div className="space-y-6">
            {contextStats && (
              <>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Total Contexts</span>
                    <span className="text-gray-200 font-medium">
                      {formatNumber(contextStats.total_contexts)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Average Size</span>
                    <span className="text-gray-200 font-medium">
                      {contextStats.avg_context_size.toFixed(1)} tokens
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Cache Hit Rate</span>
                    <span className="text-green-400 font-medium">
                      {formatPercentage(contextStats.cache_hit_rate)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Memory Efficiency</span>
                    <span className="text-blue-400 font-medium">
                      {formatPercentage(contextStats.memory_efficiency)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Tokens Saved</span>
                    <span className="text-yellow-400 font-medium">
                      {formatBytes(contextStats.optimization_savings)}
                    </span>
                  </div>
                </div>

                {/* Optimization Techniques */}
                {contextStats.optimization_techniques && (
                  <div>
                    <h4 className="text-sm font-medium text-gray-300 mb-3">Optimization Techniques</h4>
                    <div className="space-y-2">
                      {Object.entries(contextStats.optimization_techniques).map(([technique, count]) => {
                        const total = Object.values(contextStats.optimization_techniques).reduce((a, b) => a + b, 0);
                        const percentage = (count / total) * 100;
                        return (
                          <div key={technique} className="flex items-center justify-between">
                            <span className="text-gray-400 text-sm capitalize">
                              {technique.replace('_', ' ')}
                            </span>
                            <div className="text-right">
                              <span className="text-gray-200 font-medium">{count}</span>
                              <span className="text-xs text-gray-500 ml-2">
                                ({percentage.toFixed(1)}%)
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      {/* Performance Summary */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-200">Resource Performance Summary</h3>
          <TrendingUp className="h-5 w-5 text-gray-400" />
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-400 mb-2">
              {routingStats ? formatPercentage(routingStats.routing_accuracy) : '0%'}
            </div>
            <div className="text-sm text-gray-400">Routing Efficiency</div>
            <div className="text-xs text-gray-500 mt-1">
              {routingStats ? `${routingStats.avg_routing_time.toFixed(3)}s avg time` : 'No data'}
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400 mb-2">
              {contextStats ? formatPercentage(contextStats.memory_efficiency) : '0%'}
            </div>
            <div className="text-sm text-gray-400">Memory Efficiency</div>
            <div className="text-xs text-gray-500 mt-1">
              {contextStats ? `${formatPercentage(contextStats.cache_hit_rate)} cache hits` : 'No data'}
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-400 mb-2">
              {complexityStats ? (complexityStats.avg_complexity_score * 100).toFixed(0) + '%' : '0%'}
            </div>
            <div className="text-sm text-gray-400">Avg Complexity</div>
            <div className="text-xs text-gray-500 mt-1">
              {complexityStats ? `${formatNumber(complexityStats.total_analyses)} analyses` : 'No data'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}