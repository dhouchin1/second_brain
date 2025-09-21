import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  Users, 
  Cpu, 
  Database, 
  Clock, 
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Zap
} from 'lucide-react';
import { MetricCard } from '@/components/charts/MetricCard';
import { LineChart } from '@/components/charts/LineChart';
import { DoughnutChart } from '@/components/charts/DoughnutChart';
import { useSystemStatus, useAgentsList, useModelsStatus } from '@/hooks/useApi';
import { useWebSocket } from '@/hooks/useWebSocket';
import { SystemStatus } from '@/types/api';
import { formatDuration, formatBytes, formatPercentage } from '@/utils/format';
import { chartColors, createDataset } from '@/components/charts/chartConfig';

export function OverviewDashboard() {
  const { data: systemStatus, isLoading: systemLoading } = useSystemStatus();
  const { data: agentsList, isLoading: agentsLoading } = useAgentsList();
  const { data: modelsStatus, isLoading: modelsLoading } = useModelsStatus();
  const { subscribe, isConnected } = useWebSocket();
  
  const [realtimeMetrics, setRealtimeMetrics] = useState<SystemStatus[]>([]);
  const [eventCounts, setEventCounts] = useState<Record<string, number>>({});

  // Subscribe to real-time metrics updates
  useEffect(() => {
    if (!isConnected) return;

    const unsubscribe = subscribe('metrics', (data: SystemStatus) => {
      setRealtimeMetrics(prev => {
        const updated = [...prev, data].slice(-50); // Keep last 50 data points
        return updated;
      });
      
      // Extract event counts if available
      if (data.memory_usage?.events) {
        setEventCounts(data.memory_usage.events);
      }
    });

    return unsubscribe;
  }, [subscribe, isConnected]);

  // System Health Status
  const getSystemHealthColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'green';
      case 'degraded': return 'yellow';
      case 'error': return 'red';
      default: return 'gray';
    }
  };

  // Prepare chart data for real-time metrics
  const metricsChartData = {
    labels: realtimeMetrics.map((_, index) => index.toString()),
    datasets: [
      createDataset(
        'Active Agents',
        realtimeMetrics.map(m => m.active_agents),
        0,
        'area'
      ),
      createDataset(
        'Total Events',
        realtimeMetrics.map(m => m.total_events),
        1,
        'line'
      ),
    ],
  };

  // Agent status distribution for doughnut chart
  const agentStatusData = agentsList ? {
    labels: ['Active', 'Idle'],
    datasets: [{
      data: [agentsList.active_count, agentsList.idle_count],
      backgroundColor: [chartColors[2], chartColors[3]], // green and yellow
      borderColor: [chartColors[2], chartColors[3]],
      borderWidth: 2,
    }],
  } : { labels: [], datasets: [] };

  // Model availability for doughnut chart
  const modelAvailabilityData = modelsStatus ? {
    labels: ['Available', 'Unavailable'],
    datasets: [{
      data: [
        modelsStatus.models.filter(m => m.available).length,
        modelsStatus.models.filter(m => !m.available).length
      ],
      backgroundColor: [chartColors[2], chartColors[4]], // green and red
      borderColor: [chartColors[2], chartColors[4]],
      borderWidth: 2,
    }],
  } : { labels: [], datasets: [] };

  const loading = systemLoading || agentsLoading || modelsLoading;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="flex items-center space-x-3">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="text-gray-400">Loading dashboard...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Dashboard Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">System Overview</h1>
          <p className="text-gray-400 mt-1">Real-time monitoring and system health</p>
        </div>
        <div className="flex items-center space-x-2">
          <div className={`flex items-center space-x-2 px-3 py-2 rounded-full ${
            isConnected ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
          }`}>
            <div className={`w-2 h-2 rounded-full ${
              isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'
            }`} />
            <span className="text-sm font-medium">
              {isConnected ? 'Live' : 'Disconnected'}
            </span>
          </div>
        </div>
      </div>

      {/* Key Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="System Status"
          value={systemStatus?.status || 'Unknown'}
          subtitle={`Uptime: ${formatDuration(systemStatus?.uptime || 0)}`}
          icon={systemStatus?.status === 'healthy' ? CheckCircle : AlertTriangle}
          color={getSystemHealthColor(systemStatus?.status || '')}
        />
        
        <MetricCard
          title="Active Agents"
          value={systemStatus?.active_agents || 0}
          subtitle={`${agentsList?.total_count || 0} total agents`}
          icon={Users}
          color="blue"
          trend={
            realtimeMetrics.length >= 2 ? {
              value: Math.abs(realtimeMetrics[realtimeMetrics.length - 1].active_agents - 
                              realtimeMetrics[realtimeMetrics.length - 2].active_agents),
              direction: realtimeMetrics[realtimeMetrics.length - 1].active_agents > 
                        realtimeMetrics[realtimeMetrics.length - 2].active_agents ? 'up' : 'down'
            } : undefined
          }
        />
        
        <MetricCard
          title="Total Events"
          value={systemStatus?.total_events || 0}
          subtitle="Events processed"
          icon={Activity}
          color="purple"
        />
        
        <MetricCard
          title="Available Models"
          value={modelsStatus?.models.filter(m => m.available).length || 0}
          subtitle={`${modelsStatus?.models.length || 0} total models`}
          icon={Cpu}
          color="green"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Real-time System Metrics */}
        <div className="bg-slate-100/80 backdrop-blur-sm rounded-lg border border-slate-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-slate-800">Real-time Metrics</h3>
            <div className="flex items-center space-x-2 text-sm text-slate-500">
              <Zap className="h-4 w-4" />
              <span>Live Data</span>
            </div>
          </div>
          {realtimeMetrics.length > 0 ? (
            <LineChart
              data={metricsChartData}
              height={300}
              realtime={true}
            />
          ) : (
            <div className="h-[300px] flex items-center justify-center text-slate-500">
              <div className="text-center">
                <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Waiting for real-time data...</p>
              </div>
            </div>
          )}
        </div>

        {/* Agent Status Distribution */}
        <div className="bg-slate-100/80 backdrop-blur-sm rounded-lg border border-slate-200 p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Agent Distribution</h3>
          {agentsList && agentsList.total_count > 0 ? (
            <DoughnutChart
              data={agentStatusData}
              height={300}
              centerText="Total Agents"
              centerValue={agentsList.total_count.toString()}
            />
          ) : (
            <div className="h-[300px] flex items-center justify-center text-slate-500">
              <div className="text-center">
                <Users className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No agent data available</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Additional Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Memory Usage */}
        <div className="bg-slate-100/80 backdrop-blur-sm rounded-lg border border-slate-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-slate-800">Memory Usage</h3>
            <Database className="h-5 w-5 text-slate-500" />
          </div>
          <div className="space-y-3">
            {systemStatus?.memory_usage && Object.entries(systemStatus.memory_usage).map(([key, value]) => (
              <div key={key} className="flex justify-between items-center">
                <span className="text-slate-500 capitalize">{key.replace('_', ' ')}</span>
                <span className="text-slate-800 font-medium">
                  {typeof value === 'number' ? formatBytes(value) : String(value)}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Model Performance Summary */}
        <div className="bg-slate-100/80 backdrop-blur-sm rounded-lg border border-slate-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-slate-800">Model Performance</h3>
            <TrendingUp className="h-5 w-5 text-slate-500" />
          </div>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-slate-500">Total Requests</span>
              <span className="text-slate-800 font-medium">{modelsStatus?.total_requests || 0}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-500">Total Cost</span>
              <span className="text-slate-800 font-medium">
                ${(modelsStatus?.total_cost || 0).toFixed(4)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-500">Avg Success Rate</span>
              <span className="text-notion-500 font-medium">
                {modelsStatus && modelsStatus.models.length > 0 
                  ? formatPercentage(
                      modelsStatus.models.reduce((sum, m) => sum + m.success_rate, 0) / 
                      modelsStatus.models.length
                    )
                  : '0%'}
              </span>
            </div>
          </div>
        </div>

        {/* Redis Connection Status */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-200">System Health</h3>
            <div className={`w-3 h-3 rounded-full ${
              systemStatus?.redis_connected ? 'bg-green-400' : 'bg-red-400'
            }`} />
          </div>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Redis</span>
              <span className={`font-medium ${
                systemStatus?.redis_connected ? 'text-green-400' : 'text-red-400'
              }`}>
                {systemStatus?.redis_connected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">WebSocket</span>
              <span className={`font-medium ${
                isConnected ? 'text-green-400' : 'text-red-400'
              }`}>
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Version</span>
              <span className="text-gray-200 font-medium">{systemStatus?.version || 'Unknown'}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}