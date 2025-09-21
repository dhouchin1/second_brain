import React, { useState, useEffect } from 'react';
import {
  Users,
  User,
  Activity,
  CheckCircle,
  Clock,
  AlertTriangle,
  TrendingUp,
  Zap,
  MoreVertical
} from 'lucide-react';
import { MetricCard } from '@/components/charts/MetricCard';
import { BarChart } from '@/components/charts/BarChart';
import { LineChart } from '@/components/charts/LineChart';
import { useAgentsList } from '@/hooks/useApi';
import { useWebSocket } from '@/hooks/useWebSocket';
import { AgentInfo, AgentEvent } from '@/types/api';
import { formatRelativeTime, formatPercentage, getSuccessRateColor } from '@/utils/format';
import { chartColors, createDataset } from '@/components/charts/chartConfig';
import { cn } from '@/utils/cn';

export function AgentsDashboard() {
  const { data: agentsList, isLoading, refetch } = useAgentsList();
  const { subscribe, isConnected } = useWebSocket();
  
  const [recentEvents, setRecentEvents] = useState<AgentEvent[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);

  // Subscribe to agent events
  useEffect(() => {
    if (!isConnected) return;

    const unsubscribe = subscribe('events', (eventMessage: any) => {
      if (eventMessage.type === 'agent_event' && eventMessage.data) {
        const agentEvent = eventMessage.data as AgentEvent;
        setRecentEvents(prev => [agentEvent, ...prev.slice(0, 49)]); // Keep last 50 events
      }
    });

    return unsubscribe;
  }, [subscribe, isConnected]);

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'active': return 'text-green-400';
      case 'idle': return 'text-yellow-400';
      case 'error': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'active': return CheckCircle;
      case 'idle': return Clock;
      case 'error': return AlertTriangle;
      default: return User;
    }
  };

  // Prepare performance chart data
  const performanceChartData = agentsList?.agents ? {
    labels: agentsList.agents.map(agent => agent.name),
    datasets: [
      createDataset(
        'Performance Score',
        agentsList.agents.map(agent => agent.performance_score),
        0,
        'bar'
      ),
    ],
  } : { labels: [], datasets: [] };

  // Prepare tasks completed chart data
  const tasksChartData = agentsList?.agents ? {
    labels: agentsList.agents.map(agent => agent.name),
    datasets: [
      createDataset(
        'Tasks Completed',
        agentsList.agents.map(agent => agent.tasks_completed),
        2,
        'bar'
      ),
      createDataset(
        'Errors',
        agentsList.agents.map(agent => agent.error_count),
        4,
        'bar'
      ),
    ],
  } : { labels: [], datasets: [] };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="flex items-center space-x-3">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="text-gray-400">Loading agents data...</span>
        </div>
      </div>
    );
  }

  if (!agentsList) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <Users className="h-16 w-16 mx-auto text-gray-500 mb-4" />
          <h3 className="text-lg font-medium text-gray-200 mb-2">No Agent Data</h3>
          <p className="text-gray-400">Unable to load agent information</p>
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

  return (
    <div className="space-y-6">
      {/* Dashboard Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Agents Dashboard</h1>
          <p className="text-gray-400 mt-1">Monitor agent performance and activity</p>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={refetch}
            className="px-4 py-2 bg-gray-700 text-gray-200 rounded-lg hover:bg-gray-600 transition-colors"
          >
            Refresh
          </button>
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${
              isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'
            }`} />
            <span className="text-sm text-gray-400">
              {isConnected ? 'Live Updates' : 'Disconnected'}
            </span>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total Agents"
          value={agentsList.total_count}
          icon={Users}
          color="blue"
        />
        
        <MetricCard
          title="Active Agents"
          value={agentsList.active_count}
          subtitle={formatPercentage((agentsList.active_count / agentsList.total_count) * 100, 0)}
          icon={CheckCircle}
          color="green"
        />
        
        <MetricCard
          title="Idle Agents"
          value={agentsList.idle_count}
          subtitle={formatPercentage((agentsList.idle_count / agentsList.total_count) * 100, 0)}
          icon={Clock}
          color="yellow"
        />
        
        <MetricCard
          title="Avg Performance"
          value={formatPercentage(
            agentsList.agents.reduce((sum, agent) => sum + agent.performance_score, 0) / 
            agentsList.agents.length
          )}
          icon={TrendingUp}
          color="purple"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Agent Performance Chart */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-200 mb-4">Performance Scores</h3>
          <BarChart
            data={performanceChartData}
            height={300}
            options={{
              scales: {
                y: {
                  beginAtZero: true,
                  max: 1,
                  ticks: {
                    callback: (value) => formatPercentage(Number(value) * 100),
                  },
                },
              },
            }}
          />
        </div>

        {/* Tasks and Errors Chart */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-200 mb-4">Tasks & Errors</h3>
          <BarChart
            data={tasksChartData}
            height={300}
          />
        </div>
      </div>

      {/* Agent List and Events */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Agent List */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-200">Agent Status</h3>
            <span className="text-sm text-gray-400">
              {agentsList.agents.length} agents
            </span>
          </div>
          
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {agentsList.agents.map((agent) => {
              const StatusIcon = getStatusIcon(agent.status);
              return (
                <div
                  key={agent.id}
                  className={cn(
                    'p-4 rounded-lg border transition-all duration-200',
                    'bg-gray-700/30 border-gray-600 hover:bg-gray-700/50',
                    selectedAgent === agent.id && 'ring-2 ring-blue-500 bg-gray-700/50'
                  )}
                  onClick={() => setSelectedAgent(selectedAgent === agent.id ? null : agent.id)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-3">
                      <div className="flex-shrink-0">
                        <StatusIcon className={cn('h-5 w-5', getStatusColor(agent.status))} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2">
                          <h4 className="text-sm font-medium text-gray-200 truncate">
                            {agent.name}
                          </h4>
                          <span className={cn(
                            'px-2 py-1 rounded-full text-xs font-medium',
                            agent.status === 'active' && 'bg-green-500/20 text-green-400',
                            agent.status === 'idle' && 'bg-yellow-500/20 text-yellow-400',
                            agent.status === 'error' && 'bg-red-500/20 text-red-400'
                          )}>
                            {agent.status}
                          </span>
                        </div>
                        <p className="text-xs text-gray-400 mt-1">
                          ID: {agent.id}
                        </p>
                        {agent.current_work && (
                          <p className="text-xs text-gray-300 mt-1 truncate">
                            {agent.current_work}
                          </p>
                        )}
                        <div className="flex items-center space-x-4 mt-2 text-xs text-gray-400">
                          <span>Score: {formatPercentage(agent.performance_score * 100)}</span>
                          <span>Tasks: {agent.tasks_completed}</span>
                          {agent.error_count > 0 && (
                            <span className="text-red-400">Errors: {agent.error_count}</span>
                          )}
                        </div>
                        {agent.last_active && (
                          <p className="text-xs text-gray-500 mt-1">
                            Last active: {formatRelativeTime(agent.last_active)}
                          </p>
                        )}
                      </div>
                    </div>
                    <button className="text-gray-400 hover:text-gray-200">
                      <MoreVertical className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Recent Events */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-200">Recent Events</h3>
            <div className="flex items-center space-x-2">
              <Activity className="h-4 w-4 text-gray-400" />
              <span className="text-sm text-gray-400">
                {recentEvents.length} events
              </span>
            </div>
          </div>

          <div className="space-y-3 max-h-96 overflow-y-auto">
            {recentEvents.length === 0 ? (
              <div className="text-center py-8">
                <Activity className="h-8 w-8 mx-auto text-gray-500 mb-3" />
                <p className="text-gray-400 text-sm">
                  {isConnected ? 'Waiting for agent events...' : 'Connect to view real-time events'}
                </p>
              </div>
            ) : (
              recentEvents.map((event, index) => (
                <div
                  key={`${event.id}-${index}`}
                  className="p-3 rounded-lg bg-gray-700/30 border border-gray-600"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <span className={cn(
                          'px-2 py-1 rounded-full text-xs font-medium',
                          event.type === 'task_complete' && 'bg-green-500/20 text-green-400',
                          event.type === 'task_start' && 'bg-blue-500/20 text-blue-400',
                          event.type === 'error_occurred' && 'bg-red-500/20 text-red-400',
                          event.type === 'decision_made' && 'bg-purple-500/20 text-purple-400'
                        )}>
                          {event.type.replace('_', ' ')}
                        </span>
                        <span className={cn(
                          'px-1 py-0.5 rounded text-xs',
                          event.priority === 4 && 'bg-red-500/20 text-red-400',
                          event.priority === 3 && 'bg-yellow-500/20 text-yellow-400',
                          event.priority === 2 && 'bg-blue-500/20 text-blue-400',
                          event.priority === 1 && 'bg-gray-500/20 text-gray-400'
                        )}>
                          P{event.priority}
                        </span>
                      </div>
                      <p className="text-sm text-gray-200 mt-1">{event.summary}</p>
                      <p className="text-xs text-gray-400 mt-1">
                        Agent: {event.source_agent} • {formatRelativeTime(event.timestamp)}
                      </p>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}