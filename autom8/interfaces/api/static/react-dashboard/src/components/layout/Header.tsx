import React from 'react';
import { Wifi, WifiOff, Activity, AlertCircle } from 'lucide-react';
import { ConnectionStatus } from '@/services/websocket';
import { cn } from '@/utils/cn';
import { formatRelativeTime } from '@/utils/format';

interface HeaderProps {
  title?: string;
  connectionStatus: { metrics: ConnectionStatus; events: ConnectionStatus };
  lastUpdate?: string;
  systemStatus?: {
    status: string;
    active_agents: number;
    redis_connected: boolean;
  };
}

export function Header({ 
  title = 'Autom8 Dashboard', 
  connectionStatus, 
  lastUpdate,
  systemStatus
}: HeaderProps) {
  const isConnected = connectionStatus.metrics === 'connected' && connectionStatus.events === 'connected';
  const isConnecting = connectionStatus.metrics === 'connecting' || connectionStatus.events === 'connecting';

  const getConnectionIcon = () => {
    if (isConnecting) {
      return <Activity className="h-4 w-4 animate-pulse" />;
    }
    return isConnected ? <Wifi className="h-4 w-4" /> : <WifiOff className="h-4 w-4" />;
  };

  const getConnectionColor = () => {
    if (isConnecting) return 'text-yellow-400';
    return isConnected ? 'text-green-400' : 'text-red-400';
  };

  const getSystemStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'text-green-400';
      case 'degraded':
        return 'text-yellow-400';
      case 'error':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <header className="bg-slate-100/90 backdrop-blur-sm border-b border-slate-200 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Left side - Title and branding - Dashboard v3 theme */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-gradient-to-br from-discord-500 to-notion-500 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">A8</span>
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-800">{title}</h1>
              <p className="text-xs text-slate-500">Real-time Agent Monitoring</p>
            </div>
          </div>
        </div>

        {/* Right side - Status indicators */}
        <div className="flex items-center space-x-6">
          {/* System Status */}
          {systemStatus && (
            <div className="flex items-center space-x-2">
              <div className="flex items-center space-x-1">
                <div className={cn(
                  'w-2 h-2 rounded-full',
                  systemStatus.status === 'healthy' ? 'bg-green-400' : 
                  systemStatus.status === 'degraded' ? 'bg-yellow-400' : 'bg-red-400'
                )} />
                <span className={cn('text-sm font-medium', getSystemStatusColor(systemStatus.status))}>
                  {systemStatus.status.charAt(0).toUpperCase() + systemStatus.status.slice(1)}
                </span>
              </div>
              <span className="text-slate-400">•</span>
              <span className="text-sm text-slate-500">
                {systemStatus.active_agents} agents
              </span>
            </div>
          )}

          {/* Connection Status */}
          <div className="flex items-center space-x-2">
            <div className={cn('flex items-center space-x-1', getConnectionColor())}>
              {getConnectionIcon()}
              <span className="text-sm font-medium">
                {isConnecting ? 'Connecting...' : isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            
            {/* Connection Details - Dashboard v3 theme */}
            <div className="text-xs text-slate-500">
              <div className="flex items-center space-x-1">
                <span className={cn(
                  'w-1.5 h-1.5 rounded-full',
                  connectionStatus.metrics === 'connected' ? 'bg-notion-500' :
                  connectionStatus.metrics === 'connecting' ? 'bg-yellow-400' : 'bg-red-400'
                )} />
                <span>Metrics</span>
              </div>
            </div>
            <div className="text-xs text-slate-500">
              <div className="flex items-center space-x-1">
                <span className={cn(
                  'w-1.5 h-1.5 rounded-full',
                  connectionStatus.events === 'connected' ? 'bg-notion-500' :
                  connectionStatus.events === 'connecting' ? 'bg-yellow-400' : 'bg-red-400'
                )} />
                <span>Events</span>
              </div>
            </div>
          </div>

          {/* Last Update - Dashboard v3 theme */}
          {lastUpdate && (
            <div className="text-xs text-slate-500">
              <div>Last updated</div>
              <div className="text-slate-400">
                {formatRelativeTime(lastUpdate)}
              </div>
            </div>
          )}

          {/* Redis Status - Dashboard v3 theme */}
          {systemStatus && (
            <div className="flex items-center space-x-1">
              {systemStatus.redis_connected ? (
                <div className="flex items-center space-x-1 text-notion-500">
                  <div className="w-2 h-2 bg-notion-500 rounded-full" />
                  <span className="text-xs">Redis</span>
                </div>
              ) : (
                <div className="flex items-center space-x-1 text-red-400">
                  <AlertCircle className="h-3 w-3" />
                  <span className="text-xs">Redis Down</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </header>
  );
}