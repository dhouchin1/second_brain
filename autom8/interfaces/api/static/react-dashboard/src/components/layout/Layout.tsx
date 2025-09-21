import React from 'react';
import { Outlet } from 'react-router-dom';
import { Header } from './Header';
import { Navigation } from './Navigation';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useSystemStatus } from '@/hooks/useApi';
import { cn } from '@/utils/cn';

interface LayoutProps {
  className?: string;
}

export function Layout({ className }: LayoutProps) {
  const { connectionStatus, isConnected } = useWebSocket();
  const { data: systemStatus, isLoading } = useSystemStatus();

  return (
    <div className={cn('min-h-screen bg-slate-25', className)}>
      {/* Background gradient - Lighter Dashboard v3 theme */}
      <div className="fixed inset-0 bg-gradient-to-br from-slate-50 via-slate-100 to-slate-200 -z-10" />
      
      {/* Header */}
      <Header
        connectionStatus={connectionStatus}
        lastUpdate={systemStatus?.last_updated}
        systemStatus={systemStatus ? {
          status: systemStatus.status,
          active_agents: systemStatus.active_agents,
          redis_connected: systemStatus.redis_connected,
        } : undefined}
      />

      <div className="flex">
        {/* Sidebar Navigation */}
        <Navigation />

        {/* Main Content Area */}
        <main className="flex-1 min-h-[calc(100vh-4rem)]">
          {/* Connection status overlay - Dashboard v3 theme */}
          {!isConnected && (
            <div className="bg-red-500/20 border-l-4 border-red-500 p-4 mb-4 mx-6 mt-6">
              <div className="flex">
                <div className="ml-3">
                  <p className="text-sm text-red-400">
                    {connectionStatus.metrics === 'connecting' || connectionStatus.events === 'connecting'
                      ? 'Connecting to real-time services...'
                      : 'Connection lost. Real-time data may not be available.'}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Page Content */}
          <div className="p-6">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}