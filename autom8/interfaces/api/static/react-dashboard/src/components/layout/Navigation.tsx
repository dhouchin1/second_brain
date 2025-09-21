import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  LayoutGrid, 
  Users, 
  Cpu, 
  HardDrive, 
  Activity, 
  Settings,
  Zap
} from 'lucide-react';
import { DashboardView } from '@/types/api';
import { cn } from '@/utils/cn';

interface NavigationProps {
  className?: string;
}

interface NavigationItem {
  id: DashboardView;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  path: string;
  description?: string;
}

const navigationItems: NavigationItem[] = [
  {
    id: 'overview',
    label: 'Overview',
    icon: LayoutGrid,
    path: '/',
    description: 'System overview and key metrics',
  },
  {
    id: 'agents',
    label: 'Agents',
    icon: Users,
    path: '/agents',
    description: 'Agent status and performance',
  },
  {
    id: 'models',
    label: 'Models',
    icon: Cpu,
    path: '/models',
    description: 'Model performance and usage',
  },
  {
    id: 'resources',
    label: 'Resources',
    icon: HardDrive,
    path: '/resources',
    description: 'Resource utilization and optimization',
  },
  {
    id: 'events',
    label: 'Events',
    icon: Activity,
    path: '/events',
    description: 'Real-time event stream',
  },
];

export function Navigation({ className }: NavigationProps) {
  return (
    <nav className={cn('w-64 bg-slate-100/90 backdrop-blur-sm border-r border-slate-200', className)}>
      <div className="flex flex-col h-full">
        {/* Navigation Header - Dashboard v3 theme */}
        <div className="p-6 border-b border-slate-200">
          <div className="flex items-center space-x-2">
            <Zap className="h-5 w-5 text-discord-500" />
            <span className="font-semibold text-slate-700">Navigation</span>
          </div>
        </div>

        {/* Navigation Items */}
        <div className="flex-1 py-6">
          <ul className="space-y-2 px-4">
            {navigationItems.map((item) => {
              const Icon = item.icon;
              return (
                <li key={item.id}>
                  <NavLink
                    to={item.path}
                    className={({ isActive }) =>
                      cn(
                        'flex items-center space-x-3 px-3 py-2.5 rounded-lg transition-all duration-200',
                        'hover:bg-slate-200/50 group',
                        isActive
                          ? 'bg-discord-500/20 text-discord-400 border-l-2 border-discord-400'
                          : 'text-slate-400 hover:text-slate-600'
                      )
                    }
                  >
                    {({ isActive }) => (
                      <>
                        <Icon className={cn(
                          'h-5 w-5 transition-colors duration-200',
                          isActive ? 'text-discord-400' : 'text-slate-450 group-hover:text-slate-300'
                        )} />
                        <div className="flex-1">
                          <div className={cn(
                            'font-medium transition-colors duration-200',
                            isActive ? 'text-discord-400' : 'text-slate-600'
                          )}>
                            {item.label}
                          </div>
                          {item.description && (
                            <div className="text-xs text-slate-500 mt-0.5">
                              {item.description}
                            </div>
                          )}
                        </div>
                      </>
                    )}
                  </NavLink>
                </li>
              );
            })}
          </ul>
        </div>

        {/* Navigation Footer - Dashboard v3 theme */}
        <div className="p-4 border-t border-slate-200">
          <NavLink
            to="/settings"
            className={({ isActive }) =>
              cn(
                'flex items-center space-x-3 px-3 py-2.5 rounded-lg transition-all duration-200',
                'hover:bg-slate-200/50 group',
                isActive
                  ? 'bg-discord-500/20 text-discord-400'
                  : 'text-slate-400 hover:text-slate-600'
              )
            }
          >
            <Settings className="h-5 w-5" />
            <span className="font-medium">Settings</span>
          </NavLink>
        </div>
      </div>
    </nav>
  );
}