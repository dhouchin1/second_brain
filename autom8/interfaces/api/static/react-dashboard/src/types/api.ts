// API Response Types for Autom8 Dashboard

export interface SystemStatus {
  status: string;
  uptime: number;
  version: string;
  redis_connected: boolean;
  active_agents: number;
  total_events: number;
  memory_usage: Record<string, any>;
  last_updated: string;
}

export interface AgentInfo {
  id: string;
  name: string;
  status: string;
  last_active: string | null;
  current_work: string | null;
  performance_score: number;
  tasks_completed: number;
  error_count: number;
}

export interface AgentsList {
  agents: AgentInfo[];
  total_count: number;
  active_count: number;
  idle_count: number;
}

export interface ModelInfo {
  name: string;
  provider: string;
  available: boolean;
  requests_count: number;
  avg_response_time: number;
  success_rate: number;
  cost_per_request: number;
  last_used: string | null;
}

export interface ModelsStatus {
  models: ModelInfo[];
  total_requests: number;
  total_cost: number;
  preferred_model: string | null;
}

export interface ComplexityStats {
  total_analyses: number;
  avg_complexity_score: number;
  distribution: Record<string, number>;
  trend_data: Array<Record<string, any>>;
  last_analysis: string | null;
}

export interface RoutingStats {
  total_routes: number;
  local_routes: number;
  cloud_routes: number;
  avg_routing_time: number;
  routing_accuracy: number;
  model_distribution: Record<string, number>;
}

export interface ContextStats {
  total_contexts: number;
  avg_context_size: number;
  optimization_savings: number;
  cache_hit_rate: number;
  memory_efficiency: number;
}

// WebSocket Message Types
export interface EventMessage {
  type: string;
  data: Record<string, any>;
  timestamp: string;
  source?: string;
}

export interface MetricsMessage {
  type: 'metrics';
  system_status: SystemStatus;
  timestamp: string;
}

// Chart Data Types
export interface ChartDataPoint {
  x: string | number;
  y: number;
  label?: string;
}

export interface TimeSeriesData {
  timestamp: string;
  [key: string]: string | number;
}

// Dashboard State Types
export interface DashboardState {
  isConnected: boolean;
  lastUpdate: string;
  connectionStatus: 'connected' | 'connecting' | 'disconnected' | 'error';
  metricsSocket: WebSocket | null;
  eventsSocket: WebSocket | null;
}

export type EventType = 
  | 'task_start'
  | 'task_complete' 
  | 'decision_made'
  | 'context_updated'
  | 'error_occurred'
  | 'agent_ready'
  | 'agent_shutdown'
  | 'coordination_request';

export type Priority = 1 | 2 | 3 | 4; // LOW | MEDIUM | HIGH | CRITICAL

export interface AgentEvent {
  id: string;
  type: EventType;
  source_agent: string;
  target_agent?: string;
  summary: string;
  priority: Priority;
  timestamp: string;
  data_size?: number;
}

// Navigation and Layout Types
export type DashboardView = 'overview' | 'agents' | 'models' | 'resources' | 'events';

export interface NavigationItem {
  id: DashboardView;
  label: string;
  icon: string;
  path: string;
}

// Utility Types
export type LoadingState = 'idle' | 'loading' | 'success' | 'error';

export interface ApiError {
  message: string;
  status?: number;
  details?: any;
}