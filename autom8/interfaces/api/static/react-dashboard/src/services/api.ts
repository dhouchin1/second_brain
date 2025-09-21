import {
  SystemStatus,
  AgentsList,
  ModelsStatus,
  ComplexityStats,
  RoutingStats,
  ContextStats,
  ApiError
} from '@/types/api';

export class AutomApiService {
  private baseUrl: string;

  constructor(baseUrl: string = '') {
    this.baseUrl = baseUrl;
  }

  /**
   * Generic API request handler with error handling
   */
  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    try {
      const response = await fetch(`${this.baseUrl}/api${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
        ...options,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new ApiError({
          message: errorData.detail || `HTTP ${response.status}: ${response.statusText}`,
          status: response.status,
          details: errorData
        });
      }

      return await response.json();
    } catch (error) {
      if (error instanceof ApiError) {
        throw error;
      }
      
      // Network or parsing errors
      throw new ApiError({
        message: error instanceof Error ? error.message : 'Unknown error occurred',
        details: error
      });
    }
  }

  /**
   * Get system health and status
   */
  public async getSystemStatus(): Promise<SystemStatus> {
    return this.request<SystemStatus>('/system/status');
  }

  /**
   * Get active agents and their status
   */
  public async getAgentsList(): Promise<AgentsList> {
    return this.request<AgentsList>('/agents/list');
  }

  /**
   * Get models status and performance
   */
  public async getModelsStatus(): Promise<ModelsStatus> {
    return this.request<ModelsStatus>('/models/status');
  }

  /**
   * Get complexity analysis statistics
   */
  public async getComplexityStats(): Promise<ComplexityStats> {
    return this.request<ComplexityStats>('/complexity/stats');
  }

  /**
   * Get routing statistics and performance
   */
  public async getRoutingStats(): Promise<RoutingStats> {
    return this.request<RoutingStats>('/routing/stats');
  }

  /**
   * Get context usage and optimization metrics
   */
  public async getContextStats(): Promise<ContextStats> {
    return this.request<ContextStats>('/context/stats');
  }

  /**
   * Get WebSocket connection statistics
   */
  public async getWebSocketStats(): Promise<any> {
    return this.request<any>('/websocket/stats');
  }

  /**
   * Health check endpoint
   */
  public async healthCheck(): Promise<any> {
    return this.request<any>('/health', { 
      headers: { 'Content-Type': 'application/json' } 
    });
  }

  /**
   * Get all dashboard data in a single request (for initial load)
   */
  public async getDashboardData(): Promise<{
    systemStatus: SystemStatus;
    agents: AgentsList;
    models: ModelsStatus;
    complexity: ComplexityStats;
    routing: RoutingStats;
    context: ContextStats;
  }> {
    const [systemStatus, agents, models, complexity, routing, context] = await Promise.all([
      this.getSystemStatus(),
      this.getAgentsList(),
      this.getModelsStatus(),
      this.getComplexityStats(),
      this.getRoutingStats(),
      this.getContextStats(),
    ]);

    return {
      systemStatus,
      agents,
      models,
      complexity,
      routing,
      context,
    };
  }
}

// Custom error class for API errors
class ApiError extends Error {
  public status?: number;
  public details?: any;

  constructor(params: { message: string; status?: number; details?: any }) {
    super(params.message);
    this.name = 'ApiError';
    this.status = params.status;
    this.details = params.details;
  }
}

export { ApiError };

// Singleton instance
export const apiService = new AutomApiService();