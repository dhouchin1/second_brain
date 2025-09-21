import { useState, useEffect, useCallback } from 'react';
import { apiService, ApiError } from '@/services/api';
import { LoadingState } from '@/types/api';

interface UseApiOptions {
  autoFetch?: boolean;
  refreshInterval?: number;
  retryCount?: number;
  retryDelay?: number;
}

interface UseApiState<T> {
  data: T | null;
  loading: LoadingState;
  error: ApiError | null;
}

export function useApi<T>(
  apiCall: () => Promise<T>,
  options: UseApiOptions = {}
) {
  const { 
    autoFetch = true, 
    refreshInterval,
    retryCount = 3,
    retryDelay = 1000
  } = options;

  const [state, setState] = useState<UseApiState<T>>({
    data: null,
    loading: 'idle',
    error: null,
  });

  const [retryAttempts, setRetryAttempts] = useState(0);

  const fetchData = useCallback(async (isRetry = false) => {
    if (!isRetry) {
      setState(prev => ({
        ...prev,
        loading: prev.data ? 'loading' : 'loading',
        error: null,
      }));
    }

    try {
      const result = await apiCall();
      setState({
        data: result,
        loading: 'success',
        error: null,
      });
      setRetryAttempts(0);
    } catch (error) {
      const apiError = error instanceof ApiError ? error : new ApiError({
        message: error instanceof Error ? error.message : 'Unknown error',
        details: error
      });

      if (retryAttempts < retryCount && !isRetry) {
        // Retry with exponential backoff
        const delay = retryDelay * Math.pow(2, retryAttempts);
        setTimeout(() => {
          setRetryAttempts(prev => prev + 1);
          fetchData(true);
        }, delay);
      } else {
        setState(prev => ({
          ...prev,
          loading: 'error',
          error: apiError,
        }));
        setRetryAttempts(0);
      }
    }
  }, [apiCall, retryAttempts, retryCount, retryDelay]);

  const refetch = useCallback(() => {
    setRetryAttempts(0);
    fetchData();
  }, [fetchData]);

  // Initial fetch
  useEffect(() => {
    if (autoFetch) {
      fetchData();
    }
  }, [autoFetch, fetchData]);

  // Refresh interval
  useEffect(() => {
    if (refreshInterval && refreshInterval > 0) {
      const interval = setInterval(() => {
        if (state.loading !== 'loading') {
          fetchData();
        }
      }, refreshInterval);

      return () => clearInterval(interval);
    }
  }, [refreshInterval, fetchData, state.loading]);

  return {
    ...state,
    refetch,
    isLoading: state.loading === 'loading',
    isError: state.loading === 'error',
    isSuccess: state.loading === 'success',
  };
}

// Specialized hooks for different endpoints
export function useSystemStatus(options?: UseApiOptions) {
  return useApi(() => apiService.getSystemStatus(), {
    autoFetch: true,
    refreshInterval: 5000, // Refresh every 5 seconds
    ...options
  });
}

export function useAgentsList(options?: UseApiOptions) {
  return useApi(() => apiService.getAgentsList(), {
    autoFetch: true,
    refreshInterval: 10000, // Refresh every 10 seconds
    ...options
  });
}

export function useModelsStatus(options?: UseApiOptions) {
  return useApi(() => apiService.getModelsStatus(), {
    autoFetch: true,
    refreshInterval: 15000, // Refresh every 15 seconds
    ...options
  });
}

export function useComplexityStats(options?: UseApiOptions) {
  return useApi(() => apiService.getComplexityStats(), {
    autoFetch: true,
    refreshInterval: 30000, // Refresh every 30 seconds
    ...options
  });
}

export function useRoutingStats(options?: UseApiOptions) {
  return useApi(() => apiService.getRoutingStats(), {
    autoFetch: true,
    refreshInterval: 30000, // Refresh every 30 seconds
    ...options
  });
}

export function useContextStats(options?: UseApiOptions) {
  return useApi(() => apiService.getContextStats(), {
    autoFetch: true,
    refreshInterval: 30000, // Refresh every 30 seconds
    ...options
  });
}

export function useDashboardData(options?: UseApiOptions) {
  return useApi(() => apiService.getDashboardData(), {
    autoFetch: true,
    refreshInterval: 60000, // Refresh every minute for bulk data
    ...options
  });
}