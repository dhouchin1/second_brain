import { useEffect, useState, useCallback, useRef } from 'react';
import { webSocketService, WebSocketEventType, ConnectionStatus } from '@/services/websocket';

interface UseWebSocketOptions {
  autoConnect?: boolean;
  reconnectOnMount?: boolean;
}

interface WebSocketState {
  isConnected: boolean;
  connectionStatus: { metrics: ConnectionStatus; events: ConnectionStatus };
  error: Error | null;
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const { autoConnect = true, reconnectOnMount = true } = options;
  
  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    connectionStatus: { metrics: 'disconnected', events: 'disconnected' },
    error: null,
  });

  const subscriptionsRef = useRef<Set<string>>(new Set());

  // Update connection status
  const updateConnectionStatus = useCallback(() => {
    const status = webSocketService.getConnectionStatus();
    const isConnected = status.metrics === 'connected' && status.events === 'connected';
    
    setState(prev => ({
      ...prev,
      isConnected,
      connectionStatus: status,
    }));
  }, []);

  // Connection status change handler
  const handleConnectionStatusChange = useCallback((type: WebSocketEventType, status: ConnectionStatus) => {
    updateConnectionStatus();
    
    if (status === 'error') {
      setState(prev => ({
        ...prev,
        error: new Error(`${type} WebSocket connection error`),
      }));
    } else if (status === 'connected') {
      setState(prev => ({
        ...prev,
        error: null,
      }));
    }
  }, [updateConnectionStatus]);

  // Connect function
  const connect = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, error: null }));
      await webSocketService.connect();
      updateConnectionStatus();
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error : new Error('Connection failed'),
      }));
    }
  }, [updateConnectionStatus]);

  // Disconnect function
  const disconnect = useCallback(() => {
    webSocketService.disconnect();
    updateConnectionStatus();
  }, [updateConnectionStatus]);

  // Subscribe to WebSocket messages
  const subscribe = useCallback((type: WebSocketEventType, callback: (data: any) => void) => {
    const subscriptionId = webSocketService.subscribe(type, callback);
    subscriptionsRef.current.add(subscriptionId);
    
    return () => {
      webSocketService.unsubscribe(subscriptionId);
      subscriptionsRef.current.delete(subscriptionId);
    };
  }, []);

  // Send ping to keep connection alive
  const ping = useCallback(() => {
    webSocketService.ping();
  }, []);

  // Setup and cleanup
  useEffect(() => {
    // Set up connection status callback
    webSocketService.setConnectionStatusCallback(handleConnectionStatusChange);

    // Auto-connect if enabled
    if (autoConnect) {
      connect();
    }

    // Set up ping interval to keep connections alive
    const pingInterval = setInterval(() => {
      if (state.isConnected) {
        ping();
      }
    }, 30000); // Ping every 30 seconds

    return () => {
      clearInterval(pingInterval);
      
      // Clean up all subscriptions
      subscriptionsRef.current.forEach(id => {
        webSocketService.unsubscribe(id);
      });
      subscriptionsRef.current.clear();
      
      // Don't disconnect on unmount to allow other components to use the same connection
    };
  }, [autoConnect, connect, ping, state.isConnected, handleConnectionStatusChange]);

  // Reconnect on mount if connection is lost and option is enabled
  useEffect(() => {
    if (reconnectOnMount && !state.isConnected) {
      const timer = setTimeout(() => {
        connect();
      }, 1000);
      
      return () => clearTimeout(timer);
    }
  }, [reconnectOnMount, state.isConnected, connect]);

  return {
    ...state,
    connect,
    disconnect,
    subscribe,
    ping,
  };
}