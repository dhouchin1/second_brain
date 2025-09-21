import { EventMessage, MetricsMessage, AgentEvent, SystemStatus } from '@/types/api';

export type WebSocketEventType = 'metrics' | 'events';
export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface WebSocketSubscription {
  id: string;
  type: WebSocketEventType;
  callback: (data: any) => void;
}

export class AutomWebSocketService {
  private metricsSocket: WebSocket | null = null;
  private eventsSocket: WebSocket | null = null;
  private subscriptions: Map<string, WebSocketSubscription> = new Map();
  private reconnectAttempts: Map<WebSocketEventType, number> = new Map();
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private isReconnecting: Map<WebSocketEventType, boolean> = new Map();
  
  // Connection status callbacks
  private onConnectionStatusChange?: (type: WebSocketEventType, status: ConnectionStatus) => void;

  constructor() {
    this.reconnectAttempts.set('metrics', 0);
    this.reconnectAttempts.set('events', 0);
    this.isReconnecting.set('metrics', false);
    this.isReconnecting.set('events', false);
  }

  /**
   * Set callback for connection status changes
   */
  public setConnectionStatusCallback(callback: (type: WebSocketEventType, status: ConnectionStatus) => void): void {
    this.onConnectionStatusChange = callback;
  }

  /**
   * Connect to WebSocket endpoints
   */
  public async connect(): Promise<void> {
    await Promise.all([
      this.connectMetrics(),
      this.connectEvents()
    ]);
  }

  /**
   * Connect to metrics WebSocket
   */
  private async connectMetrics(): Promise<void> {
    if (this.metricsSocket?.readyState === WebSocket.OPEN) {
      return;
    }

    return new Promise((resolve, reject) => {
      try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/metrics`;
        
        this.notifyConnectionStatus('metrics', 'connecting');
        this.metricsSocket = new WebSocket(wsUrl);

        this.metricsSocket.onopen = () => {
          console.log('Metrics WebSocket connected');
          this.reconnectAttempts.set('metrics', 0);
          this.isReconnecting.set('metrics', false);
          this.notifyConnectionStatus('metrics', 'connected');
          resolve();
        };

        this.metricsSocket.onmessage = (event) => {
          try {
            const message: MetricsMessage = JSON.parse(event.data);
            this.handleMetricsMessage(message);
          } catch (error) {
            console.error('Error parsing metrics message:', error);
          }
        };

        this.metricsSocket.onerror = (error) => {
          console.error('Metrics WebSocket error:', error);
          this.notifyConnectionStatus('metrics', 'error');
        };

        this.metricsSocket.onclose = (event) => {
          console.log('Metrics WebSocket closed:', event.code, event.reason);
          this.notifyConnectionStatus('metrics', 'disconnected');
          this.scheduleReconnect('metrics');
        };

        // Timeout for connection
        setTimeout(() => {
          if (this.metricsSocket?.readyState === WebSocket.CONNECTING) {
            this.metricsSocket?.close();
            reject(new Error('Metrics WebSocket connection timeout'));
          }
        }, 5000);

      } catch (error) {
        console.error('Failed to connect metrics WebSocket:', error);
        this.notifyConnectionStatus('metrics', 'error');
        reject(error);
      }
    });
  }

  /**
   * Connect to events WebSocket
   */
  private async connectEvents(): Promise<void> {
    if (this.eventsSocket?.readyState === WebSocket.OPEN) {
      return;
    }

    return new Promise((resolve, reject) => {
      try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/events`;
        
        this.notifyConnectionStatus('events', 'connecting');
        this.eventsSocket = new WebSocket(wsUrl);

        this.eventsSocket.onopen = () => {
          console.log('Events WebSocket connected');
          this.reconnectAttempts.set('events', 0);
          this.isReconnecting.set('events', false);
          this.notifyConnectionStatus('events', 'connected');
          resolve();
        };

        this.eventsSocket.onmessage = (event) => {
          try {
            const message: EventMessage = JSON.parse(event.data);
            this.handleEventMessage(message);
          } catch (error) {
            console.error('Error parsing event message:', error);
          }
        };

        this.eventsSocket.onerror = (error) => {
          console.error('Events WebSocket error:', error);
          this.notifyConnectionStatus('events', 'error');
        };

        this.eventsSocket.onclose = (event) => {
          console.log('Events WebSocket closed:', event.code, event.reason);
          this.notifyConnectionStatus('events', 'disconnected');
          this.scheduleReconnect('events');
        };

        // Timeout for connection
        setTimeout(() => {
          if (this.eventsSocket?.readyState === WebSocket.CONNECTING) {
            this.eventsSocket?.close();
            reject(new Error('Events WebSocket connection timeout'));
          }
        }, 5000);

      } catch (error) {
        console.error('Failed to connect events WebSocket:', error);
        this.notifyConnectionStatus('events', 'error');
        reject(error);
      }
    });
  }

  /**
   * Schedule reconnection attempt
   */
  private scheduleReconnect(type: WebSocketEventType): void {
    const attempts = this.reconnectAttempts.get(type) || 0;
    
    if (attempts >= this.maxReconnectAttempts || this.isReconnecting.get(type)) {
      return;
    }

    this.isReconnecting.set(type, true);
    const delay = this.reconnectDelay * Math.pow(2, attempts);
    
    setTimeout(() => {
      this.reconnectAttempts.set(type, attempts + 1);
      
      if (type === 'metrics') {
        this.connectMetrics().catch(console.error);
      } else if (type === 'events') {
        this.connectEvents().catch(console.error);
      }
    }, delay);
  }

  /**
   * Handle incoming metrics message
   */
  private handleMetricsMessage(message: MetricsMessage): void {
    // Notify all metrics subscribers
    this.subscriptions.forEach((subscription) => {
      if (subscription.type === 'metrics') {
        subscription.callback(message.system_status);
      }
    });
  }

  /**
   * Handle incoming event message
   */
  private handleEventMessage(message: EventMessage): void {
    // Convert to AgentEvent if it's an agent_event type
    let eventData = message.data;
    if (message.type === 'agent_event') {
      eventData = message.data as AgentEvent;
    }

    // Notify all event subscribers
    this.subscriptions.forEach((subscription) => {
      if (subscription.type === 'events') {
        subscription.callback({
          ...message,
          data: eventData
        });
      }
    });
  }

  /**
   * Subscribe to WebSocket messages
   */
  public subscribe(type: WebSocketEventType, callback: (data: any) => void): string {
    const id = `${type}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    this.subscriptions.set(id, { id, type, callback });
    return id;
  }

  /**
   * Unsubscribe from WebSocket messages
   */
  public unsubscribe(subscriptionId: string): void {
    this.subscriptions.delete(subscriptionId);
  }

  /**
   * Get connection status
   */
  public getConnectionStatus(): { metrics: ConnectionStatus; events: ConnectionStatus } {
    const getStatus = (socket: WebSocket | null): ConnectionStatus => {
      if (!socket) return 'disconnected';
      switch (socket.readyState) {
        case WebSocket.CONNECTING:
          return 'connecting';
        case WebSocket.OPEN:
          return 'connected';
        case WebSocket.CLOSING:
        case WebSocket.CLOSED:
        default:
          return 'disconnected';
      }
    };

    return {
      metrics: getStatus(this.metricsSocket),
      events: getStatus(this.eventsSocket)
    };
  }

  /**
   * Send keepalive ping
   */
  public ping(): void {
    if (this.metricsSocket?.readyState === WebSocket.OPEN) {
      this.metricsSocket.send('ping');
    }
    if (this.eventsSocket?.readyState === WebSocket.OPEN) {
      this.eventsSocket.send('ping');
    }
  }

  /**
   * Disconnect all WebSocket connections
   */
  public disconnect(): void {
    if (this.metricsSocket) {
      this.metricsSocket.close();
      this.metricsSocket = null;
    }
    
    if (this.eventsSocket) {
      this.eventsSocket.close();
      this.eventsSocket = null;
    }
    
    this.subscriptions.clear();
  }

  /**
   * Notify connection status change
   */
  private notifyConnectionStatus(type: WebSocketEventType, status: ConnectionStatus): void {
    if (this.onConnectionStatusChange) {
      this.onConnectionStatusChange(type, status);
    }
  }
}

// Singleton instance
export const webSocketService = new AutomWebSocketService();