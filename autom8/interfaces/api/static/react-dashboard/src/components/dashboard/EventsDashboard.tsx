import React, { useState, useEffect, useRef } from 'react';
import {
  Activity,
  Filter,
  Play,
  Pause,
  AlertTriangle,
  CheckCircle,
  Clock,
  Zap,
  User,
  ArrowRight,
  MoreVertical,
  Search
} from 'lucide-react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { AgentEvent, EventType, Priority } from '@/types/api';
import { formatRelativeTime, snakeToTitle } from '@/utils/format';
import { cn } from '@/utils/cn';

interface EventFilter {
  eventTypes: EventType[];
  priorities: Priority[];
  sourceAgents: string[];
  searchTerm: string;
}

export function EventsDashboard() {
  const { subscribe, isConnected } = useWebSocket();
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [filteredEvents, setFilteredEvents] = useState<AgentEvent[]>([]);
  const [isPaused, setIsPaused] = useState(false);
  const [selectedEvent, setSelectedEvent] = useState<string | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const eventsEndRef = useRef<HTMLDivElement>(null);

  const [filter, setFilter] = useState<EventFilter>({
    eventTypes: [],
    priorities: [],
    sourceAgents: [],
    searchTerm: '',
  });

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    eventsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Subscribe to events
  useEffect(() => {
    if (!isConnected) return;

    const unsubscribe = subscribe('events', (eventMessage: any) => {
      if (eventMessage.type === 'agent_event' && eventMessage.data && !isPaused) {
        const agentEvent = eventMessage.data as AgentEvent;
        setEvents(prev => {
          const updated = [agentEvent, ...prev.slice(0, 999)]; // Keep last 1000 events
          return updated;
        });
      }
    });

    return unsubscribe;
  }, [subscribe, isConnected, isPaused]);

  // Auto-scroll when new events arrive
  useEffect(() => {
    if (!isPaused) {
      scrollToBottom();
    }
  }, [events, isPaused]);

  // Filter events
  useEffect(() => {
    let filtered = events;

    // Filter by event types
    if (filter.eventTypes.length > 0) {
      filtered = filtered.filter(event => filter.eventTypes.includes(event.type));
    }

    // Filter by priorities
    if (filter.priorities.length > 0) {
      filtered = filtered.filter(event => filter.priorities.includes(event.priority));
    }

    // Filter by source agents
    if (filter.sourceAgents.length > 0) {
      filtered = filtered.filter(event => filter.sourceAgents.includes(event.source_agent));
    }

    // Filter by search term
    if (filter.searchTerm) {
      const searchTerm = filter.searchTerm.toLowerCase();
      filtered = filtered.filter(event =>
        event.summary.toLowerCase().includes(searchTerm) ||
        event.source_agent.toLowerCase().includes(searchTerm) ||
        event.type.toLowerCase().includes(searchTerm)
      );
    }

    setFilteredEvents(filtered);
  }, [events, filter]);

  const getEventTypeIcon = (type: EventType) => {
    switch (type) {
      case 'task_start': return Play;
      case 'task_complete': return CheckCircle;
      case 'decision_made': return Zap;
      case 'error_occurred': return AlertTriangle;
      case 'agent_ready': return User;
      case 'agent_shutdown': return User;
      default: return Activity;
    }
  };

  const getEventTypeColor = (type: EventType) => {
    switch (type) {
      case 'task_start': return 'text-blue-400';
      case 'task_complete': return 'text-green-400';
      case 'decision_made': return 'text-purple-400';
      case 'error_occurred': return 'text-red-400';
      case 'agent_ready': return 'text-cyan-400';
      case 'agent_shutdown': return 'text-gray-400';
      default: return 'text-gray-300';
    }
  };

  const getPriorityColor = (priority: Priority) => {
    switch (priority) {
      case 4: return 'text-red-400 bg-red-500/20';
      case 3: return 'text-yellow-400 bg-yellow-500/20';
      case 2: return 'text-blue-400 bg-blue-500/20';
      case 1: return 'text-gray-400 bg-gray-500/20';
      default: return 'text-gray-400 bg-gray-500/20';
    }
  };

  const getPriorityLabel = (priority: Priority) => {
    switch (priority) {
      case 4: return 'Critical';
      case 3: return 'High';
      case 2: return 'Medium';
      case 1: return 'Low';
      default: return 'Unknown';
    }
  };

  // Get unique values for filter dropdowns
  const uniqueEventTypes = [...new Set(events.map(e => e.type))];
  const uniqueSourceAgents = [...new Set(events.map(e => e.source_agent))];
  const uniquePriorities = [...new Set(events.map(e => e.priority))];

  const clearFilters = () => {
    setFilter({
      eventTypes: [],
      priorities: [],
      sourceAgents: [],
      searchTerm: '',
    });
  };

  const toggleEventType = (eventType: EventType) => {
    setFilter(prev => ({
      ...prev,
      eventTypes: prev.eventTypes.includes(eventType)
        ? prev.eventTypes.filter(t => t !== eventType)
        : [...prev.eventTypes, eventType],
    }));
  };

  const togglePriority = (priority: Priority) => {
    setFilter(prev => ({
      ...prev,
      priorities: prev.priorities.includes(priority)
        ? prev.priorities.filter(p => p !== priority)
        : [...prev.priorities, priority],
    }));
  };

  const toggleSourceAgent = (agent: string) => {
    setFilter(prev => ({
      ...prev,
      sourceAgents: prev.sourceAgents.includes(agent)
        ? prev.sourceAgents.filter(a => a !== agent)
        : [...prev.sourceAgents, agent],
    }));
  };

  return (
    <div className="space-y-6">
      {/* Dashboard Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Events Dashboard</h1>
          <p className="text-gray-400 mt-1">Real-time event stream and activity monitoring</p>
        </div>
        
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setIsPaused(!isPaused)}
            className={cn(
              'flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors',
              isPaused
                ? 'bg-green-600 hover:bg-green-700 text-white'
                : 'bg-yellow-600 hover:bg-yellow-700 text-white'
            )}
          >
            {isPaused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
            <span>{isPaused ? 'Resume' : 'Pause'}</span>
          </button>
          
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={cn(
              'flex items-center space-x-2 px-4 py-2 rounded-lg border transition-colors',
              showFilters
                ? 'bg-blue-600 text-white border-blue-500'
                : 'bg-gray-700 text-gray-200 border-gray-600 hover:bg-gray-600'
            )}
          >
            <Filter className="h-4 w-4" />
            <span>Filters</span>
            {(filter.eventTypes.length || filter.priorities.length || filter.sourceAgents.length || filter.searchTerm) > 0 && (
              <span className="bg-blue-500 text-white text-xs rounded-full px-2 py-1">
                {filter.eventTypes.length + filter.priorities.length + filter.sourceAgents.length + (filter.searchTerm ? 1 : 0)}
              </span>
            )}
          </button>
          
          <div className="flex items-center space-x-2">
            <div className={cn(
              'w-2 h-2 rounded-full',
              isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'
            )} />
            <span className="text-sm text-gray-400">
              {isConnected ? 'Live' : 'Disconnected'}
            </span>
          </div>
        </div>
      </div>

      {/* Connection Status */}
      {!isConnected && (
        <div className="bg-red-600/20 border border-red-600/30 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="h-5 w-5 text-red-400" />
            <div>
              <p className="text-red-300 font-medium">WebSocket Disconnected</p>
              <p className="text-red-400 text-sm">Real-time events are not available. Connect to receive live updates.</p>
            </div>
          </div>
        </div>
      )}

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-800/50 rounded-lg border border-gray-700 p-4">
          <div className="flex items-center space-x-2">
            <Activity className="h-5 w-5 text-blue-400" />
            <span className="text-sm text-gray-400">Total Events</span>
          </div>
          <div className="text-2xl font-bold text-white mt-1">{events.length}</div>
        </div>
        
        <div className="bg-gray-800/50 rounded-lg border border-gray-700 p-4">
          <div className="flex items-center space-x-2">
            <Filter className="h-5 w-5 text-purple-400" />
            <span className="text-sm text-gray-400">Filtered</span>
          </div>
          <div className="text-2xl font-bold text-white mt-1">{filteredEvents.length}</div>
        </div>
        
        <div className="bg-gray-800/50 rounded-lg border border-gray-700 p-4">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="h-5 w-5 text-red-400" />
            <span className="text-sm text-gray-400">Critical</span>
          </div>
          <div className="text-2xl font-bold text-white mt-1">
            {events.filter(e => e.priority === 4).length}
          </div>
        </div>
        
        <div className="bg-gray-800/50 rounded-lg border border-gray-700 p-4">
          <div className="flex items-center space-x-2">
            <User className="h-5 w-5 text-cyan-400" />
            <span className="text-sm text-gray-400">Active Agents</span>
          </div>
          <div className="text-2xl font-bold text-white mt-1">
            {uniqueSourceAgents.length}
          </div>
        </div>
      </div>

      {/* Filters */}
      {showFilters && (
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium text-gray-200">Event Filters</h3>
            <button
              onClick={clearFilters}
              className="text-sm text-blue-400 hover:text-blue-300"
            >
              Clear All
            </button>
          </div>

          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search events..."
              value={filter.searchTerm}
              onChange={(e) => setFilter(prev => ({ ...prev, searchTerm: e.target.value }))}
              className="w-full pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-gray-200 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Event Types */}
            <div>
              <h4 className="text-sm font-medium text-gray-300 mb-2">Event Types</h4>
              <div className="space-y-1">
                {uniqueEventTypes.map(eventType => (
                  <label key={eventType} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={filter.eventTypes.includes(eventType)}
                      onChange={() => toggleEventType(eventType)}
                      className="rounded border-gray-600 bg-gray-700 text-blue-600"
                    />
                    <span className="text-sm text-gray-300">{snakeToTitle(eventType)}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Priorities */}
            <div>
              <h4 className="text-sm font-medium text-gray-300 mb-2">Priorities</h4>
              <div className="space-y-1">
                {uniquePriorities.map(priority => (
                  <label key={priority} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={filter.priorities.includes(priority)}
                      onChange={() => togglePriority(priority)}
                      className="rounded border-gray-600 bg-gray-700 text-blue-600"
                    />
                    <span className="text-sm text-gray-300">{getPriorityLabel(priority)}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Source Agents */}
            <div>
              <h4 className="text-sm font-medium text-gray-300 mb-2">Source Agents</h4>
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {uniqueSourceAgents.map(agent => (
                  <label key={agent} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={filter.sourceAgents.includes(agent)}
                      onChange={() => toggleSourceAgent(agent)}
                      className="rounded border-gray-600 bg-gray-700 text-blue-600"
                    />
                    <span className="text-sm text-gray-300 truncate">{agent}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Events Stream */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700">
        <div className="p-6 border-b border-gray-700">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-200">Event Stream</h3>
            <div className="text-sm text-gray-400">
              {isPaused && <span className="text-yellow-400">Paused • </span>}
              Showing {filteredEvents.length} of {events.length} events
            </div>
          </div>
        </div>

        <div className="max-h-96 overflow-y-auto">
          {filteredEvents.length === 0 ? (
            <div className="p-12 text-center">
              <Activity className="h-12 w-12 mx-auto text-gray-500 mb-4" />
              <h3 className="text-lg font-medium text-gray-300 mb-2">No Events</h3>
              <p className="text-gray-400">
                {events.length === 0
                  ? isConnected
                    ? 'Waiting for events...'
                    : 'Connect to receive real-time events'
                  : 'No events match your current filters'}
              </p>
            </div>
          ) : (
            <div className="divide-y divide-gray-700">
              {filteredEvents.map((event, index) => {
                const EventIcon = getEventTypeIcon(event.type);
                return (
                  <div
                    key={`${event.id}-${index}`}
                    className={cn(
                      'p-4 hover:bg-gray-700/30 transition-colors cursor-pointer',
                      selectedEvent === event.id && 'bg-gray-700/50'
                    )}
                    onClick={() => setSelectedEvent(selectedEvent === event.id ? null : event.id)}
                  >
                    <div className="flex items-start space-x-4">
                      {/* Event Icon */}
                      <div className={cn(
                        'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center mt-0.5',
                        'bg-gray-700'
                      )}>
                        <EventIcon className={cn('h-4 w-4', getEventTypeColor(event.type))} />
                      </div>

                      {/* Event Content */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center space-x-2 mb-1">
                              <span className={cn(
                                'px-2 py-1 rounded-full text-xs font-medium',
                                getEventTypeColor(event.type),
                                'bg-gray-600/30'
                              )}>
                                {snakeToTitle(event.type)}
                              </span>
                              <span className={cn(
                                'px-2 py-1 rounded-full text-xs font-medium',
                                getPriorityColor(event.priority)
                              )}>
                                {getPriorityLabel(event.priority)}
                              </span>
                              <span className="text-xs text-gray-500">
                                {formatRelativeTime(event.timestamp)}
                              </span>
                            </div>
                            
                            <p className="text-sm text-gray-200 mb-2">{event.summary}</p>
                            
                            <div className="flex items-center space-x-3 text-xs text-gray-400">
                              <div className="flex items-center space-x-1">
                                <User className="h-3 w-3" />
                                <span>{event.source_agent}</span>
                              </div>
                              {event.target_agent && (
                                <>
                                  <ArrowRight className="h-3 w-3" />
                                  <span>{event.target_agent}</span>
                                </>
                              )}
                              {event.data_size && (
                                <span>{event.data_size} bytes</span>
                              )}
                            </div>
                          </div>

                          <button className="text-gray-400 hover:text-gray-200 ml-2">
                            <MoreVertical className="h-4 w-4" />
                          </button>
                        </div>

                        {/* Expanded Details */}
                        {selectedEvent === event.id && (
                          <div className="mt-3 pt-3 border-t border-gray-600">
                            <div className="grid grid-cols-2 gap-4 text-sm">
                              <div>
                                <span className="text-gray-400">Event ID:</span>
                                <div className="text-gray-300 font-mono text-xs break-all">
                                  {event.id}
                                </div>
                              </div>
                              <div>
                                <span className="text-gray-400">Timestamp:</span>
                                <div className="text-gray-300">
                                  {new Date(event.timestamp).toLocaleString()}
                                </div>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
          <div ref={eventsEndRef} />
        </div>
      </div>
    </div>
  );
}