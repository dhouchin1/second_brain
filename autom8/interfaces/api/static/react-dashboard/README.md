# Autom8 React Dashboard

A modern, real-time React dashboard for monitoring the Autom8 agent system. Built with TypeScript, React, and Tailwind CSS, featuring WebSocket integration for live updates and comprehensive visualization components.

## Features

### 🚀 Real-time Monitoring
- **Live WebSocket Connections**: Real-time metrics and event streaming
- **System Health**: Monitor overall system status, uptime, and resource usage
- **Agent Monitoring**: Track agent performance, status, and activity
- **Model Performance**: Monitor AI model usage, response times, and costs
- **Resource Optimization**: View complexity analysis, routing statistics, and context optimization

### 📊 Rich Visualizations
- **Interactive Charts**: Line charts, bar charts, and doughnut charts using Chart.js
- **Metric Cards**: Key performance indicators with trend analysis
- **Real-time Graphs**: Live updating charts for system metrics
- **Event Stream**: Filterable, real-time event feed with search capabilities

### 🎨 Modern UI/UX
- **Dark Theme**: Sleek dark mode design optimized for monitoring
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **Accessible**: Built with accessibility best practices
- **Smooth Animations**: Subtle animations and transitions for better UX

## Architecture

### Components Structure
```
src/
├── components/
│   ├── charts/          # Chart components (Line, Bar, Doughnut, MetricCard)
│   ├── dashboard/       # Dashboard views (Overview, Agents, Models, Resources, Events)
│   └── layout/          # Layout components (Header, Navigation, Layout)
├── hooks/               # Custom React hooks
├── services/            # API and WebSocket services
├── types/               # TypeScript type definitions
└── utils/               # Utility functions
```

### Key Technologies
- **React 18**: Modern React with hooks and concurrent features
- **TypeScript**: Full type safety and excellent developer experience
- **Tailwind CSS**: Utility-first CSS framework for rapid styling
- **Chart.js + react-chartjs-2**: Powerful charting library
- **React Router**: Client-side routing
- **Vite**: Fast build tool and development server

## Dashboard Views

### 1. Overview Dashboard
- System health status and uptime
- Active agents and event counts
- Real-time metrics with live charts
- Memory usage and performance summary
- Quick status indicators for all system components

### 2. Agents Dashboard
- List of all agents with status and performance metrics
- Agent performance scores and task completion rates
- Real-time agent events and activity feed
- Detailed agent information with recent decisions
- Performance charts and error tracking

### 3. Models Dashboard
- Available AI models and their performance metrics
- Request counts, success rates, and response times
- Cost tracking and usage analytics
- Provider distribution (local vs cloud models)
- Model availability and last usage information

### 4. Resources Dashboard
- Complexity analysis distribution and statistics
- Model routing efficiency and accuracy metrics
- Context optimization and memory usage
- Resource utilization trends and performance
- Optimization technique effectiveness

### 5. Events Dashboard
- Real-time event stream with live updates
- Advanced filtering by event type, priority, and source
- Event search and detailed information
- Pause/resume functionality for event monitoring
- Event statistics and agent activity tracking

## WebSocket Integration

### Real-time Data Streams
- **Metrics Stream** (`/ws/metrics`): System metrics updated every 5 seconds
- **Events Stream** (`/ws/events`): Real-time agent events and activities

### Connection Management
- Automatic reconnection with exponential backoff
- Connection status indicators
- Graceful handling of network interruptions
- Ping/pong keepalive mechanism

## API Integration

### REST Endpoints
All dashboard data is fetched from the FastAPI backend:
- `/api/system/status` - System health and metrics
- `/api/agents/list` - Agent information and status
- `/api/models/status` - Model performance data
- `/api/complexity/stats` - Complexity analysis statistics
- `/api/routing/stats` - Routing performance metrics
- `/api/context/stats` - Context optimization data

### Error Handling
- Automatic retry with exponential backoff
- User-friendly error messages
- Fallback UI states for offline/error scenarios
- Loading states and skeleton screens

## Development

### Prerequisites
- Node.js 18+ and npm/yarn
- Running Autom8 FastAPI server on port 8000

### Setup
```bash
# Navigate to the dashboard directory
cd autom8/interfaces/api/static/react-dashboard

# Install dependencies
npm install

# Start development server
npm run dev

# Open browser to http://localhost:3000
```

### Build for Production
```bash
# Build optimized production bundle
npm run build

# Preview production build
npm run preview
```

### Available Scripts
- `npm run dev` - Start development server with hot reload
- `npm run build` - Build optimized production bundle
- `npm run preview` - Preview production build locally
- `npm run type-check` - Run TypeScript type checking

## Configuration

### Environment Variables
The dashboard automatically connects to the FastAPI server. In development, the Vite proxy handles API and WebSocket connections to `http://localhost:8000`.

### Customization
- **Colors**: Modify the Tailwind color palette in `tailwind.config.js`
- **Charts**: Customize chart themes in `src/components/charts/chartConfig.ts`
- **Layout**: Adjust responsive breakpoints and spacing in the component files

## Performance Optimizations

### Chart Performance
- Optimized chart rendering with proper data structures
- Animation disabled for real-time charts to improve performance
- Efficient data updates using Chart.js update methods

### Memory Management
- Limited event history (last 1000 events)
- Automatic cleanup of expired cache entries
- Proper WebSocket subscription cleanup

### Bundle Optimization
- Code splitting with manual chunks for vendor libraries
- Tree shaking to eliminate unused code
- Optimized asset loading with Vite

## Browser Support

Supports all modern browsers:
- Chrome/Edge 88+
- Firefox 85+
- Safari 14+

## Future Enhancements

### Planned Features
- **Settings Page**: Dashboard configuration options
- **User Authentication**: Role-based access control
- **Export Capabilities**: Download charts and reports
- **Alert System**: Configurable alerts and notifications
- **Custom Dashboards**: User-defined dashboard layouts
- **Historical Data**: Time-range selection and historical analysis

### Technical Improvements
- **PWA Support**: Progressive Web App capabilities
- **Offline Mode**: Basic functionality when disconnected
- **Advanced Filtering**: More sophisticated data filtering options
- **Performance Monitoring**: Client-side performance tracking

## Troubleshooting

### Common Issues
1. **WebSocket Connection Failed**: Ensure the FastAPI server is running and accessible
2. **Charts Not Rendering**: Check browser console for Chart.js errors
3. **Missing Data**: Verify API endpoints are responding correctly
4. **Styling Issues**: Ensure Tailwind CSS is properly configured

### Debug Mode
Set environment variable `DEBUG=true` to enable detailed logging in development.

## Contributing

When contributing to the dashboard:
1. Follow the established TypeScript patterns
2. Maintain consistent component structure
3. Add proper type definitions for new features
4. Update this README for significant changes
5. Test across different screen sizes and browsers