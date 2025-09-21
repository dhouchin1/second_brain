import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Layout } from '@/components/layout/Layout';
import { OverviewDashboard } from '@/components/dashboard/OverviewDashboard';
import { AgentsDashboard } from '@/components/dashboard/AgentsDashboard';
import { ModelsDashboard } from '@/components/dashboard/ModelsDashboard';
import { ResourcesDashboard } from '@/components/dashboard/ResourcesDashboard';
import { EventsDashboard } from '@/components/dashboard/EventsDashboard';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<OverviewDashboard />} />
            <Route path="agents" element={<AgentsDashboard />} />
            <Route path="models" element={<ModelsDashboard />} />
            <Route path="resources" element={<ResourcesDashboard />} />
            <Route path="events" element={<EventsDashboard />} />
            {/* Settings placeholder */}
            <Route path="settings" element={
              <div className="flex items-center justify-center h-96">
                <div className="text-center">
                  <div className="text-6xl mb-4">⚙️</div>
                  <h2 className="text-2xl font-bold text-slate-800 mb-2">Settings</h2>
                  <p className="text-slate-500">Dashboard configuration options coming soon</p>
                </div>
              </div>
            } />
            {/* 404 fallback */}
            <Route path="*" element={
              <div className="flex items-center justify-center h-96">
                <div className="text-center">
                  <div className="text-6xl mb-4">🔍</div>
                  <h2 className="text-2xl font-bold text-slate-800 mb-2">Page Not Found</h2>
                  <p className="text-slate-500">The page you're looking for doesn't exist</p>
                </div>
              </div>
            } />
          </Route>
        </Routes>
      </div>
    </Router>
  );
}

export default App;