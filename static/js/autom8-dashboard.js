/**
 * Autom8 AI Operations Dashboard Widget
 *
 * Provides real-time monitoring of AI model usage, costs, and performance
 * for the Autom8 integration within Second Brain.
 */

class Autom8Dashboard {
    constructor() {
        this.initialized = false;
        this.updateInterval = null;
        this.lastUpdate = null;
        this.data = {
            status: null,
            stats: null,
            models: [],
            costs: null
        };
        this.chartInstances = {};
    }

    async init() {
        if (this.initialized) return;

        try {
            this.data = await this.loadDashboardData();
            this.renderDashboard();
            this.startPeriodicUpdates();
            this.initialized = true;
            console.log('Autom8 Dashboard initialized');
        } catch (error) {
            console.error('Failed to initialize Autom8 Dashboard:', error);
            this.renderError('Failed to initialize AI operations dashboard');
        }
    }

    async loadDashboardData() {
        try {
            const response = await fetch('/api/autom8/dashboard-data');

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            this.data = data;
            this.lastUpdate = new Date();
            return data;
        } catch (error) {
            console.error('Failed to load Autom8 data:', error);
            // Return fallback data instead of throwing
            return {
                status: {
                    enabled: true,
                    available: false,
                    api_url: "http://localhost:8000",
                    fallback_enabled: true
                },
                stats: {
                    total_requests: 0,
                    autom8_requests: 0,
                    ollama_fallbacks: 0,
                    autom8_usage_rate: 0.0,
                    cost_savings_estimate: 0.0
                },
                models: [],
                costs: {
                    total_cost: 0.0,
                    requests_count: 0,
                    projected_monthly_cost: 0.0
                }
            };
        }
    }

    renderDashboard() {
        const container = document.getElementById('autom8-dashboard');
        if (!container) {
            console.warn('Autom8 dashboard container not found');
            return;
        }

        container.innerHTML = this.buildDashboardHTML();
        this.attachEventListeners();
    }

    buildDashboardHTML() {
        const { status, stats, models, costs, system_status, performance, cost_analytics, ai_operations } = this.data;

        return `
            <div class="autom8-dashboard bg-slate-800 rounded-lg p-6 space-y-6">
                <!-- Header -->
                <div class="flex items-center justify-between">
                    <h3 class="text-lg font-semibold text-white flex items-center">
                        <svg class="w-6 h-6 mr-2 text-discord-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                  d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
                        </svg>
                        AI Operations Dashboard
                    </h3>
                    <div class="flex items-center space-x-3">
                        ${this.buildStatusIndicator(status, system_status)}
                        <div class="flex space-x-2">
                            <button onclick="autom8Dashboard.triggerHealthCheck()"
                                    class="text-slate-400 hover:text-green-400 transition-colors"
                                    title="Test Connection">
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                          d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                                </svg>
                            </button>
                            <button onclick="autom8Dashboard.refresh()"
                                    class="text-slate-400 hover:text-white transition-colors"
                                    title="Refresh Data">
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                          d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>

                <!-- System Status Overview -->
                ${this.buildSystemStatusOverview(system_status, status)}

                <!-- Quick Stats -->
                ${this.buildQuickStats(stats, costs)}

                <!-- Performance & SLO Monitoring -->
                ${this.buildPerformanceMonitoring(performance)}

                <!-- Charts Row -->
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    ${this.buildUsageChart(stats)}
                    ${this.buildAdvancedCostChart(cost_analytics)}
                    ${this.buildPerformanceChart(stats, performance)}
                </div>

                <!-- AI Operations Intelligence -->
                ${this.buildAiOperationsIntelligence(ai_operations)}

                <!-- Advanced Analytics Row -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    ${this.buildModelComparison(models)}
                    ${this.buildCostBreakdown(costs, cost_analytics)}
                </div>

                <!-- Models List -->
                ${this.buildModelsTable(models)}

                <!-- Cost Analytics & Alerts -->
                ${this.buildCostAnalytics(cost_analytics)}

                <!-- Last Updated -->
                <div class="text-xs text-slate-400 text-center">
                    Last updated: ${this.lastUpdate ? this.lastUpdate.toLocaleTimeString() : 'Never'}
                </div>
            </div>
        `;
    }

    buildSystemStatusOverview(system_status, status) {
        const uptime = system_status?.uptime ? (system_status.uptime / 3600).toFixed(1) : '0.0';
        const version = system_status?.version || 'unknown';
        const active_agents = system_status?.active_agents || 0;
        const redis_status = system_status?.redis_connected ? 'Connected' : 'Disconnected';

        return `
            <div class="bg-slate-700 rounded-lg p-4">
                <h4 class="text-white font-medium mb-3">System Status Overview</h4>
                <div class="grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
                    <div>
                        <div class="text-slate-400">Uptime</div>
                        <div class="text-green-400 font-medium">${uptime}h</div>
                    </div>
                    <div>
                        <div class="text-slate-400">Version</div>
                        <div class="text-blue-400 font-medium">${version}</div>
                    </div>
                    <div>
                        <div class="text-slate-400">Active Agents</div>
                        <div class="text-purple-400 font-medium">${active_agents}</div>
                    </div>
                    <div>
                        <div class="text-slate-400">Redis</div>
                        <div class="text-${system_status?.redis_connected ? 'green' : 'yellow'}-400 font-medium">${redis_status}</div>
                    </div>
                </div>
            </div>
        `;
    }

    buildPerformanceMonitoring(performance) {
        if (!performance) return '';

        const slos = performance.slo_status || {};

        return `
            <div class="bg-slate-700 rounded-lg p-4">
                <h4 class="text-white font-medium mb-3">Performance & SLO Monitoring</h4>
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
                    ${Object.entries(slos).map(([name, slo]) => `
                        <div class="bg-slate-600 rounded p-3">
                            <div class="flex items-center justify-between mb-2">
                                <span class="text-slate-300 text-sm">${name.replace('_', ' ').toUpperCase()}</span>
                                <span class="w-2 h-2 rounded-full bg-${slo.status === 'healthy' ? 'green' : slo.status === 'warning' ? 'yellow' : 'red'}-500"></span>
                            </div>
                            <div class="text-white font-medium">${name === 'response_time' ? slo.current + 'ms' : slo.current + '%'}</div>
                            <div class="text-xs text-slate-400">Target: ${name === 'response_time' ? slo.target + 'ms' : slo.target + '%'}</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    buildAdvancedCostChart(cost_analytics) {
        if (!cost_analytics) return this.buildCostChart(null);

        const current = cost_analytics.current_period || {};
        const projections = cost_analytics.projections || {};
        const budget = cost_analytics.budget_alerts || {};

        return `
            <div class="bg-slate-700 rounded-lg p-6">
                <h4 class="text-white font-medium mb-4">Advanced Cost Analytics</h4>
                <div class="space-y-4">
                    <div class="flex justify-between">
                        <span class="text-slate-300">Daily Cost</span>
                        <span class="text-notion-400 font-medium">$${(current.daily_cost || 0).toFixed(4)}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-slate-300">Monthly Projection</span>
                        <span class="text-blue-400 font-medium">$${(projections.monthly || 0).toFixed(2)}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-slate-300">Budget Usage</span>
                        <span class="text-${budget.current_usage_pct > 80 ? 'red' : budget.current_usage_pct > 60 ? 'yellow' : 'green'}-400 font-medium">${(budget.current_usage_pct || 0).toFixed(1)}%</span>
                    </div>
                    <div class="w-full bg-slate-600 rounded-full h-2">
                        <div class="bg-${budget.current_usage_pct > 80 ? 'red' : budget.current_usage_pct > 60 ? 'yellow' : 'green'}-500 h-2 rounded-full" style="width: ${Math.min(100, budget.current_usage_pct || 0)}%"></div>
                    </div>
                    <div class="pt-2 border-t border-slate-600">
                        <div class="text-xs text-slate-400">
                            ${current.requests_today || 0} requests today • $${(current.cost_per_request || 0).toFixed(4)} per request
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    buildAiOperationsIntelligence(ai_operations) {
        if (!ai_operations) return '';

        const routing = ai_operations.routing_intelligence || {};
        const model_perf = ai_operations.model_performance || {};
        const context = ai_operations.context_optimization || {};

        return `
            <div class="bg-slate-700 rounded-lg p-4">
                <h4 class="text-white font-medium mb-3">AI Operations Intelligence</h4>
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
                    <div class="bg-slate-600 rounded p-3">
                        <h5 class="text-discord-400 font-medium mb-2">Routing Performance</h5>
                        <div class="space-y-2 text-sm">
                            <div class="flex justify-between">
                                <span class="text-slate-300">Success Rate</span>
                                <span class="text-green-400">${(routing.autom8_success_rate || 0).toFixed(1)}%</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-slate-300">Fallback Rate</span>
                                <span class="text-yellow-400">${(routing.fallback_rate || 0).toFixed(1)}%</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-slate-300">Optimal Routing</span>
                                <span class="text-blue-400">${(routing.optimal_routing_pct || 0).toFixed(1)}%</span>
                            </div>
                        </div>
                    </div>
                    <div class="bg-slate-600 rounded p-3">
                        <h5 class="text-notion-400 font-medium mb-2">Model Recommendations</h5>
                        <div class="space-y-2 text-sm">
                            <div class="flex justify-between">
                                <span class="text-slate-300">Fastest</span>
                                <span class="text-green-400">${model_perf.fastest_model || 'N/A'}</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-slate-300">Most Cost-Effective</span>
                                <span class="text-blue-400">${model_perf.most_cost_effective || 'N/A'}</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-slate-300">Recommended</span>
                                <span class="text-purple-400 font-medium">${model_perf.recommendation || 'N/A'}</span>
                            </div>
                        </div>
                    </div>
                    <div class="bg-slate-600 rounded p-3">
                        <h5 class="text-cyan-400 font-medium mb-2">Context Optimization</h5>
                        <div class="space-y-2 text-sm">
                            <div class="flex justify-between">
                                <span class="text-slate-300">Avg Context</span>
                                <span class="text-white">${context.avg_context_length || 0} tokens</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-slate-300">Reduction</span>
                                <span class="text-green-400">${(context.context_reduction_pct || 0).toFixed(1)}%</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-slate-300">Tokens Saved</span>
                                <span class="text-cyan-400">${context.token_savings || 0}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    buildCostAnalytics(cost_analytics) {
        if (!cost_analytics) return '';

        const savings = cost_analytics.savings || {};
        const budget = cost_analytics.budget_alerts || {};

        return `
            <div class="bg-slate-700 rounded-lg p-4">
                <h4 class="text-white font-medium mb-3">Cost Analytics & Alerts</h4>
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <div class="bg-slate-600 rounded p-3">
                        <h5 class="text-green-400 font-medium mb-2">Cost Savings</h5>
                        <div class="space-y-2 text-sm">
                            <div class="flex justify-between">
                                <span class="text-slate-300">vs OpenAI GPT-4</span>
                                <span class="text-green-400">$${(savings.vs_openai_gpt4 || 0).toFixed(4)}</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-slate-300">vs Anthropic Claude</span>
                                <span class="text-green-400">$${(savings.vs_anthropic_claude || 0).toFixed(4)}</span>
                            </div>
                        </div>
                    </div>
                    <div class="bg-slate-600 rounded p-3">
                        <h5 class="text-${budget.projected_monthly_pct > 80 ? 'red' : budget.projected_monthly_pct > 60 ? 'yellow' : 'blue'}-400 font-medium mb-2">Budget Monitoring</h5>
                        <div class="space-y-2 text-sm">
                            <div class="flex justify-between">
                                <span class="text-slate-300">Daily Limit</span>
                                <span class="text-white">$${budget.daily_limit || 0}</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-slate-300">Monthly Limit</span>
                                <span class="text-white">$${budget.monthly_limit || 0}</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-slate-300">Projected Usage</span>
                                <span class="text-${budget.projected_monthly_pct > 80 ? 'red' : budget.projected_monthly_pct > 60 ? 'yellow' : 'green'}-400">${(budget.projected_monthly_pct || 0).toFixed(1)}%</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    buildStatusIndicator(status, system_status) {
        if (!status) {
            return '<span class="flex items-center text-slate-400"><span class="w-2 h-2 bg-slate-500 rounded-full mr-2"></span>Unknown</span>';
        }

        const isEnabled = status.enabled;
        const isAvailable = status.available;
        const lastHealthCheck = status.last_health_check;
        const errorMessage = status.error_message;

        if (!isEnabled) {
            return `
                <div class="flex items-center">
                    <span class="flex items-center text-yellow-400">
                        <span class="w-2 h-2 bg-yellow-500 rounded-full mr-2"></span>
                        Disabled
                    </span>
                    <div class="ml-2 text-xs">
                        <button onclick="this.classList.toggle('hidden'); this.nextElementSibling.classList.toggle('hidden')"
                                class="text-blue-400 hover:text-blue-300">ℹ️</button>
                        <div class="hidden absolute z-10 bg-slate-800 border border-slate-600 rounded p-2 text-xs max-w-xs">
                            Autom8 integration is disabled in configuration.
                            Enable in settings to access advanced AI routing.
                        </div>
                    </div>
                </div>
            `;
        }

        if (isAvailable) {
            return `
                <div class="flex items-center">
                    <span class="flex items-center text-green-400">
                        <span class="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></span>
                        Online
                    </span>
                    <div class="ml-2 text-xs text-slate-400">
                        ${lastHealthCheck ? `Last check: ${new Date(lastHealthCheck).toLocaleTimeString()}` : ''}
                    </div>
                </div>
            `;
        }

        return `
            <div class="flex items-center">
                <span class="flex items-center text-orange-400">
                    <span class="w-2 h-2 bg-orange-500 rounded-full mr-2"></span>
                    Offline (Fallback Mode)
                </span>
                <div class="ml-2 text-xs">
                    <button onclick="this.classList.toggle('hidden'); this.nextElementSibling.classList.toggle('hidden')"
                            class="text-blue-400 hover:text-blue-300">ℹ️</button>
                    <div class="hidden absolute z-10 bg-slate-800 border border-slate-600 rounded p-2 text-xs max-w-xs">
                        <div class="font-semibold mb-1">Service Unavailable</div>
                        <div class="text-slate-300 mb-2">${errorMessage || 'Autom8 microservice is not running'}</div>
                        <div class="text-slate-400 mb-2">Using Ollama fallback for AI requests.</div>
                        <div class="font-mono text-xs bg-slate-700 p-1 rounded">
                            python scripts/start_autom8.py start
                        </div>
                        ${lastHealthCheck ? `<div class="text-slate-500 mt-1">Last check: ${new Date(lastHealthCheck).toLocaleTimeString()}</div>` : ''}
                    </div>
                </div>
            </div>
        `;
    }

    buildQuickStats(stats, costs) {
        if (!stats) {
            return '<div class="text-slate-400 text-center py-4">No statistics available</div>';
        }

        const responseTime = stats.average_response_time || 0;
        const successRate = stats.success_rate || 100;
        const costPerRequest = costs?.total_cost > 0 ? (costs.total_cost / costs.requests_count) : 0;

        return `
            <div class="grid grid-cols-2 lg:grid-cols-6 gap-4">
                <div class="bg-slate-700 rounded-lg p-4 hover:bg-slate-600 transition-colors">
                    <div class="text-2xl font-bold text-white">${stats.total_requests || 0}</div>
                    <div class="text-sm text-slate-400">Total Requests</div>
                    <div class="text-xs text-green-400 mt-1">
                        ${stats.autom8_requests || 0} via Autom8, ${stats.ollama_fallbacks || 0} via Ollama
                    </div>
                </div>
                <div class="bg-slate-700 rounded-lg p-4 hover:bg-slate-600 transition-colors">
                    <div class="text-2xl font-bold text-discord-400">${Math.round((stats.autom8_usage_rate || 0) * 100)}%</div>
                    <div class="text-sm text-slate-400">Autom8 Usage</div>
                    <div class="text-xs ${stats.autom8_usage_rate > 0.8 ? 'text-green-400' : stats.autom8_usage_rate > 0.5 ? 'text-yellow-400' : 'text-red-400'} mt-1">
                        ${stats.autom8_usage_rate > 0.8 ? 'Excellent' : stats.autom8_usage_rate > 0.5 ? 'Good' : 'Low'} utilization
                    </div>
                </div>
                <div class="bg-slate-700 rounded-lg p-4 hover:bg-slate-600 transition-colors">
                    <div class="text-2xl font-bold text-notion-400">$${(costs?.total_cost || 0).toFixed(4)}</div>
                    <div class="text-sm text-slate-400">24h Cost</div>
                    <div class="text-xs text-blue-400 mt-1">
                        $${costPerRequest.toFixed(4)} per request
                    </div>
                </div>
                <div class="bg-slate-700 rounded-lg p-4 hover:bg-slate-600 transition-colors">
                    <div class="text-2xl font-bold text-green-400">$${(stats.cost_savings_estimate || 0).toFixed(2)}</div>
                    <div class="text-sm text-slate-400">Est. Savings</div>
                    <div class="text-xs text-green-400 mt-1">
                        vs. premium APIs
                    </div>
                </div>
                <div class="bg-slate-700 rounded-lg p-4 hover:bg-slate-600 transition-colors">
                    <div class="text-2xl font-bold text-purple-400">${responseTime.toFixed(1)}s</div>
                    <div class="text-sm text-slate-400">Avg Response</div>
                    <div class="text-xs ${responseTime < 2 ? 'text-green-400' : responseTime < 5 ? 'text-yellow-400' : 'text-red-400'} mt-1">
                        ${responseTime < 2 ? 'Fast' : responseTime < 5 ? 'Moderate' : 'Slow'}
                    </div>
                </div>
                <div class="bg-slate-700 rounded-lg p-4 hover:bg-slate-600 transition-colors">
                    <div class="text-2xl font-bold text-cyan-400">${successRate.toFixed(1)}%</div>
                    <div class="text-sm text-slate-400">Success Rate</div>
                    <div class="text-xs ${successRate >= 99 ? 'text-green-400' : successRate >= 95 ? 'text-yellow-400' : 'text-red-400'} mt-1">
                        ${successRate >= 99 ? 'Excellent' : successRate >= 95 ? 'Good' : 'Needs attention'}
                    </div>
                </div>
            </div>
        `;
    }

    buildUsageChart(stats) {
        if (!stats) return '<div class="bg-slate-700 rounded-lg p-4 text-slate-400 text-center">No usage data</div>';

        const autom8Pct = Math.round((stats.autom8_usage_rate || 0) * 100);
        const ollamaPct = 100 - autom8Pct;

        return `
            <div class="bg-slate-700 rounded-lg p-6">
                <h4 class="text-white font-medium mb-4">Request Distribution</h4>
                <div class="space-y-3">
                    <div class="flex items-center justify-between">
                        <span class="text-slate-300">Autom8 Routing</span>
                        <span class="text-discord-400 font-medium">${stats.autom8_requests || 0} (${autom8Pct}%)</span>
                    </div>
                    <div class="w-full bg-slate-600 rounded-full h-2">
                        <div class="bg-discord-500 h-2 rounded-full" style="width: ${autom8Pct}%"></div>
                    </div>
                    <div class="flex items-center justify-between">
                        <span class="text-slate-300">Ollama Fallback</span>
                        <span class="text-yellow-400 font-medium">${stats.ollama_fallbacks || 0} (${ollamaPct}%)</span>
                    </div>
                    <div class="w-full bg-slate-600 rounded-full h-2">
                        <div class="bg-yellow-500 h-2 rounded-full" style="width: ${ollamaPct}%"></div>
                    </div>
                </div>
            </div>
        `;
    }

    buildCostChart(costs) {
        if (!costs) return '<div class="bg-slate-700 rounded-lg p-4 text-slate-400 text-center">No cost data</div>';

        const projectedMonthlyCost = costs.projected_monthly_cost || 0;
        const avgCostPerRequest = costs.total_cost > 0 ? (costs.total_cost / costs.requests_count) : 0;

        return `
            <div class="bg-slate-700 rounded-lg p-6">
                <h4 class="text-white font-medium mb-4">Cost Analysis</h4>
                <div class="space-y-4">
                    <div class="flex justify-between">
                        <span class="text-slate-300">24h Total</span>
                        <span class="text-notion-400 font-medium">$${costs.total_cost.toFixed(4)}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-slate-300">Avg/Request</span>
                        <span class="text-blue-400 font-medium">$${avgCostPerRequest.toFixed(4)}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-slate-300">Monthly Proj.</span>
                        <span class="text-purple-400 font-medium">$${projectedMonthlyCost.toFixed(2)}</span>
                    </div>
                    <div class="pt-2 border-t border-slate-600">
                        <div class="text-xs text-slate-400">
                            Based on ${costs.requests_count} requests in ${costs.period_hours}h
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    buildPerformanceChart(stats, performance) {
        if (!stats) return '<div class="bg-slate-700 rounded-lg p-4 text-slate-400 text-center">No performance data</div>';

        const responseTime = performance?.response_times?.average_5m || stats.average_response_time || 0;
        const successRate = stats.success_rate || 100;
        const performanceScore = Math.round((successRate / 100) * (responseTime < 2 ? 100 : responseTime < 5 ? 80 : 60));

        return `
            <div class="bg-slate-700 rounded-lg p-6">
                <h4 class="text-white font-medium mb-4">Performance Metrics</h4>
                <div class="space-y-4">
                    <div class="flex justify-between">
                        <span class="text-slate-300">Response Time</span>
                        <span class="text-${responseTime < 2 ? 'green' : responseTime < 5 ? 'yellow' : 'red'}-400 font-medium">${responseTime.toFixed(1)}s</span>
                    </div>
                    <div class="w-full bg-slate-600 rounded-full h-2">
                        <div class="bg-${responseTime < 2 ? 'green' : responseTime < 5 ? 'yellow' : 'red'}-500 h-2 rounded-full" style="width: ${Math.max(0, 100 - (responseTime * 10))}%"></div>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-slate-300">Success Rate</span>
                        <span class="text-${successRate >= 99 ? 'green' : successRate >= 95 ? 'yellow' : 'red'}-400 font-medium">${successRate.toFixed(1)}%</span>
                    </div>
                    <div class="w-full bg-slate-600 rounded-full h-2">
                        <div class="bg-${successRate >= 99 ? 'green' : successRate >= 95 ? 'yellow' : 'red'}-500 h-2 rounded-full" style="width: ${successRate}%"></div>
                    </div>
                    <div class="pt-2 border-t border-slate-600">
                        <div class="flex justify-between">
                            <span class="text-slate-300">Performance Score</span>
                            <span class="text-discord-400 font-bold">${performanceScore}/100</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    buildModelComparison(models) {
        if (!models || models.length === 0) {
            return '<div class="bg-slate-700 rounded-lg p-4 text-slate-400 text-center">No model comparison data</div>';
        }

        const topModels = models.slice(0, 3);
        const modelBars = topModels.map((model, index) => {
            const maxRequests = Math.max(...topModels.map(m => m.requests_count || 0));
            const percentage = maxRequests > 0 ? ((model.requests_count || 0) / maxRequests) * 100 : 0;
            const colors = ['discord', 'notion', 'yellow'];
            const color = colors[index] || 'slate';

            return `
                <div class="flex items-center justify-between mb-3">
                    <div class="flex items-center space-x-2 flex-1">
                        <span class="w-2 h-2 rounded-full bg-${color}-500"></span>
                        <span class="text-slate-300 text-sm truncate">${model.name}</span>
                    </div>
                    <span class="text-${color}-400 text-sm font-medium ml-2">${model.requests_count || 0}</span>
                </div>
                <div class="w-full bg-slate-600 rounded-full h-2 mb-4">
                    <div class="bg-${color}-500 h-2 rounded-full transition-all duration-300" style="width: ${percentage}%"></div>
                </div>
            `;
        }).join('');

        return `
            <div class="bg-slate-700 rounded-lg p-6">
                <h4 class="text-white font-medium mb-4">Model Usage Comparison</h4>
                <div class="space-y-2">
                    ${modelBars}
                    ${models.length > 3 ? `<div class="text-xs text-slate-400 text-center pt-2">+${models.length - 3} more models</div>` : ''}
                </div>
            </div>
        `;
    }

    buildCostBreakdown(costs, cost_analytics) {
        if (!costs) return '<div class="bg-slate-700 rounded-lg p-4 text-slate-400 text-center">No cost breakdown data</div>';

        const costByProvider = costs.cost_by_provider || {};
        const costByModel = costs.cost_by_model || {};
        const totalCost = costs.total_cost || 0;

        const providerBreakdown = Object.entries(costByProvider).map(([provider, cost]) => {
            const percentage = totalCost > 0 ? (cost / totalCost) * 100 : 0;
            return `
                <div class="flex justify-between items-center mb-2">
                    <span class="text-slate-300 text-sm">${provider}</span>
                    <div class="flex items-center space-x-2">
                        <span class="text-green-400 text-sm">$${cost.toFixed(4)}</span>
                        <span class="text-slate-400 text-xs">(${percentage.toFixed(1)}%)</span>
                    </div>
                </div>
                <div class="w-full bg-slate-600 rounded-full h-1 mb-3">
                    <div class="bg-green-500 h-1 rounded-full" style="width: ${percentage}%"></div>
                </div>
            `;
        }).join('');

        return `
            <div class="bg-slate-700 rounded-lg p-6">
                <h4 class="text-white font-medium mb-4">Cost Breakdown</h4>
                <div class="space-y-1">
                    ${providerBreakdown || '<div class="text-slate-400 text-sm text-center">No cost data available</div>'}
                    <div class="pt-3 border-t border-slate-600">
                        <div class="flex justify-between">
                            <span class="text-slate-300 font-medium">Total Cost</span>
                            <span class="text-green-400 font-bold">$${totalCost.toFixed(4)}</span>
                        </div>
                        <div class="text-xs text-slate-400 mt-1">
                            Over ${costs.period_hours || 24} hours
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    buildModelsTable(models) {
        if (!models || models.length === 0) {
            return '<div class="bg-slate-700 rounded-lg p-4 text-slate-400 text-center">No model data available</div>';
        }

        const modelRows = models.slice(0, 5).map(model => `
            <tr class="border-b border-slate-600">
                <td class="py-3 px-4">
                    <div class="flex items-center">
                        <span class="w-2 h-2 rounded-full mr-2 ${model.available ? 'bg-green-500' : 'bg-red-500'}"></span>
                        <span class="text-white font-medium">${model.name}</span>
                    </div>
                    <div class="text-xs text-slate-400">${model.provider}</div>
                </td>
                <td class="py-3 px-4 text-slate-300">${model.requests_count}</td>
                <td class="py-3 px-4 text-slate-300">${model.avg_response_time.toFixed(1)}s</td>
                <td class="py-3 px-4 text-slate-300">$${model.cost_per_request.toFixed(4)}</td>
            </tr>
        `).join('');

        return `
            <div class="bg-slate-700 rounded-lg overflow-hidden">
                <div class="px-6 py-4 border-b border-slate-600">
                    <h4 class="text-white font-medium">Active Models</h4>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full">
                        <thead class="bg-slate-800">
                            <tr>
                                <th class="py-3 px-4 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">Model</th>
                                <th class="py-3 px-4 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">Requests</th>
                                <th class="py-3 px-4 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">Avg Time</th>
                                <th class="py-3 px-4 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">Cost/Req</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${modelRows}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }

    attachEventListeners() {
        // Add any interactive functionality here
    }

    async refresh() {
        try {
            this.data = await this.loadDashboardData();
            this.renderDashboard();
            console.log('Autom8 Dashboard refreshed');
        } catch (error) {
            console.error('Failed to refresh dashboard:', error);
        }
    }

    async triggerHealthCheck() {
        try {
            const response = await fetch('/api/autom8/health-check', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const result = await response.json();

            // Show a brief status update
            const statusContainer = document.querySelector('.autom8-dashboard .flex.items-center.justify-between');
            if (statusContainer) {
                const existingAlert = statusContainer.querySelector('.health-check-alert');
                if (existingAlert) {
                    existingAlert.remove();
                }

                const alert = document.createElement('div');
                alert.className = `health-check-alert text-xs px-2 py-1 rounded ${result.healthy ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`;
                alert.textContent = result.message;
                statusContainer.appendChild(alert);

                // Remove alert after 3 seconds
                setTimeout(() => {
                    if (alert.parentNode) {
                        alert.remove();
                    }
                }, 3000);
            }

            // Refresh data after health check
            await this.refresh();

        } catch (error) {
            console.error('Health check failed:', error);
            // Show error notification
            const statusContainer = document.querySelector('.autom8-dashboard .flex.items-center.justify-between');
            if (statusContainer) {
                const alert = document.createElement('div');
                alert.className = 'health-check-alert text-xs px-2 py-1 rounded bg-red-500/20 text-red-400';
                alert.textContent = 'Health check failed';
                statusContainer.appendChild(alert);

                setTimeout(() => {
                    if (alert.parentNode) {
                        alert.remove();
                    }
                }, 3000);
            }
        }
    }

    startPeriodicUpdates() {
        // Update every 5 minutes when service is unavailable
        this.updateInterval = setInterval(() => {
            this.refresh();
        }, 300000);  // 5 minutes instead of 30 seconds
    }

    stopPeriodicUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }

    renderError(message) {
        const container = document.getElementById('autom8-dashboard');
        if (container) {
            container.innerHTML = `
                <div class="bg-red-900/20 border border-red-700 rounded-lg p-6 text-center">
                    <div class="text-red-400 mb-2">⚠️ Error</div>
                    <div class="text-slate-300">${message}</div>
                    <button onclick="autom8Dashboard.refresh()"
                            class="mt-3 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors">
                        Retry
                    </button>
                </div>
            `;
        }
    }

    destroy() {
        this.stopPeriodicUpdates();
        this.initialized = false;
    }
}

// Global instance - will be created when needed
window.autom8Dashboard = null;

// Make the class available globally
window.Autom8Dashboard = Autom8Dashboard;

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.autom8Dashboard) {
        window.autom8Dashboard.destroy();
    }
});