/**
 * Dashboard Performance & Advanced Features - Enhanced with Error Boundaries
 * Phase 1-3 of Dashboard v3 completion plan
 * 
 * Features:
 * - Comprehensive error boundary system
 * - Performance regression detection
 * - Data loading optimization with caching & deduplication
 * - Rendering performance with virtual scrolling & DOM efficiency
 * - Memory management & cleanup
 * - Enhanced search with debouncing & autocomplete
 * - Advanced drag & drop functionality
 * - Comprehensive keyboard shortcuts
 * - Interactive data visualization
 * - Smart features & auto-save
 */

class DashboardPerformance {
    constructor() {
        // Error boundary system
        this.errorBoundary = {
            errors: [],
            maxErrors: 50,
            errorThreshold: 10,
            recoveryAttempts: 0,
            maxRecoveryAttempts: 3,
            suppressedErrors: new Set(),
            lastErrorTime: 0,
            errorFrequencyLimit: 5, // max 5 errors per second
            componentStates: new Map(),
            fallbackComponents: new Map()
        };

        // Performance monitoring with regression detection
        this.performanceMetrics = {
            fcpTime: 0,
            lcpTime: 0,
            clsScore: 0,
            fidTime: 0,
            memoryUsage: 0,
            apiCallCount: 0,
            cacheHitRate: 0,
            // Regression detection
            baselines: {},
            regressionThreshold: 1.5, // 50% slower is a regression
            performanceAlerts: [],
            componentRenderTimes: new Map(),
            criticalPathMetrics: {
                searchResponseTime: [],
                noteLoadTime: [],
                uiInteractionDelay: []
            }
        };

        // Caching system
        this.cache = new Map();
        this.cacheConfig = {
            maxSize: 100,
            ttl: 5 * 60 * 1000, // 5 minutes
            analytics: 1 * 60 * 1000, // 1 minute for analytics
            recentActivity: 30 * 1000, // 30 seconds for activity
            notes: 2 * 60 * 1000 // 2 minutes for notes
        };

        // Request deduplication
        this.pendingRequests = new Map();
        this.requestQueue = [];

        // Search system
        this.searchState = {
            query: '',
            filters: [],
            suggestions: [],
            debounceTimeout: null,
            debounceDelay: 300,
            results: [],
            history: JSON.parse(localStorage.getItem('searchHistory') || '[]')
        };

        // Keyboard shortcuts
        this.shortcuts = new Map();
        this.shortcutHelpVisible = false;

        // Virtual scrolling
        this.virtualScrollers = new Map();

        // Drag & drop state
        this.dragState = {
            isDragging: false,
            draggedElement: null,
            dropZones: [],
            draggedFiles: []
        };

        // Memory management
        this.eventListeners = [];
        this.intervals = [];
        this.timeouts = [];
        this.observers = [];

        // Smart features
        this.smartFeatures = {
            autoSaveEnabled: true,
            smartSuggestions: true,
            contextualRecommendations: true,
            behaviorTracking: true
        };

        this.init();
    }

    /**
     * INITIALIZATION
     */
    async init() {
        console.log('🚀 Initializing Dashboard Performance System with Error Boundaries');
        
        try {
            // Initialize error boundary system first
            this.initializeErrorBoundary();
            
            // Start performance monitoring with regression detection
            this.startPerformanceMonitoring();
            this.initializePerformanceRegression();
            
            // Initialize core systems with error protection
            await this.safeExecute('core-initialization', async () => {
                this.initializeCaching();
                this.initializeRequestDeduplication();
                this.initializeSearchSystem();
                this.initializeKeyboardShortcuts();
                this.initializeDragAndDrop();
                this.initializeVirtualScrolling();
                this.initializeSmartFeatures();
                this.initializeDataVisualization();
            });
            
            // Setup memory management
            this.setupMemoryManagement();
            
            // Initialize data preloading
            await this.preloadCriticalData();
            
            // Setup cleanup on page unload
            this.setupPageCleanup();
            
            console.log('✅ Dashboard Performance System initialized with error boundaries');
            
        } catch (error) {
            this.handleCriticalError('dashboard-initialization', error);
            // Try to recover with minimal functionality
            await this.initializeFailsafeMode();
        }
    }

    /**
     * ERROR BOUNDARY SYSTEM
     */
    initializeErrorBoundary() {
        // Global error handler
        window.addEventListener('error', (event) => {
            this.handleError('global', event.error, {
                message: event.message,
                filename: event.filename,
                lineno: event.lineno,
                colno: event.colno
            });
        });

        // Unhandled promise rejection handler
        window.addEventListener('unhandledrejection', (event) => {
            this.handleError('promise', event.reason, {
                type: 'unhandledrejection'
            });
        });

        // Setup component error boundaries
        this.initializeComponentBoundaries();
        
        console.log('🛡️ Error boundary system initialized');
    }

    initializeComponentBoundaries() {
        // Register fallback components for critical areas
        this.errorBoundary.fallbackComponents.set('search', () => 
            this.createFallbackSearchComponent()
        );
        this.errorBoundary.fallbackComponents.set('notes-list', () => 
            this.createFallbackNotesComponent()
        );
        this.errorBoundary.fallbackComponents.set('analytics', () => 
            this.createFallbackAnalyticsComponent()
        );
        this.errorBoundary.fallbackComponents.set('file-upload', () => 
            this.createFallbackUploadComponent()
        );
    }

    async safeExecute(componentId, operation, fallback = null) {
        try {
            const startTime = performance.now();
            const result = await operation();
            const duration = performance.now() - startTime;
            
            // Track component performance
            if (!this.performanceMetrics.componentRenderTimes.has(componentId)) {
                this.performanceMetrics.componentRenderTimes.set(componentId, []);
            }
            this.performanceMetrics.componentRenderTimes.get(componentId).push(duration);
            
            return result;
        } catch (error) {
            return this.handleComponentError(componentId, error, fallback);
        }
    }

    handleError(source, error, context = {}) {
        const now = Date.now();
        
        // Check error frequency to prevent spam
        if (now - this.errorBoundary.lastErrorTime < 1000 / this.errorBoundary.errorFrequencyLimit) {
            return; // Rate limited
        }
        
        this.errorBoundary.lastErrorTime = now;

        const errorInfo = {
            id: `${source}-${now}`,
            source,
            error: {
                name: error?.name || 'UnknownError',
                message: error?.message || 'Unknown error occurred',
                stack: error?.stack || 'No stack trace available'
            },
            context,
            timestamp: now,
            userAgent: navigator.userAgent,
            url: window.location.href,
            userId: this.getCurrentUserId()
        };

        // Add to error history
        this.errorBoundary.errors.unshift(errorInfo);
        if (this.errorBoundary.errors.length > this.errorBoundary.maxErrors) {
            this.errorBoundary.errors.pop();
        }

        // Log to console in development
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            console.error('🚨 Dashboard Error:', errorInfo);
        }

        // Send to error tracking service
        this.reportError(errorInfo);

        // Show user notification for critical errors
        if (this.isCriticalError(error)) {
            this.showErrorNotification(errorInfo);
        }

        // Check if recovery is needed
        if (this.errorBoundary.errors.length >= this.errorBoundary.errorThreshold) {
            this.attemptRecovery();
        }
    }

    handleComponentError(componentId, error, fallback) {
        this.handleError('component', error, { componentId });

        // Store component state before showing fallback
        const element = document.getElementById(componentId) || 
                       document.querySelector(`[data-component="${componentId}"]`);
        
        if (element) {
            this.errorBoundary.componentStates.set(componentId, {
                html: element.innerHTML,
                classes: [...element.classList],
                attributes: this.getElementAttributes(element)
            });
        }

        // Show fallback component
        if (fallback) {
            return fallback();
        } else if (this.errorBoundary.fallbackComponents.has(componentId)) {
            const fallbackComponent = this.errorBoundary.fallbackComponents.get(componentId);
            return fallbackComponent();
        }

        // Default fallback
        return this.createGenericFallback(componentId, error);
    }

    handleCriticalError(operation, error) {
        console.error(`💥 Critical dashboard error in ${operation}:`, error);
        
        // Track critical error
        this.handleError('critical', error, { operation });
        
        // Show prominent error message to user
        this.showCriticalErrorModal(operation, error);
        
        // Attempt immediate recovery if possible
        if (this.errorBoundary.recoveryAttempts < this.errorBoundary.maxRecoveryAttempts) {
            this.errorBoundary.recoveryAttempts++;
            setTimeout(() => this.attemptRecovery(), 1000);
        }
    }

    async attemptRecovery() {
        console.log('🔄 Attempting dashboard recovery...');
        
        try {
            // Clear problematic data
            this.cache.clear();
            this.pendingRequests.clear();
            
            // Reset component states
            for (const [componentId, state] of this.errorBoundary.componentStates.entries()) {
                try {
                    const element = document.getElementById(componentId) || 
                                   document.querySelector(`[data-component="${componentId}"]`);
                    if (element && state) {
                        element.innerHTML = state.html;
                        element.className = state.classes.join(' ');
                    }
                } catch (e) {
                    console.warn(`Failed to restore component ${componentId}:`, e);
                }
            }
            
            // Re-initialize critical systems
            await this.safeExecute('recovery-search', () => this.initializeSearchSystem());
            await this.safeExecute('recovery-cache', () => this.initializeCaching());
            
            console.log('✅ Dashboard recovery completed');
            this.showNotification('Dashboard recovered successfully', 'success');
            
            // Reset error counter after successful recovery
            this.errorBoundary.errors = [];
            this.errorBoundary.recoveryAttempts = 0;
            
        } catch (recoveryError) {
            console.error('❌ Recovery failed:', recoveryError);
            this.initializeFailsafeMode();
        }
    }

    async initializeFailsafeMode() {
        console.log('🚨 Initializing failsafe mode');
        
        try {
            // Remove all advanced features, keep only basic functionality
            const essentialElements = ['basic-search', 'note-list', 'create-note'];
            
            // Hide advanced features
            document.querySelectorAll('[data-feature="advanced"]').forEach(el => {
                el.style.display = 'none';
            });
            
            // Show failsafe notification
            this.showFailsafeNotification();
            
            // Initialize minimal search functionality
            const searchInput = document.getElementById('search-input');
            if (searchInput) {
                searchInput.addEventListener('input', this.basicSearch.bind(this));
            }
            
        } catch (error) {
            console.error('💥 Even failsafe mode failed:', error);
            // Last resort - show static error page
            this.showStaticErrorPage();
        }
    }

    // Fallback component creators
    createFallbackSearchComponent() {
        return `
            <div class="error-fallback bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
                <h3 class="text-red-800 font-medium">Search Temporarily Unavailable</h3>
                <p class="text-red-600 text-sm mt-1">Please refresh the page to restore search functionality.</p>
                <button onclick="location.reload()" class="mt-2 bg-red-600 text-white px-3 py-1 rounded text-sm">
                    Refresh Page
                </button>
            </div>
        `;
    }

    createFallbackNotesComponent() {
        return `
            <div class="error-fallback bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <h3 class="text-yellow-800 font-medium">Notes List Unavailable</h3>
                <p class="text-yellow-600 text-sm mt-1">There was an issue loading your notes. Try refreshing the page.</p>
                <button onclick="location.reload()" class="mt-2 bg-yellow-600 text-white px-3 py-1 rounded text-sm">
                    Reload Notes
                </button>
            </div>
        `;
    }

    createFallbackAnalyticsComponent() {
        return `
            <div class="error-fallback bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h3 class="text-blue-800 font-medium">Analytics Temporarily Unavailable</h3>
                <p class="text-blue-600 text-sm mt-1">Analytics data will be available after the next page refresh.</p>
            </div>
        `;
    }

    createFallbackUploadComponent() {
        return `
            <div class="error-fallback bg-gray-50 border border-gray-200 rounded-lg p-4">
                <h3 class="text-gray-800 font-medium">File Upload Unavailable</h3>
                <p class="text-gray-600 text-sm mt-1">File upload is temporarily disabled. Please try again later.</p>
            </div>
        `;
    }

    createGenericFallback(componentId, error) {
        return `
            <div class="error-fallback bg-gray-50 border border-gray-200 rounded-lg p-4">
                <h3 class="text-gray-800 font-medium">Component Error</h3>
                <p class="text-gray-600 text-sm mt-1">The ${componentId} component encountered an error.</p>
                <details class="mt-2">
                    <summary class="text-xs text-gray-500 cursor-pointer">Error Details</summary>
                    <pre class="text-xs text-gray-400 mt-1 max-w-full overflow-auto">${error?.message || 'Unknown error'}</pre>
                </details>
            </div>
        `;
    }

    // Performance regression detection
    initializePerformanceRegression() {
        // Load performance baselines from localStorage
        const storedBaselines = localStorage.getItem('performance-baselines');
        if (storedBaselines) {
            try {
                this.performanceMetrics.baselines = JSON.parse(storedBaselines);
            } catch (e) {
                console.warn('Failed to parse performance baselines:', e);
            }
        }

        // Set up periodic regression checks
        setInterval(() => this.checkPerformanceRegressions(), 30000); // Every 30 seconds
    }

    checkPerformanceRegressions() {
        const metrics = this.getCurrentPerformanceMetrics();
        const regressions = [];

        for (const [metric, currentValue] of Object.entries(metrics)) {
            const baseline = this.performanceMetrics.baselines[metric];
            if (baseline && currentValue > baseline * this.performanceMetrics.regressionThreshold) {
                regressions.push({
                    metric,
                    currentValue,
                    baselineValue: baseline,
                    regressionFactor: currentValue / baseline
                });
            }
        }

        if (regressions.length > 0) {
            this.handlePerformanceRegression(regressions);
        }

        // Update baselines periodically (keep the best performances)
        this.updatePerformanceBaselines(metrics);
    }

    handlePerformanceRegression(regressions) {
        console.warn('⚠️ Performance regression detected:', regressions);
        
        this.performanceMetrics.performanceAlerts.push({
            timestamp: Date.now(),
            type: 'regression',
            regressions: regressions
        });

        // Show warning to developers
        if (window.location.hostname === 'localhost') {
            console.table(regressions);
        }

        // Attempt performance optimization
        this.optimizePerformance(regressions);
    }

    optimizePerformance(regressions) {
        // Implement automatic performance optimizations
        for (const regression of regressions) {
            switch (regression.metric) {
                case 'searchResponseTime':
                    this.optimizeSearch();
                    break;
                case 'noteLoadTime':
                    this.optimizeNoteLoading();
                    break;
                case 'memoryUsage':
                    this.performMemoryCleanup();
                    break;
                default:
                    console.log(`No optimization available for ${regression.metric}`);
            }
        }
    }

    // Utility methods for error boundary
    isCriticalError(error) {
        const criticalPatterns = [
            /TypeError.*cannot read/i,
            /ReferenceError/i,
            /Network request failed/i,
            /CORS error/i,
            /Authentication failed/i
        ];
        
        const message = error?.message || '';
        return criticalPatterns.some(pattern => pattern.test(message));
    }

    getCurrentUserId() {
        // Extract user ID from various sources
        return localStorage.getItem('userId') || 
               sessionStorage.getItem('userId') || 
               'anonymous';
    }

    getElementAttributes(element) {
        const attributes = {};
        for (const attr of element.attributes) {
            attributes[attr.name] = attr.value;
        }
        return attributes;
    }

    getCurrentPerformanceMetrics() {
        return {
            searchResponseTime: this.calculateAverageMetric(this.performanceMetrics.criticalPathMetrics.searchResponseTime),
            noteLoadTime: this.calculateAverageMetric(this.performanceMetrics.criticalPathMetrics.noteLoadTime),
            memoryUsage: this.performanceMetrics.memoryUsage || 0,
            cacheHitRate: this.performanceMetrics.cacheHitRate || 0
        };
    }

    calculateAverageMetric(metricArray) {
        if (!metricArray || metricArray.length === 0) return 0;
        return metricArray.reduce((sum, val) => sum + val, 0) / metricArray.length;
    }

    updatePerformanceBaselines(currentMetrics) {
        for (const [metric, value] of Object.entries(currentMetrics)) {
            const baseline = this.performanceMetrics.baselines[metric];
            if (!baseline || value < baseline) {
                this.performanceMetrics.baselines[metric] = value;
            }
        }

        // Save to localStorage
        localStorage.setItem('performance-baselines', JSON.stringify(this.performanceMetrics.baselines));
    }

    async reportError(errorInfo) {
        try {
            // Send error to backend for logging
            await fetch('/api/errors', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(errorInfo)
            });
        } catch (e) {
            // Silently fail if error reporting fails
            console.warn('Failed to report error:', e);
        }
    }

    showErrorNotification(errorInfo) {
        const notification = document.createElement('div');
        notification.className = 'fixed top-4 right-4 bg-red-500 text-white px-4 py-2 rounded-lg shadow-lg z-50 max-w-sm';
        notification.innerHTML = `
            <div class="flex items-center space-x-2">
                <span class="text-sm">⚠️ An error occurred</span>
                <button onclick="this.parentElement.parentElement.remove()" class="text-white hover:text-gray-200">
                    ✕
                </button>
            </div>
            <p class="text-xs mt-1 opacity-80">${errorInfo.error.message}</p>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    showCriticalErrorModal(operation, error) {
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        modal.innerHTML = `
            <div class="bg-white rounded-lg p-6 max-w-md mx-4">
                <h2 class="text-xl font-bold text-red-600 mb-4">Critical Error</h2>
                <p class="text-gray-700 mb-4">
                    A critical error occurred in ${operation}. The dashboard will attempt to recover automatically.
                </p>
                <details class="mb-4">
                    <summary class="cursor-pointer text-sm text-gray-500">Technical Details</summary>
                    <pre class="text-xs text-gray-400 mt-2 max-h-32 overflow-auto">${error?.stack || error?.message || 'Unknown error'}</pre>
                </details>
                <div class="flex space-x-2">
                    <button onclick="location.reload()" class="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700">
                        Refresh Page
                    </button>
                    <button onclick="this.closest('.fixed').remove()" class="bg-gray-300 text-gray-700 px-4 py-2 rounded hover:bg-gray-400">
                        Dismiss
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }

    showFailsafeNotification() {
        const notification = document.createElement('div');
        notification.className = 'fixed top-0 left-0 right-0 bg-yellow-500 text-black px-4 py-2 text-center z-50';
        notification.innerHTML = `
            <strong>⚠️ Safe Mode Active</strong> - Advanced features are temporarily disabled. 
            <button onclick="location.reload()" class="underline ml-2">Refresh to restore full functionality</button>
        `;
        
        document.body.appendChild(notification);
    }

    showStaticErrorPage() {
        document.body.innerHTML = `
            <div class="min-h-screen flex items-center justify-center bg-gray-100">
                <div class="bg-white rounded-lg shadow-lg p-8 max-w-md text-center">
                    <div class="text-6xl mb-4">💥</div>
                    <h1 class="text-2xl font-bold text-gray-800 mb-4">Dashboard Error</h1>
                    <p class="text-gray-600 mb-6">
                        The dashboard encountered a critical error and cannot recover automatically.
                    </p>
                    <button onclick="location.reload()" 
                            class="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors">
                        Reload Dashboard
                    </button>
                </div>
            </div>
        `;
    }

    basicSearch(event) {
        const query = event.target.value.trim();
        if (query.length < 2) return;

        // Simple search implementation for failsafe mode
        fetch(`/api/search?q=${encodeURIComponent(query)}`)
            .then(response => response.json())
            .then(data => {
                const resultsContainer = document.getElementById('search-results');
                if (resultsContainer && data.results) {
                    resultsContainer.innerHTML = data.results.map(result => `
                        <div class="border-b border-gray-200 py-2">
                            <h4 class="font-medium">${result.title || 'Untitled'}</h4>
                            <p class="text-sm text-gray-600">${result.snippet || ''}</p>
                        </div>
                    `).join('');
                }
            })
            .catch(error => {
                console.error('Basic search failed:', error);
            });
    }

    /**
     * PERFORMANCE MONITORING
     */
    startPerformanceMonitoring() {
        // Core Web Vitals monitoring
        this.observeWebVitals();
        
        // Memory usage monitoring
        this.monitorMemoryUsage();
        
        // API performance monitoring
        this.monitorAPIPerformance();
        
        // Start metrics reporting
        this.startMetricsReporting();
    }

    observeWebVitals() {
        // First Contentful Paint
        new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
                if (entry.name === 'first-contentful-paint') {
                    this.performanceMetrics.fcpTime = entry.startTime;
                    console.log(`📊 FCP: ${entry.startTime.toFixed(2)}ms`);
                }
            }
        }).observe({ entryTypes: ['paint'] });

        // Largest Contentful Paint
        new PerformanceObserver((list) => {
            const entries = list.getEntries();
            const lastEntry = entries[entries.length - 1];
            this.performanceMetrics.lcpTime = lastEntry.startTime;
            console.log(`📊 LCP: ${lastEntry.startTime.toFixed(2)}ms`);
        }).observe({ entryTypes: ['largest-contentful-paint'] });

        // Cumulative Layout Shift
        new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
                if (!entry.hadRecentInput) {
                    this.performanceMetrics.clsScore += entry.value;
                }
            }
            console.log(`📊 CLS: ${this.performanceMetrics.clsScore.toFixed(4)}`);
        }).observe({ entryTypes: ['layout-shift'] });

        // First Input Delay
        new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
                this.performanceMetrics.fidTime = entry.processingStart - entry.startTime;
                console.log(`📊 FID: ${this.performanceMetrics.fidTime.toFixed(2)}ms`);
            }
        }).observe({ entryTypes: ['first-input'] });
    }

    monitorMemoryUsage() {
        if ('memory' in performance) {
            const updateMemory = () => {
                this.performanceMetrics.memoryUsage = performance.memory.usedJSHeapSize / 1048576; // MB
                
                // Warn if memory usage is high
                if (this.performanceMetrics.memoryUsage > 100) {
                    console.warn(`⚠️ High memory usage: ${this.performanceMetrics.memoryUsage.toFixed(2)}MB`);
                }
            };

            updateMemory();
            this.intervals.push(setInterval(updateMemory, 10000)); // Check every 10 seconds
        }
    }

    monitorAPIPerformance() {
        const originalFetch = window.fetch;
        const self = this;
        
        window.fetch = function(...args) {
            const startTime = performance.now();
            self.performanceMetrics.apiCallCount++;
            
            return originalFetch.apply(this, args).then(response => {
                const endTime = performance.now();
                const duration = endTime - startTime;
                
                console.log(`🌐 API Call: ${args[0]} - ${duration.toFixed(2)}ms`);
                
                // Track slow API calls
                if (duration > 2000) {
                    console.warn(`🐌 Slow API call: ${args[0]} - ${duration.toFixed(2)}ms`);
                }
                
                return response;
            });
        };
    }

    startMetricsReporting() {
        // Report metrics every 30 seconds
        this.intervals.push(setInterval(() => {
            this.reportMetrics();
        }, 30000));
    }

    reportMetrics() {
        const metrics = {
            ...this.performanceMetrics,
            cacheSize: this.cache.size,
            cacheHitRate: this.calculateCacheHitRate(),
            timestamp: Date.now()
        };

        console.log('📈 Performance Metrics:', metrics);
        
        // Store metrics in localStorage for analysis
        const storedMetrics = JSON.parse(localStorage.getItem('performanceMetrics') || '[]');
        storedMetrics.push(metrics);
        
        // Keep only last 100 entries
        if (storedMetrics.length > 100) {
            storedMetrics.splice(0, storedMetrics.length - 100);
        }
        
        localStorage.setItem('performanceMetrics', JSON.stringify(storedMetrics));
    }

    /**
     * CACHING SYSTEM
     */
    initializeCaching() {
        console.log('🗄️ Initializing intelligent caching system');
        
        // Setup cache cleanup
        this.intervals.push(setInterval(() => {
            this.cleanupCache();
        }, 60000)); // Cleanup every minute
    }

    async get(key, fetcher, options = {}) {
        const cacheKey = this.getCacheKey(key, options);
        const cached = this.cache.get(cacheKey);
        
        if (cached && !this.isCacheExpired(cached, options.ttl)) {
            console.log(`💾 Cache hit: ${key}`);
            this.performanceMetrics.cacheHitRate++;
            return cached.data;
        }

        console.log(`🌐 Cache miss: ${key}`);
        
        // Check for pending request to avoid duplication
        if (this.pendingRequests.has(cacheKey)) {
            console.log(`⏳ Deduplicating request: ${key}`);
            return this.pendingRequests.get(cacheKey);
        }

        // Make the request
        const promise = fetcher();
        this.pendingRequests.set(cacheKey, promise);

        try {
            const data = await promise;
            
            // Cache the result
            this.cache.set(cacheKey, {
                data,
                timestamp: Date.now(),
                key: cacheKey
            });
            
            // Cleanup if cache is too large
            if (this.cache.size > this.cacheConfig.maxSize) {
                this.evictOldestEntries();
            }
            
            return data;
        } finally {
            this.pendingRequests.delete(cacheKey);
        }
    }

    getCacheKey(key, options) {
        return `${key}_${JSON.stringify(options)}`;
    }

    isCacheExpired(cached, customTTL) {
        const ttl = customTTL || this.cacheConfig.ttl;
        return Date.now() - cached.timestamp > ttl;
    }

    cleanupCache() {
        const now = Date.now();
        let cleaned = 0;
        
        for (const [key, cached] of this.cache.entries()) {
            if (this.isCacheExpired(cached)) {
                this.cache.delete(key);
                cleaned++;
            }
        }
        
        if (cleaned > 0) {
            console.log(`🧹 Cleaned ${cleaned} expired cache entries`);
        }
    }

    evictOldestEntries() {
        const entries = Array.from(this.cache.entries());
        entries.sort((a, b) => a[1].timestamp - b[1].timestamp);
        
        const toRemove = entries.slice(0, Math.floor(this.cacheConfig.maxSize * 0.1));
        toRemove.forEach(([key]) => this.cache.delete(key));
        
        console.log(`🗑️ Evicted ${toRemove.length} oldest cache entries`);
    }

    calculateCacheHitRate() {
        const totalRequests = this.performanceMetrics.apiCallCount;
        const hits = this.performanceMetrics.cacheHitRate;
        return totalRequests > 0 ? (hits / totalRequests) * 100 : 0;
    }

    /**
     * REQUEST DEDUPLICATION
     */
    initializeRequestDeduplication() {
        console.log('🔄 Initializing request deduplication');
    }

    /**
     * ENHANCED SEARCH SYSTEM
     */
    initializeSearchSystem() {
        console.log('🔍 Initializing enhanced search system');
        
        this.setupSearchInput();
        this.setupSearchSuggestions();
        this.setupSearchFilters();
        this.setupSearchHistory();
    }

    setupSearchInput() {
        const searchInputs = document.querySelectorAll('#globalSearch, #advancedSearchInput');
        
        searchInputs.forEach(input => {
            if (!input) return;
            
            this.addEventListener(input, 'input', (e) => {
                this.handleSearchInput(e.target.value, input);
            });
            
            this.addEventListener(input, 'keydown', (e) => {
                this.handleSearchKeydown(e, input);
            });
            
            this.addEventListener(input, 'focus', () => {
                this.showSearchSuggestions(input);
            });
        });
    }

    handleSearchInput(query, input) {
        this.searchState.query = query;
        
        // Clear existing debounce
        if (this.searchState.debounceTimeout) {
            clearTimeout(this.searchState.debounceTimeout);
        }
        
        // Debounced search
        this.searchState.debounceTimeout = setTimeout(() => {
            this.performSearch(query, input);
        }, this.searchState.debounceDelay);
        
        // Real-time suggestions
        this.updateSearchSuggestions(query, input);
    }

    handleSearchKeydown(e, input) {
        const suggestionsContainer = this.getSuggestionsContainer(input);
        const suggestions = suggestionsContainer?.querySelectorAll('.search-suggestion');
        
        if (e.key === 'ArrowDown' && suggestions?.length > 0) {
            e.preventDefault();
            this.focusNextSuggestion(suggestions, 0);
        } else if (e.key === 'Escape') {
            this.hideSuggestions(input);
        } else if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.performSearch(input.value, input);
        }
    }

    async performSearch(query, input) {
        if (!query.trim()) {
            this.clearSearchResults(input);
            return;
        }
        
        console.log(`🔍 Performing search: "${query}"`);
        
        try {
            const cacheKey = `search_${query}_${this.searchState.filters.join(',')}`;
            
            const results = await this.get(cacheKey, async () => {
                const params = new URLSearchParams();
                params.append('q', query);
                
                if (this.searchState.filters.length > 0) {
                    params.append('filters', this.searchState.filters.join(','));
                }
                
                const response = await fetch(`/api/search?${params}`);
                if (!response.ok) throw new Error('Search failed');
                
                return response.json();
            }, { ttl: 60000 }); // Cache search results for 1 minute
            
            this.displaySearchResults(results, query, input);
            this.addToSearchHistory(query);
            
        } catch (error) {
            console.error('Search error:', error);
            this.showSearchError(input, 'Search failed. Please try again.');
        }
    }

    setupSearchSuggestions() {
        // Create suggestion containers if they don't exist
        const searchInputs = document.querySelectorAll('#globalSearch, #advancedSearchInput');
        
        searchInputs.forEach(input => {
            if (!input) return;
            
            let suggestionsContainer = this.getSuggestionsContainer(input);
            
            if (!suggestionsContainer) {
                suggestionsContainer = document.createElement('div');
                suggestionsContainer.className = 'search-suggestions absolute z-50 bg-white dark:bg-slate-800 rounded-lg shadow-lg max-h-64 overflow-y-auto hidden';
                suggestionsContainer.id = `${input.id}Suggestions`;
                
                input.parentElement.style.position = 'relative';
                input.parentElement.appendChild(suggestionsContainer);
            }
        });
    }

    updateSearchSuggestions(query, input) {
        if (!query.trim()) {
            this.hideSuggestions(input);
            return;
        }
        
        const suggestions = this.generateSuggestions(query);
        this.renderSuggestions(suggestions, input);
    }

    generateSuggestions(query) {
        const suggestions = [];
        const queryLower = query.toLowerCase();
        
        // Search history matches
        const historyMatches = this.searchState.history
            .filter(item => item.toLowerCase().includes(queryLower))
            .slice(0, 3);
        
        suggestions.push(...historyMatches.map(item => ({
            type: 'history',
            text: item,
            icon: '🕒'
        })));
        
        // Smart suggestions based on content
        const smartSuggestions = this.getSmartSuggestions(query);
        suggestions.push(...smartSuggestions);
        
        return suggestions.slice(0, 8); // Limit to 8 suggestions
    }

    getSmartSuggestions(query) {
        // This would ideally be populated by analyzing user's notes and common patterns
        const commonPatterns = [
            { pattern: /meeting/i, suggestions: ['meeting notes', 'meeting summary', 'meeting action items'] },
            { pattern: /project/i, suggestions: ['project planning', 'project update', 'project documentation'] },
            { pattern: /idea/i, suggestions: ['idea brainstorm', 'idea validation', 'idea implementation'] },
            { pattern: /todo/i, suggestions: ['todo list', 'todo priority', 'todo completed'] }
        ];
        
        const suggestions = [];
        
        for (const { pattern, suggestions: patternSuggestions } of commonPatterns) {
            if (pattern.test(query)) {
                suggestions.push(...patternSuggestions.map(text => ({
                    type: 'smart',
                    text,
                    icon: '💡'
                })));
                break;
            }
        }
        
        return suggestions.slice(0, 3);
    }

    renderSuggestions(suggestions, input) {
        const container = this.getSuggestionsContainer(input);
        if (!container) return;
        
        if (suggestions.length === 0) {
            container.classList.add('hidden');
            return;
        }
        
        container.innerHTML = suggestions.map((suggestion, index) => `
            <div class="search-suggestion flex items-center px-3 py-2 hover:bg-slate-100 dark:hover:bg-slate-700 cursor-pointer" 
                 data-suggestion="${suggestion.text}" 
                 data-index="${index}">
                <span class="mr-2">${suggestion.icon}</span>
                <span class="flex-1">${this.highlightQuery(suggestion.text, this.searchState.query)}</span>
                <span class="text-xs text-slate-400 ml-2">${suggestion.type}</span>
            </div>
        `).join('');
        
        // Add click handlers
        container.querySelectorAll('.search-suggestion').forEach(item => {
            this.addEventListener(item, 'click', () => {
                const suggestionText = item.dataset.suggestion;
                input.value = suggestionText;
                this.hideSuggestions(input);
                this.performSearch(suggestionText, input);
            });
        });
        
        container.classList.remove('hidden');
    }

    highlightQuery(text, query) {
        if (!query.trim()) return text;
        
        const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
        return text.replace(regex, '<mark class="bg-yellow-200 dark:bg-yellow-800">$1</mark>');
    }

    getSuggestionsContainer(input) {
        return document.getElementById(`${input.id}Suggestions`) || 
               input.parentElement.querySelector('.search-suggestions');
    }

    hideSuggestions(input) {
        const container = this.getSuggestionsContainer(input);
        if (container) {
            container.classList.add('hidden');
        }
    }

    showSearchSuggestions(input) {
        if (input.value.trim()) {
            this.updateSearchSuggestions(input.value, input);
        }
    }

    addToSearchHistory(query) {
        if (!query.trim() || this.searchState.history.includes(query)) return;
        
        this.searchState.history.unshift(query);
        this.searchState.history = this.searchState.history.slice(0, 20);
        
        localStorage.setItem('searchHistory', JSON.stringify(this.searchState.history));
    }

    setupSearchFilters() {
        const filterButtons = document.querySelectorAll('.search-filter');
        
        filterButtons.forEach(button => {
            this.addEventListener(button, 'click', () => {
                this.toggleSearchFilter(button);
            });
        });
    }

    toggleSearchFilter(button) {
        const filter = button.dataset.filter || button.textContent.toLowerCase().trim();
        
        button.classList.toggle('active');
        
        if (button.classList.contains('active')) {
            if (!this.searchState.filters.includes(filter)) {
                this.searchState.filters.push(filter);
            }
        } else {
            this.searchState.filters = this.searchState.filters.filter(f => f !== filter);
        }
        
        // Re-run search if there's an active query
        if (this.searchState.query.trim()) {
            const activeInput = document.querySelector('#globalSearch:focus, #advancedSearchInput:focus') ||
                              document.getElementById('advancedSearchInput');
            if (activeInput) {
                this.performSearch(this.searchState.query, activeInput);
            }
        }
    }

    setupSearchHistory() {
        // Load search history from localStorage
        this.searchState.history = JSON.parse(localStorage.getItem('searchHistory') || '[]');
    }

    displaySearchResults(results, query, input) {
        console.log(`📋 Displaying ${results.length || 0} search results for "${query}"`);
        
        // This would integrate with the existing search results display
        if (typeof window.displaySearchResults === 'function') {
            window.displaySearchResults(results, query);
        }
    }

    clearSearchResults(input) {
        // Clear results display
        const resultsContainer = document.getElementById('searchResultsContainer');
        if (resultsContainer) {
            resultsContainer.classList.add('hidden');
        }
    }

    showSearchError(input, message) {
        console.error('Search error:', message);
        // This would show an error message in the UI
    }

    /**
     * KEYBOARD SHORTCUTS SYSTEM
     */
    initializeKeyboardShortcuts() {
        console.log('⌨️ Initializing comprehensive keyboard shortcuts');
        
        this.registerShortcuts();
        this.setupShortcutHelp();
        this.bindShortcutHandler();
    }

    registerShortcuts() {
        // Navigation shortcuts
        this.shortcuts.set('ctrl+k,cmd+k', {
            description: 'Global search',
            handler: () => this.focusGlobalSearch()
        });
        
        this.shortcuts.set('ctrl+n,cmd+n', {
            description: 'New note',
            handler: () => this.focusNewNote()
        });
        
        this.shortcuts.set('ctrl+s,cmd+s', {
            description: 'Save current note',
            handler: () => this.saveCurrentNote()
        });
        
        this.shortcuts.set('ctrl+/,cmd+/', {
            description: 'Show keyboard shortcuts',
            handler: () => this.toggleShortcutHelp()
        });
        
        // View switching
        this.shortcuts.set('g h', {
            description: 'Go to dashboard home',
            handler: () => this.goToView('dashboard')
        });
        
        this.shortcuts.set('g s', {
            description: 'Go to search',
            handler: () => this.goToView('search')
        });
        
        this.shortcuts.set('g n', {
            description: 'Go to notes',
            handler: () => this.goToView('notes')
        });
        
        this.shortcuts.set('g a', {
            description: 'Go to analytics',
            handler: () => this.goToView('analytics')
        });
        
        // Quick actions
        this.shortcuts.set('a', {
            description: 'Add new note',
            handler: () => this.quickAddNote(),
            condition: () => !this.isInputFocused()
        });
        
        this.shortcuts.set('r', {
            description: 'Refresh current view',
            handler: () => this.refreshCurrentView(),
            condition: () => !this.isInputFocused()
        });
        
        this.shortcuts.set('escape', {
            description: 'Cancel current action',
            handler: () => this.cancelCurrentAction()
        });
        
        // Search shortcuts
        this.shortcuts.set('ctrl+shift+f,cmd+shift+f', {
            description: 'Advanced search',
            handler: () => this.openAdvancedSearch()
        });
        
        this.shortcuts.set('ctrl+shift+r,cmd+shift+r', {
            description: 'Recent notes search',
            handler: () => this.focusRecentSearch()
        });
    }

    bindShortcutHandler() {
        this.addEventListener(document, 'keydown', (e) => {
            this.handleKeyboardShortcut(e);
        });
    }

    handleKeyboardShortcut(e) {
        // Build the key combination string
        const keys = [];
        
        if (e.ctrlKey) keys.push('ctrl');
        if (e.metaKey) keys.push('cmd');
        if (e.shiftKey) keys.push('shift');
        if (e.altKey) keys.push('alt');
        
        // Add the main key
        const mainKey = e.key.toLowerCase();
        keys.push(mainKey === ' ' ? 'space' : mainKey);
        
        const combination = keys.join('+');
        
        // Check for matching shortcuts
        for (const [shortcut, config] of this.shortcuts.entries()) {
            const variants = shortcut.split(',');
            
            if (variants.includes(combination)) {
                // Check condition if exists
                if (config.condition && !config.condition()) {
                    continue;
                }
                
                e.preventDefault();
                config.handler();
                break;
            }
        }
    }

    // Shortcut handlers
    focusGlobalSearch() {
        const globalSearch = document.getElementById('globalSearch');
        if (globalSearch) {
            globalSearch.focus();
            globalSearch.select();
        }
    }

    focusNewNote() {
        const noteInput = document.getElementById('note');
        if (noteInput) {
            noteInput.focus();
        }
    }

    saveCurrentNote() {
        const form = document.getElementById('quickCaptureForm');
        if (form) {
            form.requestSubmit();
        }
    }

    goToView(viewName) {
        if (typeof showView === 'function') {
            showView(viewName);
        }
    }

    quickAddNote() {
        this.goToView('dashboard');
        setTimeout(() => this.focusNewNote(), 100);
    }

    refreshCurrentView() {
        // Determine current view and refresh it
        const activeView = document.querySelector('.view:not(.hidden)');
        if (activeView) {
            const viewId = activeView.id;
            console.log(`🔄 Refreshing view: ${viewId}`);
            
            // Trigger refresh based on view
            if (viewId.includes('dashboard')) {
                this.refreshDashboardData();
            } else if (viewId.includes('search')) {
                const searchInput = document.getElementById('advancedSearchInput');
                if (searchInput?.value.trim()) {
                    this.performSearch(searchInput.value, searchInput);
                }
            }
        }
    }

    async refreshDashboardData() {
        // Clear relevant caches
        for (const key of this.cache.keys()) {
            if (key.includes('analytics') || key.includes('recent') || key.includes('activity')) {
                this.cache.delete(key);
            }
        }
        
        // Reload dashboard data
        if (typeof loadTodaysActivity === 'function') {
            loadTodaysActivity();
        }
        if (typeof loadRecentNotes === 'function') {
            loadRecentNotes();
        }
    }

    cancelCurrentAction() {
        // Hide any open modals or suggestions
        const modals = document.querySelectorAll('.modal:not(.hidden)');
        modals.forEach(modal => modal.classList.add('hidden'));
        
        // Hide search suggestions
        const suggestionContainers = document.querySelectorAll('.search-suggestions:not(.hidden)');
        suggestionContainers.forEach(container => container.classList.add('hidden'));
        
        // Cancel any pending searches
        if (this.searchState.debounceTimeout) {
            clearTimeout(this.searchState.debounceTimeout);
            this.searchState.debounceTimeout = null;
        }
    }

    openAdvancedSearch() {
        this.goToView('search');
        setTimeout(() => {
            const advancedSearch = document.getElementById('advancedSearchInput');
            if (advancedSearch) {
                advancedSearch.focus();
            }
        }, 100);
    }

    focusRecentSearch() {
        const recentSearch = document.getElementById('recentSearch');
        if (recentSearch) {
            recentSearch.focus();
        }
    }

    isInputFocused() {
        const activeElement = document.activeElement;
        return activeElement && (
            activeElement.tagName === 'INPUT' ||
            activeElement.tagName === 'TEXTAREA' ||
            activeElement.contentEditable === 'true'
        );
    }

    setupShortcutHelp() {
        // Create keyboard shortcuts help modal
        const helpModal = document.createElement('div');
        helpModal.id = 'keyboardShortcutsHelp';
        helpModal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 hidden';
        
        helpModal.innerHTML = `
            <div class="bg-white dark:bg-slate-800 rounded-lg p-6 max-w-2xl max-h-96 overflow-y-auto">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-lg font-semibold">Keyboard Shortcuts</h3>
                    <button id="closeShortcutHelp" class="text-gray-500 hover:text-gray-700">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                        </svg>
                    </button>
                </div>
                <div id="shortcutsList" class="space-y-2"></div>
            </div>
        `;
        
        document.body.appendChild(helpModal);
        
        // Bind close handler
        this.addEventListener(document.getElementById('closeShortcutHelp'), 'click', () => {
            this.hideShortcutHelp();
        });
        
        this.addEventListener(helpModal, 'click', (e) => {
            if (e.target === helpModal) {
                this.hideShortcutHelp();
            }
        });
    }

    toggleShortcutHelp() {
        if (this.shortcutHelpVisible) {
            this.hideShortcutHelp();
        } else {
            this.showShortcutHelp();
        }
    }

    showShortcutHelp() {
        const helpModal = document.getElementById('keyboardShortcutsHelp');
        const shortcutsList = document.getElementById('shortcutsList');
        
        if (!helpModal || !shortcutsList) return;
        
        // Populate shortcuts list
        shortcutsList.innerHTML = Array.from(this.shortcuts.entries())
            .map(([shortcut, config]) => {
                const displayShortcut = shortcut.split(',')[0]
                    .replace(/ctrl/g, '⌘')
                    .replace(/cmd/g, '⌘')
                    .replace(/shift/g, '⇧')
                    .replace(/alt/g, '⌥')
                    .replace(/\+/g, ' + ')
                    .toUpperCase();
                
                return `
                    <div class="flex justify-between items-center py-2 border-b border-gray-200 dark:border-gray-600">
                        <span class="text-sm">${config.description}</span>
                        <kbd class="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs font-mono">${displayShortcut}</kbd>
                    </div>
                `;
            }).join('');
        
        helpModal.classList.remove('hidden');
        this.shortcutHelpVisible = true;
    }

    hideShortcutHelp() {
        const helpModal = document.getElementById('keyboardShortcutsHelp');
        if (helpModal) {
            helpModal.classList.add('hidden');
        }
        this.shortcutHelpVisible = false;
    }

    /**
     * DRAG & DROP FUNCTIONALITY
     */
    initializeDragAndDrop() {
        console.log('🖱️ Initializing enhanced drag & drop functionality');
        
        this.setupFileDragAndDrop();
        this.setupNoteDragAndDrop();
        this.setupWidgetDragAndDrop();
    }

    setupFileDragAndDrop() {
        // Enhanced file drop zones
        const dropZones = document.querySelectorAll('[data-drop-zone], .file-drop-zone');
        
        dropZones.forEach(zone => {
            this.setupDropZone(zone);
        });
        
        // Global drag and drop handlers
        this.addEventListener(document, 'dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'copy';
            this.showDropIndicators();
        });
        
        this.addEventListener(document, 'dragleave', (e) => {
            if (!e.relatedTarget) {
                this.hideDropIndicators();
            }
        });
        
        this.addEventListener(document, 'drop', (e) => {
            e.preventDefault();
            this.hideDropIndicators();
            
            if (e.dataTransfer.files.length > 0) {
                this.handleFilesDrop(e.dataTransfer.files, e.target);
            }
        });
    }

    setupDropZone(zone) {
        this.addEventListener(zone, 'dragenter', (e) => {
            e.preventDefault();
            zone.classList.add('drag-over');
        });
        
        this.addEventListener(zone, 'dragleave', (e) => {
            e.preventDefault();
            if (!zone.contains(e.relatedTarget)) {
                zone.classList.remove('drag-over');
            }
        });
        
        this.addEventListener(zone, 'drop', (e) => {
            e.preventDefault();
            zone.classList.remove('drag-over');
            
            if (e.dataTransfer.files.length > 0) {
                this.handleFilesDrop(e.dataTransfer.files, zone);
            }
        });
    }

    async handleFilesDrop(files, dropTarget) {
        console.log(`📁 Files dropped: ${files.length} files`);
        
        const fileArray = Array.from(files);
        const validFiles = fileArray.filter(file => this.isValidFile(file));
        
        if (validFiles.length === 0) {
            this.showNotification('No valid files to upload', 'warning');
            return;
        }
        
        if (validFiles.length !== fileArray.length) {
            this.showNotification(`${fileArray.length - validFiles.length} files skipped (invalid format)`, 'warning');
        }
        
        // Show upload progress
        this.showUploadProgress(validFiles);
        
        try {
            const results = await this.uploadFiles(validFiles);
            this.showNotification(`Successfully uploaded ${results.length} files`, 'success');
            
            // Refresh relevant data
            this.refreshDashboardData();
        } catch (error) {
            console.error('File upload error:', error);
            this.showNotification('File upload failed', 'error');
        } finally {
            this.hideUploadProgress();
        }
    }

    isValidFile(file) {
        const allowedTypes = [
            'text/plain',
            'text/markdown',
            'application/pdf',
            'image/jpeg',
            'image/png',
            'image/gif',
            'audio/mpeg',
            'audio/wav',
            'audio/ogg'
        ];
        
        const maxSize = 50 * 1024 * 1024; // 50MB
        
        return allowedTypes.includes(file.type) && file.size <= maxSize;
    }

    async uploadFiles(files) {
        const results = [];
        
        for (const file of files) {
            try {
                const formData = new FormData();
                formData.append('file', file);
                
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) throw new Error('Upload failed');
                
                const result = await response.json();
                results.push(result);
                
                this.updateUploadProgress(results.length, files.length);
            } catch (error) {
                console.error(`Failed to upload ${file.name}:`, error);
            }
        }
        
        return results;
    }

    showDropIndicators() {
        document.body.classList.add('drag-active');
        
        // Show drop zones
        const dropZones = document.querySelectorAll('[data-drop-zone]');
        dropZones.forEach(zone => zone.classList.add('drop-zone-visible'));
    }

    hideDropIndicators() {
        document.body.classList.remove('drag-active');
        
        // Hide drop zones
        const dropZones = document.querySelectorAll('[data-drop-zone]');
        dropZones.forEach(zone => {
            zone.classList.remove('drop-zone-visible', 'drag-over');
        });
    }

    showUploadProgress(files) {
        const progressModal = document.createElement('div');
        progressModal.id = 'uploadProgressModal';
        progressModal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        
        progressModal.innerHTML = `
            <div class="bg-white dark:bg-slate-800 rounded-lg p-6 w-96">
                <h3 class="text-lg font-semibold mb-4">Uploading Files</h3>
                <div class="space-y-2">
                    <div class="flex justify-between text-sm">
                        <span>Progress:</span>
                        <span id="uploadProgressText">0 / ${files.length}</span>
                    </div>
                    <div class="w-full bg-gray-200 rounded-full h-2">
                        <div id="uploadProgressBar" class="bg-blue-600 h-2 rounded-full transition-all duration-300" style="width: 0%"></div>
                    </div>
                </div>
                <div class="mt-4">
                    <div id="uploadFileList" class="text-sm text-gray-600 max-h-32 overflow-y-auto">
                        ${files.map(file => `<div class="py-1">${file.name}</div>`).join('')}
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(progressModal);
    }

    updateUploadProgress(completed, total) {
        const progressText = document.getElementById('uploadProgressText');
        const progressBar = document.getElementById('uploadProgressBar');
        
        if (progressText && progressBar) {
            progressText.textContent = `${completed} / ${total}`;
            progressBar.style.width = `${(completed / total) * 100}%`;
        }
    }

    hideUploadProgress() {
        const progressModal = document.getElementById('uploadProgressModal');
        if (progressModal) {
            progressModal.remove();
        }
    }

    setupNoteDragAndDrop() {
        // Enable dragging of note items for reordering
        const noteItems = document.querySelectorAll('.recent-note-item, .note-item');
        
        noteItems.forEach(item => {
            item.draggable = true;
            
            this.addEventListener(item, 'dragstart', (e) => {
                e.dataTransfer.setData('text/plain', item.dataset.noteId);
                item.classList.add('dragging');
            });
            
            this.addEventListener(item, 'dragend', () => {
                item.classList.remove('dragging');
            });
        });
    }

    setupWidgetDragAndDrop() {
        // Enable dashboard widget rearrangement (if applicable)
        const widgets = document.querySelectorAll('[data-widget]');
        
        widgets.forEach(widget => {
            const handle = widget.querySelector('.drag-handle');
            if (handle) {
                widget.draggable = true;
                
                this.addEventListener(widget, 'dragstart', (e) => {
                    e.dataTransfer.setData('text/plain', widget.dataset.widget);
                    widget.classList.add('dragging');
                });
                
                this.addEventListener(widget, 'dragend', () => {
                    widget.classList.remove('dragging');
                });
            }
        });
    }

    /**
     * VIRTUAL SCROLLING
     */
    initializeVirtualScrolling() {
        console.log('📜 Initializing virtual scrolling for performance');
        
        // Find scrollable containers that would benefit from virtualization
        const scrollableContainers = document.querySelectorAll('[data-virtual-scroll]');
        
        scrollableContainers.forEach(container => {
            this.setupVirtualScrolling(container);
        });
    }

    setupVirtualScrolling(container) {
        const itemHeight = parseInt(container.dataset.itemHeight) || 60;
        const bufferSize = parseInt(container.dataset.bufferSize) || 10;
        
        const virtualScroller = {
            container,
            itemHeight,
            bufferSize,
            items: [],
            visibleStart: 0,
            visibleEnd: 0,
            scrollTop: 0
        };
        
        this.virtualScrollers.set(container, virtualScroller);
        
        this.addEventListener(container, 'scroll', () => {
            this.updateVirtualScroll(virtualScroller);
        });
    }

    updateVirtualScroll(scroller) {
        const { container, itemHeight, bufferSize, items } = scroller;
        
        if (items.length === 0) return;
        
        const scrollTop = container.scrollTop;
        const containerHeight = container.clientHeight;
        
        const visibleStart = Math.max(0, Math.floor(scrollTop / itemHeight) - bufferSize);
        const visibleEnd = Math.min(items.length, Math.ceil((scrollTop + containerHeight) / itemHeight) + bufferSize);
        
        if (visibleStart !== scroller.visibleStart || visibleEnd !== scroller.visibleEnd) {
            scroller.visibleStart = visibleStart;
            scroller.visibleEnd = visibleEnd;
            
            this.renderVirtualItems(scroller);
        }
    }

    renderVirtualItems(scroller) {
        const { container, items, visibleStart, visibleEnd, itemHeight } = scroller;
        
        // Clear existing content
        container.innerHTML = '';
        
        // Create spacer for items before visible range
        if (visibleStart > 0) {
            const topSpacer = document.createElement('div');
            topSpacer.style.height = `${visibleStart * itemHeight}px`;
            container.appendChild(topSpacer);
        }
        
        // Render visible items
        for (let i = visibleStart; i < visibleEnd; i++) {
            if (items[i]) {
                const itemElement = this.createVirtualItem(items[i], i);
                container.appendChild(itemElement);
            }
        }
        
        // Create spacer for items after visible range
        if (visibleEnd < items.length) {
            const bottomSpacer = document.createElement('div');
            bottomSpacer.style.height = `${(items.length - visibleEnd) * itemHeight}px`;
            container.appendChild(bottomSpacer);
        }
    }

    createVirtualItem(item, index) {
        // This would be customized based on the item type
        const element = document.createElement('div');
        element.className = 'virtual-item';
        element.style.height = `${this.virtualScrollers.get(element.parentElement)?.itemHeight || 60}px`;
        element.innerHTML = `<div class="p-3">${item.title || `Item ${index}`}</div>`;
        return element;
    }

    /**
     * SMART FEATURES
     */
    initializeSmartFeatures() {
        console.log('🧠 Initializing smart features');
        
        this.setupContextualRecommendations();
        this.setupBehaviorTracking();
        this.setupSmartSuggestions();
        this.setupAutoSave();
    }

    setupContextualRecommendations() {
        // Analyze user behavior and provide contextual recommendations
        this.intervals.push(setInterval(() => {
            this.updateContextualRecommendations();
        }, 30000)); // Update every 30 seconds
    }

    updateContextualRecommendations() {
        if (!this.smartFeatures.contextualRecommendations) return;
        
        const currentTime = new Date();
        const hour = currentTime.getHours();
        
        // Time-based recommendations
        let recommendation = null;
        
        if (hour >= 9 && hour < 12) {
            recommendation = {
                type: 'time',
                message: 'Good morning! Consider reviewing your daily goals.',
                action: 'search',
                query: 'goals today'
            };
        } else if (hour >= 17 && hour < 19) {
            recommendation = {
                type: 'time',
                message: 'End of day - time to capture key insights.',
                action: 'new_note',
                template: 'daily_reflection'
            };
        }
        
        if (recommendation) {
            this.showSmartRecommendation(recommendation);
        }
    }

    showSmartRecommendation(recommendation) {
        // Only show if user hasn't dismissed recently
        const lastDismissal = localStorage.getItem(`recommendation_dismissed_${recommendation.type}`);
        const oneHour = 60 * 60 * 1000;
        
        if (lastDismissal && Date.now() - parseInt(lastDismissal) < oneHour) {
            return;
        }
        
        const recommendationElement = document.createElement('div');
        recommendationElement.className = 'smart-recommendation fixed bottom-4 right-4 bg-blue-500 text-white p-4 rounded-lg shadow-lg max-w-sm z-40';
        
        recommendationElement.innerHTML = `
            <div class="flex items-start">
                <div class="flex-1">
                    <div class="text-sm font-medium">💡 Smart Suggestion</div>
                    <div class="text-sm mt-1">${recommendation.message}</div>
                </div>
                <button class="ml-2 text-white hover:text-gray-200" onclick="this.parentElement.parentElement.remove()">
                    <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                    </svg>
                </button>
            </div>
            <div class="mt-3 flex space-x-2">
                <button class="bg-white bg-opacity-20 hover:bg-opacity-30 px-3 py-1 rounded text-sm" 
                        onclick="this.closest('.smart-recommendation').remove()">
                    Maybe Later
                </button>
                <button class="bg-white text-blue-500 hover:bg-gray-100 px-3 py-1 rounded text-sm font-medium" 
                        onclick="handleRecommendationAction('${recommendation.action}', '${recommendation.query || ''}')">
                    Do It
                </button>
            </div>
        `;
        
        document.body.appendChild(recommendationElement);
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (recommendationElement.parentNode) {
                recommendationElement.remove();
            }
        }, 10000);
    }

    setupBehaviorTracking() {
        if (!this.smartFeatures.behaviorTracking) return;
        
        // Track user interactions for better recommendations
        const behaviorData = JSON.parse(localStorage.getItem('userBehavior') || '{}');
        
        // Track view changes
        const originalShowView = window.showView;
        if (originalShowView) {
            window.showView = (viewName) => {
                behaviorData.viewChanges = behaviorData.viewChanges || [];
                behaviorData.viewChanges.push({
                    view: viewName,
                    timestamp: Date.now()
                });
                
                // Keep only last 100 entries
                if (behaviorData.viewChanges.length > 100) {
                    behaviorData.viewChanges = behaviorData.viewChanges.slice(-50);
                }
                
                localStorage.setItem('userBehavior', JSON.stringify(behaviorData));
                return originalShowView(viewName);
            };
        }
        
        // Track search patterns
        this.addEventListener(document, 'search-performed', (e) => {
            behaviorData.searches = behaviorData.searches || [];
            behaviorData.searches.push({
                query: e.detail.query,
                timestamp: Date.now()
            });
            
            if (behaviorData.searches.length > 50) {
                behaviorData.searches = behaviorData.searches.slice(-25);
            }
            
            localStorage.setItem('userBehavior', JSON.stringify(behaviorData));
        });
    }

    setupSmartSuggestions() {
        // Implement intelligent content suggestions based on context
        this.intervals.push(setInterval(() => {
            this.updateSmartSuggestions();
        }, 60000)); // Update every minute
    }

    updateSmartSuggestions() {
        if (!this.smartFeatures.smartSuggestions) return;
        
        const noteInput = document.getElementById('note');
        if (!noteInput || !noteInput.value.trim()) return;
        
        const content = noteInput.value;
        const suggestions = this.generateContentSuggestions(content);
        
        if (suggestions.length > 0) {
            this.showContentSuggestions(suggestions);
        }
    }

    generateContentSuggestions(content) {
        const suggestions = [];
        const contentLower = content.toLowerCase();
        
        // Meeting-related suggestions
        if (contentLower.includes('meeting') && !contentLower.includes('action items')) {
            suggestions.push({
                type: 'template',
                text: 'Add action items section',
                template: '\n\n## Action Items\n- [ ] \n- [ ] \n- [ ] '
            });
        }
        
        // Todo-related suggestions
        if (contentLower.includes('todo') || contentLower.includes('task')) {
            if (!contentLower.includes('due date')) {
                suggestions.push({
                    type: 'enhancement',
                    text: 'Add due dates to tasks',
                    template: ' (due: YYYY-MM-DD)'
                });
            }
        }
        
        // Project-related suggestions
        if (contentLower.includes('project') && content.length > 100) {
            suggestions.push({
                type: 'organization',
                text: 'Consider breaking into sections',
                template: '\n\n## Overview\n\n## Next Steps\n\n## Resources\n'
            });
        }
        
        return suggestions.slice(0, 3); // Limit to 3 suggestions
    }

    showContentSuggestions(suggestions) {
        // Remove existing suggestions
        const existing = document.getElementById('contentSuggestions');
        if (existing) existing.remove();
        
        const suggestionsContainer = document.createElement('div');
        suggestionsContainer.id = 'contentSuggestions';
        suggestionsContainer.className = 'content-suggestions absolute z-10 bg-white dark:bg-slate-800 border rounded-lg shadow-lg p-3 mt-1';
        
        suggestionsContainer.innerHTML = `
            <div class="text-xs text-gray-500 mb-2">💡 Suggestions:</div>
            ${suggestions.map((suggestion, index) => `
                <button class="suggestion-item block w-full text-left px-2 py-1 hover:bg-gray-100 dark:hover:bg-slate-700 rounded text-sm"
                        data-template="${suggestion.template}"
                        data-index="${index}">
                    ${suggestion.text}
                </button>
            `).join('')}
        `;
        
        const noteInput = document.getElementById('note');
        if (noteInput) {
            noteInput.parentElement.style.position = 'relative';
            noteInput.parentElement.appendChild(suggestionsContainer);
            
            // Add click handlers
            suggestionsContainer.querySelectorAll('.suggestion-item').forEach(item => {
                this.addEventListener(item, 'click', () => {
                    const template = item.dataset.template;
                    noteInput.value += template;
                    suggestionsContainer.remove();
                    noteInput.focus();
                });
            });
            
            // Auto-remove after 10 seconds
            setTimeout(() => {
                if (suggestionsContainer.parentNode) {
                    suggestionsContainer.remove();
                }
            }, 10000);
        }
    }

    setupAutoSave() {
        // Enhanced auto-save with intelligent timing
        const autoSaveInputs = document.querySelectorAll('#note, #tags, [data-autosave]');
        
        autoSaveInputs.forEach(input => {
            let autoSaveTimeout;
            
            this.addEventListener(input, 'input', () => {
                if (autoSaveTimeout) clearTimeout(autoSaveTimeout);
                
                // Intelligent delay based on user typing pattern
                const delay = this.calculateAutoSaveDelay(input);
                
                autoSaveTimeout = setTimeout(() => {
                    this.performAutoSave(input);
                }, delay);
            });
        });
    }

    calculateAutoSaveDelay(input) {
        // Base delay
        let delay = 2000;
        
        // Reduce delay for longer content
        if (input.value.length > 500) {
            delay = 1000;
        }
        
        // Increase delay if user is typing rapidly
        const now = Date.now();
        input.lastInputTime = input.lastInputTime || now;
        const timeSinceLastInput = now - input.lastInputTime;
        
        if (timeSinceLastInput < 500) {
            delay = 3000; // Give more time if typing rapidly
        }
        
        input.lastInputTime = now;
        return delay;
    }

    performAutoSave(input) {
        if (!this.smartFeatures.autoSaveEnabled) return;
        
        const content = input.value.trim();
        if (!content) return;
        
        const saveData = {
            content,
            timestamp: Date.now(),
            inputId: input.id
        };
        
        try {
            localStorage.setItem(`autosave_${input.id}`, JSON.stringify(saveData));
            this.showAutoSaveIndicator(input, 'saved');
        } catch (error) {
            console.error('Auto-save failed:', error);
            this.showAutoSaveIndicator(input, 'error');
        }
    }

    showAutoSaveIndicator(input, status) {
        let indicator = input.parentElement.querySelector('.auto-save-indicator');
        
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.className = 'auto-save-indicator absolute -top-6 right-0 text-xs px-2 py-1 rounded transition-all duration-200';
            input.parentElement.style.position = 'relative';
            input.parentElement.appendChild(indicator);
        }
        
        switch (status) {
            case 'saved':
                indicator.textContent = '✓ Saved';
                indicator.className = indicator.className.replace(/bg-\w+-\d+/g, '') + ' bg-green-100 text-green-800';
                break;
            case 'error':
                indicator.textContent = '✗ Save failed';
                indicator.className = indicator.className.replace(/bg-\w+-\d+/g, '') + ' bg-red-100 text-red-800';
                break;
        }
        
        indicator.style.opacity = '1';
        
        setTimeout(() => {
            indicator.style.opacity = '0';
        }, 2000);
    }

    /**
     * DATA VISUALIZATION ENHANCEMENTS
     */
    initializeDataVisualization() {
        console.log('📊 Initializing interactive data visualization');
        
        this.enhanceAnalyticsCharts();
        this.setupInteractiveElements();
        this.setupDataExport();
    }

    enhanceAnalyticsCharts() {
        // Find chart containers and enhance them
        const chartContainers = document.querySelectorAll('[data-chart]');
        
        chartContainers.forEach(container => {
            this.addChartInteractivity(container);
        });
    }

    addChartInteractivity(container) {
        // Add hover effects and click handlers
        const chartElements = container.querySelectorAll('.chart-bar, .chart-point, .chart-segment');
        
        chartElements.forEach(element => {
            this.addEventListener(element, 'mouseenter', (e) => {
                this.showChartTooltip(e.target, container);
            });
            
            this.addEventListener(element, 'mouseleave', () => {
                this.hideChartTooltip(container);
            });
            
            this.addEventListener(element, 'click', (e) => {
                this.handleChartElementClick(e.target, container);
            });
        });
    }

    showChartTooltip(element, container) {
        const tooltip = this.getOrCreateTooltip(container);
        const data = element.dataset;
        
        tooltip.innerHTML = `
            <div class="bg-black bg-opacity-75 text-white px-2 py-1 rounded text-sm">
                ${data.label || 'Value'}: ${data.value || 'N/A'}
                ${data.date ? `<br><small>${data.date}</small>` : ''}
            </div>
        `;
        
        // Position tooltip
        const rect = element.getBoundingClientRect();
        tooltip.style.left = `${rect.left + rect.width / 2}px`;
        tooltip.style.top = `${rect.top - 40}px`;
        tooltip.style.opacity = '1';
    }

    hideChartTooltip(container) {
        const tooltip = container.querySelector('.chart-tooltip');
        if (tooltip) {
            tooltip.style.opacity = '0';
        }
    }

    getOrCreateTooltip(container) {
        let tooltip = container.querySelector('.chart-tooltip');
        
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.className = 'chart-tooltip absolute z-10 pointer-events-none transition-opacity duration-200';
            tooltip.style.opacity = '0';
            document.body.appendChild(tooltip);
        }
        
        return tooltip;
    }

    handleChartElementClick(element, container) {
        const data = element.dataset;
        console.log('Chart element clicked:', data);
        
        // Example: drill down into specific data
        if (data.date) {
            this.showDateDetails(data.date);
        }
    }

    showDateDetails(date) {
        // Show detailed information for a specific date
        console.log('Showing details for:', date);
        // Implementation would depend on available data
    }

    setupInteractiveElements() {
        // Add interactive controls to data displays
        const dataContainers = document.querySelectorAll('[data-interactive]');
        
        dataContainers.forEach(container => {
            this.addDataControls(container);
        });
    }

    addDataControls(container) {
        const controlsContainer = document.createElement('div');
        controlsContainer.className = 'data-controls flex items-center space-x-2 mb-4';
        
        controlsContainer.innerHTML = `
            <select class="data-timeframe text-sm border rounded px-2 py-1">
                <option value="week">This Week</option>
                <option value="month">This Month</option>
                <option value="quarter">This Quarter</option>
                <option value="year">This Year</option>
            </select>
            <select class="data-metric text-sm border rounded px-2 py-1">
                <option value="count">Count</option>
                <option value="growth">Growth</option>
                <option value="average">Average</option>
            </select>
            <button class="data-export text-sm bg-blue-500 text-white px-3 py-1 rounded hover:bg-blue-600">
                Export
            </button>
        `;
        
        container.insertBefore(controlsContainer, container.firstChild);
        
        // Bind event handlers
        const timeframeSelect = controlsContainer.querySelector('.data-timeframe');
        const metricSelect = controlsContainer.querySelector('.data-metric');
        const exportButton = controlsContainer.querySelector('.data-export');
        
        this.addEventListener(timeframeSelect, 'change', () => {
            this.updateDataDisplay(container, timeframeSelect.value, metricSelect.value);
        });
        
        this.addEventListener(metricSelect, 'change', () => {
            this.updateDataDisplay(container, timeframeSelect.value, metricSelect.value);
        });
        
        this.addEventListener(exportButton, 'click', () => {
            this.exportData(container);
        });
    }

    updateDataDisplay(container, timeframe, metric) {
        console.log(`Updating data display: ${timeframe}, ${metric}`);
        // Implementation would fetch new data based on selections
    }

    setupDataExport() {
        // Setup data export functionality
        this.addEventListener(document, 'click', (e) => {
            if (e.target.matches('.export-data, [data-export]')) {
                e.preventDefault();
                this.exportData(e.target.closest('[data-chart], [data-interactive]'));
            }
        });
    }

    exportData(container) {
        console.log('Exporting data from:', container);
        
        // Collect data from the container
        const data = this.extractDataFromContainer(container);
        
        if (data.length === 0) {
            this.showNotification('No data to export', 'warning');
            return;
        }
        
        // Export as CSV
        const csv = this.convertToCSV(data);
        this.downloadCSV(csv, `dashboard_export_${Date.now()}.csv`);
        
        this.showNotification('Data exported successfully', 'success');
    }

    extractDataFromContainer(container) {
        // Extract data based on container type
        const data = [];
        const chartType = container.dataset.chart;
        
        // This would be customized based on the actual chart implementation
        const dataElements = container.querySelectorAll('[data-value]');
        dataElements.forEach(element => {
            data.push({
                label: element.dataset.label || '',
                value: element.dataset.value || '',
                date: element.dataset.date || new Date().toISOString().split('T')[0]
            });
        });
        
        return data;
    }

    convertToCSV(data) {
        if (data.length === 0) return '';
        
        const headers = Object.keys(data[0]);
        const csvHeaders = headers.join(',');
        
        const csvRows = data.map(row => 
            headers.map(header => `"${row[header] || ''}"`).join(',')
        );
        
        return [csvHeaders, ...csvRows].join('\n');
    }

    downloadCSV(csv, filename) {
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        
        if (link.download !== undefined) {
            const url = URL.createObjectURL(blob);
            link.setAttribute('href', url);
            link.setAttribute('download', filename);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        }
    }

    /**
     * MEMORY MANAGEMENT
     */
    setupMemoryManagement() {
        console.log('🧹 Setting up memory management');
        
        // Track all event listeners for cleanup
        this.setupEventListenerTracking();
        
        // Setup garbage collection monitoring
        this.setupGarbageCollectionMonitoring();
        
        // Setup memory leak detection
        this.setupMemoryLeakDetection();
    }

    setupEventListenerTracking() {
        // Override addEventListener to track all listeners
        const originalAddEventListener = EventTarget.prototype.addEventListener;
        const self = this;
        
        EventTarget.prototype.addEventListener = function(type, listener, options) {
            self.eventListeners.push({
                target: this,
                type,
                listener,
                options
            });
            
            return originalAddEventListener.call(this, type, listener, options);
        };
    }

    setupGarbageCollectionMonitoring() {
        if ('memory' in performance) {
            this.intervals.push(setInterval(() => {
                const memInfo = performance.memory;
                const usedMB = memInfo.usedJSHeapSize / 1048576;
                
                if (usedMB > 200) {
                    console.warn(`⚠️ High memory usage detected: ${usedMB.toFixed(2)}MB`);
                    this.performMemoryCleanup();
                }
            }, 30000));
        }
    }

    setupMemoryLeakDetection() {
        // Detect potential memory leaks
        this.intervals.push(setInterval(() => {
            this.detectMemoryLeaks();
        }, 60000));
    }

    detectMemoryLeaks() {
        // Check for excessive event listeners
        if (this.eventListeners.length > 1000) {
            console.warn(`⚠️ High number of event listeners: ${this.eventListeners.length}`);
        }
        
        // Check for excessive cache entries
        if (this.cache.size > this.cacheConfig.maxSize * 2) {
            console.warn(`⚠️ Cache size exceeded safe limits: ${this.cache.size}`);
            this.cleanupCache();
        }
        
        // Check for excessive intervals/timeouts
        if (this.intervals.length > 50) {
            console.warn(`⚠️ High number of intervals: ${this.intervals.length}`);
        }
    }

    performMemoryCleanup() {
        console.log('🧹 Performing memory cleanup');
        
        // Clear expired cache entries
        this.cleanupCache();
        
        // Clear old performance metrics
        const storedMetrics = JSON.parse(localStorage.getItem('performanceMetrics') || '[]');
        if (storedMetrics.length > 50) {
            const recentMetrics = storedMetrics.slice(-25);
            localStorage.setItem('performanceMetrics', JSON.stringify(recentMetrics));
        }
        
        // Clear old behavior data
        const behaviorData = JSON.parse(localStorage.getItem('userBehavior') || '{}');
        if (behaviorData.viewChanges && behaviorData.viewChanges.length > 50) {
            behaviorData.viewChanges = behaviorData.viewChanges.slice(-25);
            localStorage.setItem('userBehavior', JSON.stringify(behaviorData));
        }
        
        // Force garbage collection if available
        if (window.gc && typeof window.gc === 'function') {
            window.gc();
        }
    }

    setupPageCleanup() {
        // Clean up resources when page unloads
        this.addEventListener(window, 'beforeunload', () => {
            this.cleanup();
        });
        
        // Also cleanup on visibility change (for SPA navigation)
        this.addEventListener(document, 'visibilitychange', () => {
            if (document.hidden) {
                this.performMemoryCleanup();
            }
        });
    }

    cleanup() {
        console.log('🧹 Performing dashboard cleanup');
        
        // Clear all timeouts
        this.timeouts.forEach(timeout => clearTimeout(timeout));
        this.timeouts = [];
        
        // Clear all intervals
        this.intervals.forEach(interval => clearInterval(interval));
        this.intervals = [];
        
        // Disconnect all observers
        this.observers.forEach(observer => observer.disconnect());
        this.observers = [];
        
        // Clear cache
        this.cache.clear();
        
        // Clear pending requests
        this.pendingRequests.clear();
        
        console.log('✅ Dashboard cleanup completed');
    }

    /**
     * PRELOADING SYSTEM
     */
    async preloadCriticalData() {
        console.log('🚀 Preloading critical data');
        
        const preloadTasks = [
            this.preloadAnalytics(),
            this.preloadRecentActivity(),
            this.preloadUserPreferences()
        ];
        
        try {
            await Promise.allSettled(preloadTasks);
            console.log('✅ Critical data preloaded');
        } catch (error) {
            console.error('❌ Preloading failed:', error);
        }
    }

    async preloadAnalytics() {
        try {
            await this.get('analytics', async () => {
                const response = await fetch('/api/analytics');
                if (!response.ok) throw new Error('Analytics fetch failed');
                return response.json();
            }, { ttl: this.cacheConfig.analytics });
        } catch (error) {
            console.error('Failed to preload analytics:', error);
        }
    }

    async preloadRecentActivity() {
        try {
            await this.get('recent_activity', async () => {
                const response = await fetch('/api/recent-activity');
                if (!response.ok) throw new Error('Recent activity fetch failed');
                return response.json();
            }, { ttl: this.cacheConfig.recentActivity });
        } catch (error) {
            console.error('Failed to preload recent activity:', error);
        }
    }

    async preloadUserPreferences() {
        // Load user preferences from localStorage or API
        const preferences = localStorage.getItem('userPreferences');
        if (preferences) {
            try {
                const parsed = JSON.parse(preferences);
                this.applyUserPreferences(parsed);
            } catch (error) {
                console.error('Failed to parse user preferences:', error);
            }
        }
    }

    applyUserPreferences(preferences) {
        // Apply user preferences to the dashboard
        if (preferences.theme) {
            document.documentElement.setAttribute('data-theme', preferences.theme);
        }
        
        if (preferences.smartFeatures) {
            this.smartFeatures = { ...this.smartFeatures, ...preferences.smartFeatures };
        }
    }

    /**
     * UTILITY METHODS
     */
    addEventListener(target, type, handler, options) {
        target.addEventListener(type, handler, options);
        this.eventListeners.push({ target, type, handler, options });
    }

    showNotification(message, type = 'info', duration = 4000) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `
            fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 max-w-sm 
            transition-all duration-300 transform translate-x-full
            ${type === 'success' ? 'bg-green-500 text-white' :
              type === 'error' ? 'bg-red-500 text-white' :
              type === 'warning' ? 'bg-yellow-500 text-black' :
              'bg-blue-500 text-white'}
        `;
        
        notification.innerHTML = `
            <div class="flex items-center">
                <span class="flex-1">${message}</span>
                <button class="ml-2 opacity-70 hover:opacity-100" onclick="this.parentElement.parentElement.remove()">
                    <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                    </svg>
                </button>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Animate in
        requestAnimationFrame(() => {
            notification.style.transform = 'translateX(0)';
        });
        
        // Auto-remove
        this.timeouts.push(setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, duration));
    }

    // Public API methods
    getPerformanceMetrics() {
        return { ...this.performanceMetrics };
    }

    getCacheStats() {
        return {
            size: this.cache.size,
            maxSize: this.cacheConfig.maxSize,
            hitRate: this.calculateCacheHitRate()
        };
    }

    clearAllCaches() {
        this.cache.clear();
        localStorage.removeItem('searchHistory');
        localStorage.removeItem('performanceMetrics');
        localStorage.removeItem('userBehavior');
        this.showNotification('All caches cleared', 'success');
    }

    toggleSmartFeature(feature, enabled) {
        if (feature in this.smartFeatures) {
            this.smartFeatures[feature] = enabled;
            localStorage.setItem('smartFeatures', JSON.stringify(this.smartFeatures));
            this.showNotification(`Smart feature '${feature}' ${enabled ? 'enabled' : 'disabled'}`, 'info');
        }
    }
}

// Global function to handle recommendation actions
window.handleRecommendationAction = (action, query) => {
    switch (action) {
        case 'search':
            if (typeof showView === 'function') {
                showView('search');
                setTimeout(() => {
                    const searchInput = document.getElementById('advancedSearchInput');
                    if (searchInput) {
                        searchInput.value = query;
                        searchInput.focus();
                        if (window.dashboardPerformance) {
                            window.dashboardPerformance.performSearch(query, searchInput);
                        }
                    }
                }, 100);
            }
            break;
        case 'new_note':
            const noteInput = document.getElementById('note');
            if (noteInput) {
                noteInput.focus();
            }
            break;
    }
    
    // Close the recommendation
    const recommendation = document.querySelector('.smart-recommendation');
    if (recommendation) {
        recommendation.remove();
    }
};

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.dashboardPerformance = new DashboardPerformance();
    });
} else {
    window.dashboardPerformance = new DashboardPerformance();
}

// Export for debugging and testing
window.DashboardPerformance = DashboardPerformance;