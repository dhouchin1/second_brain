// production-background.js - Enhanced production service worker for Second Brain extension
class SecondBrainBackgroundProduction {
    constructor() {
        this.offlineQueue = [];
        this.retryAttempts = new Map();
        this.maxRetries = 3;
        this.retryDelay = 60000; // 1 minute
        this.apiTimeout = 10000; // 10 seconds
        this.connectionStatus = 'unknown';
        this.performanceMetrics = {
            captureCount: 0,
            successCount: 0,
            errorCount: 0,
            averageResponseTime: 0
        };
        this.init();
    }

    async init() {
        try {
            this.setupContextMenus();
            this.setupMessageHandlers();
            this.setupInstallHandler();
            this.setupAlarms();
            this.setupCommandHandlers();
            await this.loadOfflineQueue();
            this.setupOnlineListener();
            this.setupPerformanceTracking();
            console.log('Second Brain extension initialized successfully');
        } catch (error) {
            console.error('Failed to initialize Second Brain extension:', error);
            this.handleCriticalError(error);
        }
    }

    setupContextMenus() {
        chrome.runtime.onInstalled.addListener(() => {
            try {
                const contextMenus = [
                    {
                        id: 'saveSelection',
                        title: 'Save to Second Brain',
                        contexts: ['selection']
                    },
                    {
                        id: 'savePage',
                        title: 'Save page to Second Brain',
                        contexts: ['page']
                    },
                    {
                        id: 'smartCapture',
                        title: '🧠 Smart AI capture',
                        contexts: ['page']
                    },
                    {
                        id: 'saveLink',
                        title: 'Save link to Second Brain',
                        contexts: ['link']
                    },
                    {
                        id: 'saveImage',
                        title: 'Save image to Second Brain',
                        contexts: ['image']
                    }
                ];

                contextMenus.forEach(menu => {
                    chrome.contextMenus.create(menu, () => {
                        if (chrome.runtime.lastError) {
                            console.warn(`Context menu creation warning: ${chrome.runtime.lastError.message}`);
                        }
                    });
                });
            } catch (error) {
                console.error('Failed to setup context menus:', error);
            }
        });

        chrome.contextMenus.onClicked.addListener((info, tab) => {
            this.handleContextMenuClick(info, tab);
        });
    }

    async handleContextMenuClick(info, tab) {
        const startTime = Date.now();
        try {
            this.performanceMetrics.captureCount++;
            
            const captureData = await this.extractCaptureData(info, tab);
            await this.sendCaptureRequest(captureData);
            
            this.performanceMetrics.successCount++;
            const responseTime = Date.now() - startTime;
            this.updateAverageResponseTime(responseTime);
            
            // Show success notification
            chrome.notifications.create({
                type: 'basic',
                iconUrl: 'icons/brain.svg',
                title: 'Second Brain',
                message: 'Content captured successfully!'
            });
            
        } catch (error) {
            this.performanceMetrics.errorCount++;
            console.error('Context menu capture failed:', error);
            
            // Queue for offline processing
            if (this.isNetworkError(error)) {
                await this.addToOfflineQueue({
                    type: 'contextMenu',
                    info: info,
                    tab: tab,
                    timestamp: Date.now()
                });
                
                chrome.notifications.create({
                    type: 'basic',
                    iconUrl: 'icons/brain.svg',
                    title: 'Second Brain - Offline',
                    message: 'Content queued for when connection is restored'
                });
            } else {
                chrome.notifications.create({
                    type: 'basic',
                    iconUrl: 'icons/brain.svg',
                    title: 'Second Brain - Error',
                    message: 'Capture failed. Please try again.'
                });
            }
        }
    }

    setupCommandHandlers() {
        chrome.commands.onCommand.addListener(async (command) => {
            try {
                const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
                
                if (command === 'quick-capture') {
                    await this.executeQuickCapture(tab);
                } else if (command === 'capture-page') {
                    await this.executePageCapture(tab);
                }
            } catch (error) {
                console.error(`Command handler error for ${command}:`, error);
            }
        });
    }

    async executeQuickCapture(tab) {
        try {
            const result = await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                function: () => {
                    const selection = window.getSelection();
                    return selection.toString().trim();
                }
            });

            const selectedText = result[0]?.result;
            if (!selectedText) {
                throw new Error('No text selected');
            }

            await this.sendCaptureRequest({
                type: 'selection',
                content: selectedText,
                url: tab.url,
                title: tab.title,
                timestamp: Date.now()
            });

        } catch (error) {
            console.error('Quick capture failed:', error);
            throw error;
        }
    }

    async executePageCapture(tab) {
        try {
            const result = await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                function: () => {
                    // Extract main content
                    const article = document.querySelector('article') || 
                                  document.querySelector('main') || 
                                  document.querySelector('[role="main"]') ||
                                  document.body;
                    return {
                        content: article.innerText,
                        html: article.innerHTML
                    };
                }
            });

            await this.sendCaptureRequest({
                type: 'page',
                content: result[0]?.result?.content,
                html: result[0]?.result?.html,
                url: tab.url,
                title: tab.title,
                timestamp: Date.now()
            });

        } catch (error) {
            console.error('Page capture failed:', error);
            throw error;
        }
    }

    async sendCaptureRequest(captureData, timeout = this.apiTimeout) {
        const settings = await chrome.storage.sync.get(['apiUrl', 'authToken']);
        
        if (!settings.apiUrl || !settings.authToken) {
            throw new Error('Extension not configured. Please set API URL and auth token.');
        }

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        try {
            const response = await fetch(`${settings.apiUrl}/api/capture`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${settings.authToken}`
                },
                body: JSON.stringify(captureData),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`API Error ${response.status}: ${errorText}`);
            }

            this.connectionStatus = 'connected';
            return await response.json();

        } catch (error) {
            clearTimeout(timeoutId);
            
            if (error.name === 'AbortError') {
                throw new Error('Request timeout - server may be unavailable');
            }
            
            if (error instanceof TypeError && error.message.includes('fetch')) {
                this.connectionStatus = 'disconnected';
                throw new Error('Network error - check server connection');
            }
            
            throw error;
        }
    }

    async addToOfflineQueue(item) {
        this.offlineQueue.push(item);
        await chrome.storage.local.set({ 
            offlineQueue: this.offlineQueue.slice(-100) // Keep only last 100 items
        });
    }

    async loadOfflineQueue() {
        try {
            const data = await chrome.storage.local.get(['offlineQueue']);
            this.offlineQueue = data.offlineQueue || [];
        } catch (error) {
            console.error('Failed to load offline queue:', error);
            this.offlineQueue = [];
        }
    }

    setupOnlineListener() {
        // Process offline queue when coming back online
        chrome.alarms.onAlarm.addListener((alarm) => {
            if (alarm.name === 'processOfflineQueue') {
                this.processOfflineQueue();
            }
        });

        // Check connection periodically
        chrome.alarms.create('processOfflineQueue', { 
            delayInMinutes: 1, 
            periodInMinutes: 5 
        });
    }

    async processOfflineQueue() {
        if (this.offlineQueue.length === 0) return;

        console.log(`Processing ${this.offlineQueue.length} offline items...`);
        
        const processedItems = [];
        for (const item of this.offlineQueue) {
            try {
                // Skip items older than 24 hours
                if (Date.now() - item.timestamp > 24 * 60 * 60 * 1000) {
                    processedItems.push(item);
                    continue;
                }

                if (item.type === 'contextMenu') {
                    const captureData = await this.extractCaptureData(item.info, item.tab);
                    await this.sendCaptureRequest(captureData, 5000); // Shorter timeout for offline processing
                }

                processedItems.push(item);
                console.log('Processed offline item successfully');

            } catch (error) {
                console.error('Failed to process offline item:', error);
                
                // Increment retry count
                const retryCount = this.retryAttempts.get(item) || 0;
                if (retryCount >= this.maxRetries) {
                    processedItems.push(item); // Remove after max retries
                } else {
                    this.retryAttempts.set(item, retryCount + 1);
                }
            }
        }

        // Remove processed items
        this.offlineQueue = this.offlineQueue.filter(item => !processedItems.includes(item));
        await chrome.storage.local.set({ offlineQueue: this.offlineQueue });
        
        // Clear retry attempts for processed items
        processedItems.forEach(item => this.retryAttempts.delete(item));
    }

    setupPerformanceTracking() {
        // Reset metrics daily
        chrome.alarms.create('resetMetrics', { 
            when: Date.now() + (24 * 60 * 60 * 1000),
            periodInMinutes: 24 * 60 
        });

        chrome.alarms.onAlarm.addListener((alarm) => {
            if (alarm.name === 'resetMetrics') {
                this.resetPerformanceMetrics();
            }
        });
    }

    resetPerformanceMetrics() {
        this.performanceMetrics = {
            captureCount: 0,
            successCount: 0,
            errorCount: 0,
            averageResponseTime: 0
        };
        chrome.storage.local.set({ performanceMetrics: this.performanceMetrics });
    }

    updateAverageResponseTime(responseTime) {
        const totalTime = this.performanceMetrics.averageResponseTime * this.performanceMetrics.successCount;
        this.performanceMetrics.averageResponseTime = 
            (totalTime + responseTime) / (this.performanceMetrics.successCount);
    }

    async extractCaptureData(info, tab) {
        const baseData = {
            url: tab.url,
            title: tab.title,
            timestamp: Date.now()
        };

        switch (info.menuItemId) {
            case 'saveSelection':
                return {
                    ...baseData,
                    type: 'selection',
                    content: info.selectionText
                };
            
            case 'saveLink':
                return {
                    ...baseData,
                    type: 'link',
                    content: info.linkUrl,
                    linkText: info.selectionText
                };
            
            case 'saveImage':
                return {
                    ...baseData,
                    type: 'image',
                    content: info.srcUrl,
                    alt: info.selectionText
                };
            
            case 'savePage':
            case 'smartCapture':
                // Extract page content
                const result = await chrome.scripting.executeScript({
                    target: { tabId: tab.id },
                    function: () => {
                        const article = document.querySelector('article') || 
                                      document.querySelector('main') || 
                                      document.querySelector('[role="main"]') ||
                                      document.body;
                        return {
                            content: article.innerText,
                            html: article.innerHTML
                        };
                    }
                });

                return {
                    ...baseData,
                    type: info.menuItemId === 'smartCapture' ? 'smart' : 'page',
                    content: result[0]?.result?.content,
                    html: result[0]?.result?.html
                };
            
            default:
                throw new Error(`Unknown menu item: ${info.menuItemId}`);
        }
    }

    isNetworkError(error) {
        return error.message.includes('fetch') || 
               error.message.includes('Network error') ||
               error.message.includes('timeout') ||
               error.name === 'AbortError';
    }

    handleCriticalError(error) {
        console.error('Critical extension error:', error);
        
        // Store error for debugging
        chrome.storage.local.set({
            lastError: {
                message: error.message,
                stack: error.stack,
                timestamp: Date.now()
            }
        });

        // Show user notification
        chrome.notifications.create({
            type: 'basic',
            iconUrl: 'icons/brain.svg',
            title: 'Second Brain - Critical Error',
            message: 'Extension encountered an error. Please reload or reinstall.'
        });
    }

    setupMessageHandlers() {
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            this.handleMessage(request, sender)
                .then(response => sendResponse(response))
                .catch(error => {
                    console.error('Message handler error:', error);
                    sendResponse({ error: error.message });
                });
            return true; // Keep message channel open for async response
        });
    }

    async handleMessage(request, sender) {
        switch (request.action) {
            case 'getConnectionStatus':
                return { 
                    status: this.connectionStatus,
                    queueSize: this.offlineQueue.length
                };
            
            case 'getPerformanceMetrics':
                return this.performanceMetrics;
            
            case 'testConnection':
                try {
                    const settings = await chrome.storage.sync.get(['apiUrl', 'authToken']);
                    await this.sendCaptureRequest({
                        type: 'test',
                        content: 'Connection test',
                        timestamp: Date.now()
                    });
                    return { success: true };
                } catch (error) {
                    return { success: false, error: error.message };
                }
            
            case 'clearOfflineQueue':
                this.offlineQueue = [];
                await chrome.storage.local.set({ offlineQueue: [] });
                return { success: true };
            
            default:
                throw new Error(`Unknown action: ${request.action}`);
        }
    }

    setupInstallHandler() {
        chrome.runtime.onInstalled.addListener((details) => {
            if (details.reason === 'install') {
                // Open options page on first install
                chrome.runtime.openOptionsPage();
                
                // Show welcome notification
                chrome.notifications.create({
                    type: 'basic',
                    iconUrl: 'icons/brain.svg',
                    title: 'Welcome to Second Brain!',
                    message: 'Configure your connection in the settings to start capturing content.'
                });
            } else if (details.reason === 'update') {
                console.log(`Extension updated to version ${chrome.runtime.getManifest().version}`);
            }
        });
    }

    setupAlarms() {
        // Clear old alarms on startup
        chrome.alarms.clearAll(() => {
            // Set up recurring alarms
            chrome.alarms.create('processOfflineQueue', { 
                delayInMinutes: 1, 
                periodInMinutes: 5 
            });
            
            chrome.alarms.create('resetMetrics', { 
                when: Date.now() + (24 * 60 * 60 * 1000),
                periodInMinutes: 24 * 60 
            });
        });
    }
}

// Initialize the background service
new SecondBrainBackgroundProduction();