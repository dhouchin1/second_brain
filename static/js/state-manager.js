/**
 * State Manager - Centralized state management without URL pollution
 * Replaces URL-based data passing with in-memory state management
 */

class StateManager {
    constructor() {
        this.state = {
            // Navigation state
            currentView: 'dashboard',
            previousView: null,

            // Note state
            selectedNote: null,
            noteModalOpen: false,
            editMode: false,

            // Snapshot state
            selectedSnapshot: null,
            snapshotViewOpen: false,
            selectedArtifact: null,

            // Search state
            searchQuery: '',
            searchResults: [],
            searchFilters: {},
            searchHistory: [],

            // UI state
            mobileMenuOpen: false,
            notificationCenterOpen: false,
            quickActionsOpen: false,

            // App state
            isOnline: navigator.onLine,
            connectionStatus: 'connecting',
            offlineQueue: [],

            // Session state
            user: null,
            preferences: {}
        };

        this.listeners = new Map();
        this.persistentKeys = ['searchHistory', 'preferences', 'offlineQueue'];

        this.init();
    }

    init() {
        // Load persistent state from localStorage
        this.loadPersistentState();

        // Listen for online/offline events
        window.addEventListener('online', () => this.setState({ isOnline: true }));
        window.addEventListener('offline', () => this.setState({ isOnline: false }));

        // Auto-save persistent state
        setInterval(() => this.savePersistentState(), 5000);

        // Clean URLs on page load
        this.cleanUrl();
    }

    /**
     * Get current state or specific property
     */
    getState(key = null) {
        if (key) {
            return this.state[key];
        }
        return { ...this.state };
    }

    /**
     * Update state and notify listeners
     */
    setState(updates) {
        const prevState = { ...this.state };
        this.state = { ...this.state, ...updates };

        // Notify listeners of specific changes
        Object.keys(updates).forEach(key => {
            if (this.listeners.has(key)) {
                this.listeners.get(key).forEach(callback => {
                    callback(this.state[key], prevState[key]);
                });
            }
        });

        // Auto-save persistent keys
        if (Object.keys(updates).some(key => this.persistentKeys.includes(key))) {
            this.savePersistentState();
        }
    }

    /**
     * Subscribe to state changes
     */
    subscribe(key, callback) {
        if (!this.listeners.has(key)) {
            this.listeners.set(key, []);
        }
        this.listeners.get(key).push(callback);

        // Return unsubscribe function
        return () => {
            const callbacks = this.listeners.get(key);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        };
    }

    /**
     * Navigation methods without URL changes
     */
    navigateTo(view, data = {}) {
        this.setState({
            previousView: this.state.currentView,
            currentView: view,
            ...data
        });

        // Update document title
        this.updateDocumentTitle(view);

        // Clean URL to remove any query parameters
        this.cleanUrl();
    }

    /**
     * Search methods without URL parameters
     */
    setSearch(query, filters = {}) {
        this.setState({
            searchQuery: query,
            searchFilters: filters
        });

        // Add to search history if query is not empty
        if (query.trim()) {
            this.addToSearchHistory(query);
        }
    }

    setSearchResults(results) {
        this.setState({ searchResults: results });
    }

    addToSearchHistory(query) {
        const history = [...this.state.searchHistory];
        const existingIndex = history.indexOf(query);

        if (existingIndex > -1) {
            history.splice(existingIndex, 1);
        }

        history.unshift(query);

        // Keep only last 20 searches
        if (history.length > 20) {
            history.splice(20);
        }

        this.setState({ searchHistory: history });
    }

    /**
     * Note management without URL exposure
     */
    selectNote(note) {
        this.setState({
            selectedNote: note,
            noteModalOpen: true
        });
    }

    closeNote() {
        this.setState({
            selectedNote: null,
            noteModalOpen: false,
            editMode: false
        });
    }

    enterEditMode() {
        this.setState({ editMode: true });
    }

    exitEditMode() {
        this.setState({ editMode: false });
    }

    /**
     * Snapshot management without URL exposure
     */
    selectSnapshot(snapshotData) {
        this.setState({
            selectedSnapshot: snapshotData,
            snapshotViewOpen: true
        });
    }

    closeSnapshot() {
        this.setState({
            selectedSnapshot: null,
            snapshotViewOpen: false,
            selectedArtifact: null
        });
    }

    selectArtifact(artifact) {
        this.setState({
            selectedArtifact: artifact
        });
    }

    closeArtifact() {
        this.setState({
            selectedArtifact: null
        });
    }

    /**
     * UI state management
     */
    toggleMobileMenu() {
        this.setState({ mobileMenuOpen: !this.state.mobileMenuOpen });
    }

    toggleNotificationCenter() {
        this.setState({ notificationCenterOpen: !this.state.notificationCenterOpen });
    }

    toggleQuickActions() {
        this.setState({ quickActionsOpen: !this.state.quickActionsOpen });
    }

    /**
     * Offline queue management
     */
    addToOfflineQueue(action) {
        const queue = [...this.state.offlineQueue, action];
        this.setState({ offlineQueue: queue });
    }

    processOfflineQueue() {
        // This will be called when coming back online
        const queue = [...this.state.offlineQueue];
        this.setState({ offlineQueue: [] });
        return queue;
    }

    /**
     * Persistent state management
     */
    loadPersistentState() {
        this.persistentKeys.forEach(key => {
            try {
                const saved = localStorage.getItem(`second_brain_${key}`);
                if (saved) {
                    this.state[key] = JSON.parse(saved);
                }
            } catch (error) {
                console.warn(`Failed to load ${key} from localStorage:`, error);
            }
        });
    }

    savePersistentState() {
        this.persistentKeys.forEach(key => {
            try {
                localStorage.setItem(`second_brain_${key}`, JSON.stringify(this.state[key]));
            } catch (error) {
                console.warn(`Failed to save ${key} to localStorage:`, error);
            }
        });
    }

    /**
     * URL management
     */
    cleanUrl() {
        if (window.history && window.history.replaceState) {
            const cleanUrl = window.location.protocol + "//" +
                            window.location.host +
                            window.location.pathname;
            window.history.replaceState({}, document.title, cleanUrl);
        }
    }

    updateDocumentTitle(view) {
        const titles = {
            dashboard: 'Dashboard - Second Brain',
            search: 'Search - Second Brain',
            analytics: 'Analytics - Second Brain',
            note: this.state.selectedNote ?
                  `${this.state.selectedNote.title || 'Untitled'} - Second Brain` :
                  'Note - Second Brain'
        };

        document.title = titles[view] || 'Second Brain';
    }

    /**
     * Debug helpers
     */
    logState() {
        console.log('Current State:', this.getState());
    }

    resetState() {
        this.state = {
            currentView: 'dashboard',
            previousView: null,
            selectedNote: null,
            noteModalOpen: false,
            editMode: false,
            searchQuery: '',
            searchResults: [],
            searchFilters: {},
            searchHistory: [],
            mobileMenuOpen: false,
            notificationCenterOpen: false,
            quickActionsOpen: false,
            isOnline: navigator.onLine,
            connectionStatus: 'connecting',
            offlineQueue: [],
            user: null,
            preferences: {}
        };
        this.savePersistentState();
    }
}

// Create global instance
window.stateManager = new StateManager();

// Export for modules that need it
if (typeof module !== 'undefined' && module.exports) {
    module.exports = StateManager;
}