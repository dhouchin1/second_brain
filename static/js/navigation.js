/**
 * Navigation System - Clean URL navigation without exposing data
 * Replaces window.location and URL parameter manipulation
 */

class NavigationManager {
    constructor(stateManager) {
        this.stateManager = stateManager;
        this.routes = new Map();
        this.currentRoute = null;

        this.init();
    }

    init() {
        // Define routes without exposing data
        this.defineRoutes();

        // Listen for state changes to update UI
        this.stateManager.subscribe('currentView', (newView, prevView) => {
            this.handleViewChange(newView, prevView);
        });

        // Handle browser back/forward buttons
        window.addEventListener('popstate', (e) => {
            this.handlePopState(e);
        });
    }

    /**
     * Define clean routes without query parameters
     */
    defineRoutes() {
        this.routes.set('/', {
            view: 'dashboard',
            title: 'Dashboard',
            handler: () => this.showDashboard()
        });

        this.routes.set('/search', {
            view: 'search',
            title: 'Search',
            handler: () => this.showSearch()
        });

        this.routes.set('/analytics', {
            view: 'analytics',
            title: 'Analytics',
            handler: () => this.showAnalytics()
        });

        this.routes.set('/note', {
            view: 'note',
            title: 'Note',
            handler: () => this.showNote()
        });

        this.routes.set('/snapshot', {
            view: 'snapshot',
            title: 'Snapshot',
            handler: () => this.showSnapshot()
        });
    }

    /**
     * Navigate to view without URL pollution
     */
    navigateTo(path, data = {}) {
        const route = this.routes.get(path);
        if (!route) {
            console.warn(`Route not found: ${path}`);
            return;
        }

        // Update state without URL changes
        this.stateManager.navigateTo(route.view, data);

        // Update browser history with clean URL
        if (window.history && window.history.pushState) {
            window.history.pushState(
                { view: route.view, data },
                route.title,
                path
            );
        }

        // Execute route handler
        route.handler();
    }

    /**
     * Replace current URL without adding to history
     */
    replaceWith(path, data = {}) {
        const route = this.routes.get(path);
        if (!route) {
            console.warn(`Route not found: ${path}`);
            return;
        }

        this.stateManager.navigateTo(route.view, data);

        if (window.history && window.history.replaceState) {
            window.history.replaceState(
                { view: route.view, data },
                route.title,
                path
            );
        }

        route.handler();
    }

    /**
     * Go back to previous view
     */
    goBack() {
        const prevView = this.stateManager.getState('previousView');
        if (prevView) {
            // Map view to path
            const path = this.getPathForView(prevView);
            this.navigateTo(path);
        } else {
            this.navigateTo('/');
        }
    }

    /**
     * Handle browser back/forward
     */
    handlePopState(e) {
        if (e.state && e.state.view) {
            this.stateManager.navigateTo(e.state.view, e.state.data || {});
            const route = [...this.routes.values()].find(r => r.view === e.state.view);
            if (route) {
                route.handler();
            }
        }
    }

    /**
     * Handle view changes
     */
    handleViewChange(newView, prevView) {
        // Update active nav items
        this.updateActiveNavigation(newView);

        // Handle view-specific logic
        switch (newView) {
            case 'dashboard':
                this.hidePanels();
                break;
            case 'search':
                this.hidePanels();
                this.focusSearchInput();
                break;
            case 'analytics':
                this.hidePanels();
                break;
            case 'note':
                // Note modal will be handled by note state
                break;
        }
    }

    /**
     * Route handlers
     */
    showDashboard() {
        // Hide all overlays and focus on dashboard
        this.hidePanels();
        this.showMainContent();
    }

    showSearch() {
        this.hidePanels();
        this.showMainContent();
        // Focus will be handled by view change
    }

    showAnalytics() {
        // Navigate to analytics page
        if (window.location.pathname !== '/analytics') {
            window.location.href = '/analytics';
        }
    }

    showNote() {
        // Note display is handled by modal state
        // This is just for URL consistency
    }

    showSnapshot() {
        this.hidePanels();
        // Snapshot display is handled by state
        // Clean URL maintained
    }

    /**
     * Utility methods
     */
    getPathForView(view) {
        for (const [path, route] of this.routes) {
            if (route.view === view) {
                return path;
            }
        }
        return '/';
    }

    updateActiveNavigation(currentView) {
        // Update navigation highlighting
        document.querySelectorAll('[data-nav-item]').forEach(item => {
            const itemView = item.dataset.navItem;
            if (itemView === currentView) {
                item.classList.add('active', 'text-primary-600');
                item.classList.remove('text-gray-600');
            } else {
                item.classList.remove('active', 'text-primary-600');
                item.classList.add('text-gray-600');
            }
        });
    }

    hidePanels() {
        // Close all panels and modals
        this.stateManager.setState({
            mobileMenuOpen: false,
            notificationCenterOpen: false,
            quickActionsOpen: false
        });

        // Hide mobile menu
        const mobileMenu = document.getElementById('mobileMenu');
        if (mobileMenu) {
            mobileMenu.classList.add('hidden');
        }

        // Hide notification center
        const notificationCenter = document.getElementById('notificationCenter');
        if (notificationCenter) {
            notificationCenter.classList.add('hidden');
        }
    }

    showMainContent() {
        // Ensure main content is visible
        const mainContent = document.querySelector('main') || document.querySelector('.main-content');
        if (mainContent) {
            mainContent.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    focusSearchInput() {
        setTimeout(() => {
            const searchInput = document.getElementById('searchInput');
            if (searchInput) {
                searchInput.focus();
            }
        }, 100);
    }

    /**
     * External navigation (opens in new tab for external links)
     */
    openExternal(url) {
        window.open(url, '_blank', 'noopener,noreferrer');
    }

    /**
     * Safe navigation for forms and actions
     */
    submitForm(formId, successCallback) {
        const form = document.getElementById(formId);
        if (!form) return;

        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            try {
                const formData = new FormData(form);
                const response = await fetch(form.action, {
                    method: form.method || 'POST',
                    body: formData,
                    credentials: 'same-origin'
                });

                if (response.ok) {
                    if (successCallback) {
                        successCallback(response);
                    }
                } else {
                    throw new Error(`Form submission failed: ${response.status}`);
                }
            } catch (error) {
                console.error('Form submission error:', error);
                // Handle error without exposing details in URL
                this.showError('Submission failed. Please try again.');
            }
        });
    }

    showError(message) {
        // Show error without URL parameters
        if (window.showToast) {
            window.showToast(message, 'error');
        } else {
            alert(message);
        }
    }

    /**
     * Deep linking support (for shared links)
     */
    parseInitialState() {
        // Only parse minimal, non-sensitive URL parameters on initial load
        const urlParams = new URLSearchParams(window.location.search);
        const action = urlParams.get('action');

        // Handle only safe, public actions
        if (action === 'search') {
            // Don't include the query in URL, just trigger search UI
            this.navigateTo('/search');
        }

        // Clean URL after parsing
        this.stateManager.cleanUrl();
    }
}

// Create global navigation instance
window.navigationManager = new NavigationManager(window.stateManager);

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.navigationManager.parseInitialState();
    });
} else {
    window.navigationManager.parseInitialState();
}

// Export for modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NavigationManager;
}