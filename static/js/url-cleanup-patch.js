/**
 * URL Cleanup Patch - Replace URL-based data passing with state management
 * This file patches existing functions to use StateManager instead of URL parameters
 */

// Wait for state manager to be available
document.addEventListener('DOMContentLoaded', function() {
    if (!window.stateManager) {
        console.error('StateManager not available. Please include state-manager.js first.');
        return;
    }

    const stateManager = window.stateManager;

    // === SEARCH FUNCTIONS CLEANUP ===

    // Replace URL-based search with state-based search
    window.performSearchClean = function(query = null) {
        const searchQuery = query || document.getElementById('searchInput')?.value?.trim() || '';

        if (!searchQuery) {
            stateManager.setSearchResults([]);
            return;
        }

        // Store search in state instead of URL
        stateManager.setSearch(searchQuery);

        // Perform actual search via API
        searchApiCall(searchQuery);
    };

    // Clean API search function
    async function searchApiCall(query) {
        try {
            const response = await apiCall(`/api/search?q=${encodeURIComponent(query)}`);
            stateManager.setSearchResults(response);
            displaySearchResults(response, query);

            // Save to search history in state
            stateManager.addToSearchHistory(query);
        } catch (error) {
            console.error('Search failed:', error);
            showToast('Search failed. Please try again.', 'error');
        }
    }

    // === ADVANCED SEARCH CLEANUP ===

    window.performAdvancedSearchClean = function() {
        const query = document.getElementById('advancedSearchQuery')?.value?.trim() || '';
        const dateRange = document.getElementById('dateRangeFilter')?.value || '';
        const typeFilter = document.getElementById('typeFilter')?.value || '';

        const filters = {};
        if (dateRange) filters.date_range = dateRange;
        if (typeFilter) filters.type = typeFilter;

        // Store in state instead of URL
        stateManager.setSearch(query, filters);

        // Perform search
        performAdvancedApiSearch(query, filters);
    };

    async function performAdvancedApiSearch(query, filters) {
        try {
            const response = await apiCall('/api/search/advanced', {
                method: 'POST',
                body: JSON.stringify({ query, filters })
            });

            stateManager.setSearchResults(response);
            displaySearchResults(response, `Advanced search${query ? ` for "${query}"` : ''}`);
        } catch (error) {
            console.error('Advanced search failed:', error);
            showToast('Advanced search failed. Please try again.', 'error');
        }
    }

    // === NAVIGATION CLEANUP ===

    // Replace window.location navigation
    window.navigateClean = function(path, data = {}) {
        if (window.navigationManager) {
            window.navigationManager.navigateTo(path, data);
        } else {
            // Fallback to clean URLs without query parameters
            window.location.href = path;
        }
    };

    // Clean note viewing
    window.viewNoteClean = function(noteId) {
        // Fetch note and store in state instead of URL
        fetchNoteAndOpen(noteId);
    };

    async function fetchNoteAndOpen(noteId) {
        try {
            const note = await apiCall(`/api/notes/${noteId}`);
            stateManager.selectNote(note);
            openNoteModal(note);
        } catch (error) {
            console.error('Failed to fetch note:', error);
            showToast('Failed to load note', 'error');
        }
    }

    // === MODAL MANAGEMENT ===

    // Clean modal functions
    window.openNoteModalClean = function(note) {
        stateManager.selectNote(note);
        // The actual modal opening logic stays the same
        openNoteModal(note);
    };

    window.closeNoteModalClean = function() {
        stateManager.closeNote();
        closeNoteModal();
    };

    // === FORM HANDLING ===

    // Clean form submission without URL redirects
    window.handleNoteSubmitClean = async function(e) {
        e.preventDefault();

        const formData = new FormData(e.target);
        const noteData = {
            title: formData.get('title') || '',
            content: formData.get('content'),
            tags: formData.get('tags') || ''
        };

        try {
            const response = await apiCall('/api/notes', {
                method: 'POST',
                body: JSON.stringify(noteData)
            });

            if (response) {
                showToast('Note saved successfully!', 'success');

                // Refresh data without page reload
                loadRecentNotes();
                loadStats();

                // Clear form
                e.target.reset();

                // Auto-save draft cleanup
                localStorage.removeItem('second_brain_draft');
            }
        } catch (error) {
            console.error('Error saving note:', error);
            showToast('Failed to save note. Please try again.', 'error');
        }
    };

    // === FILE UPLOAD CLEANUP ===

    window.handleFileUploadClean = async function(files) {
        for (const file of files) {
            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await apiCall('/api/upload', {
                    method: 'POST',
                    body: formData,
                    headers: {} // Don't set Content-Type for FormData
                });

                if (response) {
                    showToast(`📄 ${file.name} uploaded successfully!`, 'success');
                    loadRecentNotes();
                    loadStats();
                }
            } catch (error) {
                console.error('Upload error:', error);
                showToast(`Failed to upload ${file.name}`, 'error');
            }
        }
    };

    // === MOBILE ACTIONS CLEANUP ===

    // Clean mobile capture without URL parameters
    window.handleMobileCaptureClean = function(action) {
        switch (action) {
            case 'voice-note':
                startVoiceRecording();
                break;
            case 'quick-note':
                focusNoteInput();
                break;
            case 'file-upload':
                document.getElementById('fileInput')?.click();
                break;
            default:
                console.warn('Unknown mobile action:', action);
        }
    };

    // === QUICK ACTIONS CLEANUP ===

    window.toggleQuickActionsClean = function() {
        stateManager.toggleQuickActions();

        const panel = document.getElementById('quickActionsPanel');
        if (panel) {
            const isOpen = stateManager.getState('quickActionsOpen');
            panel.classList.toggle('hidden', !isOpen);
        }
    };

    // === NOTIFICATION CENTER ===

    window.toggleNotificationCenterClean = function() {
        stateManager.toggleNotificationCenter();

        const center = document.getElementById('notificationCenter');
        if (center) {
            const isOpen = stateManager.getState('notificationCenterOpen');
            center.classList.toggle('hidden', !isOpen);
        }
    };

    // === REPLACE EXISTING FUNCTIONS ===

    // Override existing functions with clean versions
    if (typeof performSearch === 'function') {
        window.performSearch = window.performSearchClean;
    }

    if (typeof performAdvancedSearch === 'function') {
        window.performAdvancedSearch = window.performAdvancedSearchClean;
    }

    if (typeof viewNote === 'function') {
        window.viewNote = window.viewNoteClean;
    }

    if (typeof toggleQuickActions === 'function') {
        window.toggleQuickActions = window.toggleQuickActionsClean;
    }

    if (typeof toggleNotificationCenter === 'function') {
        window.toggleNotificationCenter = window.toggleNotificationCenterClean;
    }

    // === STATE SYNC WITH UI ===

    // Listen for state changes and update UI accordingly
    stateManager.subscribe('searchQuery', (query) => {
        const searchInput = document.getElementById('searchInput');
        if (searchInput && searchInput.value !== query) {
            searchInput.value = query;
        }
    });

    stateManager.subscribe('searchResults', (results) => {
        if (results.length > 0) {
            displaySearchResults(results, stateManager.getState('searchQuery'));
        }
    });

    stateManager.subscribe('noteModalOpen', (isOpen) => {
        const modal = document.getElementById('noteModal');
        if (modal) {
            modal.classList.toggle('hidden', !isOpen);
        }
    });

    // === URL CLEANUP ON LOAD ===

    // Clean any existing URL parameters
    function cleanExistingUrl() {
        const url = new URL(window.location);
        let hasParams = false;

        // Remove common parameter pollution
        const paramsToRemove = ['q', 'query', 'search', 'id', 'action', 'note', 'filter', 'type', 'date_range'];

        paramsToRemove.forEach(param => {
            if (url.searchParams.has(param)) {
                url.searchParams.delete(param);
                hasParams = true;
            }
        });

        if (hasParams && window.history && window.history.replaceState) {
            window.history.replaceState({}, document.title, url.pathname);
        }
    }

    // Clean URL on load
    cleanExistingUrl();

    // === EXPORT FUNCTIONS ===

    // Make clean functions available globally
    window.secondBrainClean = {
        search: window.performSearchClean,
        advancedSearch: window.performAdvancedSearchClean,
        navigate: window.navigateClean,
        viewNote: window.viewNoteClean,
        handleNoteSubmit: window.handleNoteSubmitClean,
        handleFileUpload: window.handleFileUploadClean,
        mobileCapture: window.handleMobileCaptureClean,
        toggleQuickActions: window.toggleQuickActionsClean,
        toggleNotificationCenter: window.toggleNotificationCenterClean
    };

    console.log('✅ URL cleanup patch applied successfully');
});

// === CSS for better UX ===
const cleanUrlStyles = document.createElement('style');
cleanUrlStyles.textContent = `
    /* Hide URL-based navigation indicators */
    .url-indicator,
    .query-display {
        display: none !important;
    }

    /* Smooth transitions for state changes */
    .state-transition {
        transition: all 0.2s ease-in-out;
    }

    /* Loading states */
    .state-loading {
        opacity: 0.7;
        pointer-events: none;
    }
`;
document.head.appendChild(cleanUrlStyles);