/**
 * Dashboard Core - Essential functionality only
 * Optimized for fast initial load
 */

class DashboardCore {
    constructor() {
        this.notes = [];
        this.isLoading = false;
        this.searchCache = new Map();
        this.searchTimeout = null;
        this.lastSearchQuery = '';

        // Lazy loading modules
        this.modules = {
            performance: null,
            help: null,
            advanced: null
        };

        this.initCore();
    }

    initCore() {
        this.bindEvents();
        this.loadInitialData();
        this.setupCoreFeatures();
    }

    bindEvents() {
        // Search functionality
        const searchInput = document.getElementById('searchInput');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => this.handleSearch(e.target.value));
            searchInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.performSearch(e.target.value);
                }
            });
        }

        // Note interaction
        document.addEventListener('click', (e) => {
            if (e.target.matches('.note-item, .note-item *')) {
                const noteElement = e.target.closest('.note-item');
                if (noteElement) {
                    this.handleNoteClick(noteElement);
                }
            }
        });

        // Quick actions
        const newNoteBtn = document.getElementById('newNoteBtn');
        if (newNoteBtn) {
            newNoteBtn.addEventListener('click', () => this.showNewNoteModal());
        }
    }

    async loadInitialData() {
        try {
            this.showLoading(true);
            const response = await fetch('/api/notes?limit=20');
            if (response.ok) {
                const data = await response.json();
                this.notes = data.notes || [];
                this.renderNotes();
            }
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.showError('Failed to load notes');
        } finally {
            this.showLoading(false);
        }
    }

    handleSearch(query) {
        clearTimeout(this.searchTimeout);
        this.searchTimeout = setTimeout(() => {
            if (query !== this.lastSearchQuery) {
                this.performSearch(query);
                this.lastSearchQuery = query;
            }
        }, 300);
    }

    async performSearch(query) {
        if (!query.trim()) {
            this.loadInitialData();
            return;
        }

        // Check cache first
        if (this.searchCache.has(query)) {
            this.renderNotes(this.searchCache.get(query));
            return;
        }

        try {
            this.showLoading(true);
            const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
            if (response.ok) {
                const data = await response.json();
                const results = data.results || [];
                this.searchCache.set(query, results);
                this.renderNotes(results);
            }
        } catch (error) {
            console.error('Search failed:', error);
            this.showError('Search failed');
        } finally {
            this.showLoading(false);
        }
    }

    renderNotes(notesToRender = null) {
        const notesList = document.getElementById('notesList');
        if (!notesList) return;

        const notes = notesToRender || this.notes;

        if (notes.length === 0) {
            notesList.innerHTML = `
                <div class="text-center py-8 text-gray-500">
                    <p>No notes found</p>
                </div>
            `;
            return;
        }

        notesList.innerHTML = notes.map(note => this.renderNoteItem(note)).join('');
    }

    renderNoteItem(note) {
        const title = note.title || 'Untitled';
        const content = note.content || note.body || '';
        const preview = content.slice(0, 150) + (content.length > 150 ? '...' : '');
        const date = note.timestamp || note.created_at || '';
        const formattedDate = date ? new Date(date).toLocaleDateString() : '';

        return `
            <div class="note-item bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow cursor-pointer p-4 mb-3" data-note-id="${note.id}">
                <div class="flex justify-between items-start mb-2">
                    <h3 class="font-semibold text-gray-900 line-clamp-1">${this.escapeHtml(title)}</h3>
                    <span class="text-xs text-gray-500">${formattedDate}</span>
                </div>
                <p class="text-gray-600 text-sm line-clamp-3">${this.escapeHtml(preview)}</p>
                ${note.tags ? `<div class="mt-2 flex flex-wrap gap-1">
                    ${note.tags.split(',').map(tag => `<span class="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded">${this.escapeHtml(tag.trim())}</span>`).join('')}
                </div>` : ''}
            </div>
        `;
    }

    handleNoteClick(noteElement) {
        const noteId = noteElement.dataset.noteId;
        if (noteId) {
            this.openNoteModal(noteId);
        }
    }

    async openNoteModal(noteId) {
        try {
            const response = await fetch(`/api/notes/${noteId}`);
            if (response.ok) {
                const note = await response.json();
                this.showNoteModal(note);
            }
        } catch (error) {
            console.error('Failed to load note:', error);
            this.showError('Failed to load note');
        }
    }

    showNoteModal(note) {
        const modal = document.getElementById('noteModal');
        if (!modal) return;

        const content = document.getElementById('noteModalContent');
        if (content) {
            content.innerHTML = `
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-xl font-bold">${this.escapeHtml(note.title || 'Untitled')}</h2>
                    <button onclick="dashboard.hideNoteModal()" class="text-gray-500 hover:text-gray-700">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                        </svg>
                    </button>
                </div>
                <div class="prose max-w-none">
                    <p>${this.escapeHtml(note.content || note.body || 'No content')}</p>
                </div>
            `;
        }

        modal.classList.remove('hidden');
    }

    hideNoteModal() {
        const modal = document.getElementById('noteModal');
        if (modal) {
            modal.classList.add('hidden');
        }
    }

    showNewNoteModal() {
        // Simple implementation - can be enhanced later
        const content = prompt('Enter note content:');
        if (content) {
            this.createNote(content);
        }
    }

    async createNote(content) {
        try {
            const response = await fetch('/api/notes', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ content })
            });

            if (response.ok) {
                this.loadInitialData(); // Refresh notes
                this.showSuccess('Note created successfully');
            }
        } catch (error) {
            console.error('Failed to create note:', error);
            this.showError('Failed to create note');
        }
    }

    setupCoreFeatures() {
        // Add keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case 'n':
                        e.preventDefault();
                        this.showNewNoteModal();
                        break;
                    case 'k':
                        e.preventDefault();
                        document.getElementById('searchInput')?.focus();
                        break;
                }
            }
        });
    }

    // Lazy load advanced modules
    async loadModule(moduleName) {
        if (this.modules[moduleName]) {
            return this.modules[moduleName];
        }

        try {
            const module = await import(`./dashboard-${moduleName}.js`);
            this.modules[moduleName] = module.default;
            return module.default;
        } catch (error) {
            console.error(`Failed to load module ${moduleName}:`, error);
            return null;
        }
    }

    showLoading(show) {
        const loader = document.getElementById('loadingIndicator');
        if (loader) {
            loader.classList.toggle('hidden', !show);
        }
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `fixed top-4 right-4 px-4 py-2 rounded-lg shadow-lg z-50 ${
            type === 'error' ? 'bg-red-500 text-white' :
            type === 'success' ? 'bg-green-500 text-white' :
            'bg-blue-500 text-white'
        }`;
        notification.textContent = message;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize dashboard when DOM is ready
let dashboard;
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        dashboard = new DashboardCore();
        window.dashboard = dashboard;
    });
} else {
    dashboard = new DashboardCore();
    window.dashboard = dashboard;
}

export default DashboardCore;