/**
 * Advanced Content Management System for Dashboard v3
 * 
 * Features:
 * - Enhanced note management with inline editing
 * - Advanced tagging system with smart suggestions
 * - Content search enhancements and saved searches
 * - Note templates system with dynamic variables
 * - Bulk operations and note organization
 * - Smart features and automation
 */

class ContentManager {
    constructor(performanceSystem = null) {
        this.performanceSystem = performanceSystem;
        
        // Mobile interface integration
        this.mobileInterface = null;
        this.isMobile = this.detectMobile();
        this.hasTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
        
        // Note management state
        this.noteState = {
            notes: new Map(),
            selectedNotes: new Set(),
            editingNote: null,
            draggedNote: null,
            viewMode: 'card', // card, list, compact
            sortBy: 'updated_at',
            sortOrder: 'desc',
            filters: {
                tags: [],
                dateRange: null,
                type: 'all',
                status: 'all'
            }
        };

        // Tagging system
        this.tagSystem = {
            tags: new Map(),
            tagHierarchy: new Map(),
            suggestedTags: [],
            tagColors: new Map(),
            recentTags: JSON.parse(localStorage.getItem('recentTags') || '[]'),
            autoTagging: true
        };

        // Search system
        this.searchSystem = {
            savedSearches: JSON.parse(localStorage.getItem('savedSearches') || '[]'),
            searchHistory: JSON.parse(localStorage.getItem('searchHistory') || '[]'),
            currentQuery: '',
            filters: {},
            suggestions: [],
            globalSearchVisible: false
        };

        // Templates system
        this.templateSystem = {
            templates: new Map(),
            variables: {
                'date': () => new Date().toISOString().split('T')[0],
                'time': () => new Date().toLocaleTimeString(),
                'datetime': () => new Date().toLocaleString(),
                'user': () => this.getCurrentUser(),
                'week': () => this.getWeekNumber(),
                'month': () => new Date().toLocaleDateString('en-US', { month: 'long' }),
                'year': () => new Date().getFullYear().toString()
            },
            defaultTemplates: [
                {
                    id: 'meeting',
                    name: 'Meeting Notes',
                    content: `# Meeting Notes - {{date}}

## Attendees
- 

## Agenda
1. 

## Discussion Points

## Action Items
- [ ] 

## Next Meeting
Date: 
Time: `
                },
                {
                    id: 'daily',
                    name: 'Daily Journal',
                    content: `# Daily Journal - {{date}}

## Today's Goals
- 

## Completed Tasks
- 

## Thoughts & Reflections

## Tomorrow's Priorities
- `
                },
                {
                    id: 'project',
                    name: 'Project Planning',
                    content: `# Project: [Project Name]

## Overview
**Start Date:** {{date}}  
**Deadline:** 
**Status:** Planning

## Objectives
- 

## Milestones
- [ ] 

## Resources Needed
- 

## Risks & Mitigation
- 

## Notes
`
                }
            ]
        };

        // Bulk operations
        this.bulkOperations = {
            inProgress: false,
            operations: ['tag', 'delete', 'archive', 'favorite', 'move'],
            history: []
        };

        // Smart features
        this.smartFeatures = {
            autoSave: true,
            smartSuggestions: true,
            relatedNotes: true,
            duplicateDetection: true,
            titleGeneration: true
        };

        this.init();
    }

    /**
     * INITIALIZATION
     */
    async init() {
        console.log('🚀 Initializing Content Management System');
        
        try {
            // Initialize all subsystems
            await this.initializeNoteManagement();
            await this.initializeTaggingSystem();
            await this.initializeSearchSystem();
            await this.initializeTemplateSystem();
            this.initializeBulkOperations();
            this.initializeSmartFeatures();
            this.initializeKeyboardShortcuts();
            
            // Load initial data
            await this.loadNotes();
            await this.loadTags();
            await this.loadTemplates();
            
            console.log('✅ Content Management System initialized');
        } catch (error) {
            console.error('❌ Content Management initialization failed:', error);
            if (this.performanceSystem) {
                this.performanceSystem.handleError('content-management-init', error);
            }
        }
    }

    /**
     * NOTE MANAGEMENT
     */
    async initializeNoteManagement() {
        // Initialize inline editing
        this.initializeInlineEditing();
        
        // Setup note actions
        this.setupNoteActions();
        
        // Initialize drag and drop
        this.initializeNoteDragDrop();
        
        // Setup view modes
        this.initializeViewModes();
    }

    initializeInlineEditing() {
        // Double-click to edit titles
        document.addEventListener('dblclick', (event) => {
            const titleElement = event.target.closest('[data-note-title]');
            if (titleElement) {
                this.startInlineTitleEdit(titleElement);
            }
            
            const contentElement = event.target.closest('[data-note-content]');
            if (contentElement) {
                this.startInlineContentEdit(contentElement);
            }
        });

        // Click outside to save
        document.addEventListener('click', (event) => {
            if (this.noteState.editingNote && !event.target.closest('.inline-editor')) {
                this.saveInlineEdit();
            }
        });
    }

    startInlineTitleEdit(titleElement) {
        if (this.noteState.editingNote) return;
        
        const noteId = titleElement.closest('[data-note-id]').dataset.noteId;
        const currentTitle = titleElement.textContent.trim();
        
        // Create inline editor
        const input = document.createElement('input');
        input.type = 'text';
        input.value = currentTitle;
        input.className = 'inline-editor w-full bg-transparent border-none outline-none font-medium text-slate-800 focus:ring-2 focus:ring-discord-500 rounded px-1';
        
        // Replace title with input
        titleElement.innerHTML = '';
        titleElement.appendChild(input);
        
        // Focus and select text
        input.focus();
        input.select();
        
        // Store editing state
        this.noteState.editingNote = {
            id: noteId,
            type: 'title',
            element: titleElement,
            input: input,
            originalValue: currentTitle
        };

        // Handle Enter/Escape
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                this.saveInlineEdit();
            } else if (e.key === 'Escape') {
                this.cancelInlineEdit();
            }
        });
    }

    startInlineContentEdit(contentElement) {
        if (this.noteState.editingNote) return;
        
        const noteId = contentElement.closest('[data-note-id]').dataset.noteId;
        const currentContent = contentElement.textContent.trim();
        
        // Create inline editor (textarea for content)
        const textarea = document.createElement('textarea');
        textarea.value = currentContent;
        textarea.className = 'inline-editor w-full bg-transparent border border-slate-300 outline-none text-slate-700 focus:ring-2 focus:ring-discord-500 rounded px-2 py-1 resize-none';
        textarea.rows = 4;
        
        // Replace content with textarea
        contentElement.innerHTML = '';
        contentElement.appendChild(textarea);
        
        // Auto-resize textarea
        this.autoResizeTextarea(textarea);
        
        // Focus
        textarea.focus();
        
        // Store editing state
        this.noteState.editingNote = {
            id: noteId,
            type: 'content',
            element: contentElement,
            input: textarea,
            originalValue: currentContent
        };

        // Handle Ctrl+Enter to save, Escape to cancel
        textarea.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                this.saveInlineEdit();
            } else if (e.key === 'Escape') {
                this.cancelInlineEdit();
            }
        });

        textarea.addEventListener('input', () => this.autoResizeTextarea(textarea));
    }

    async saveInlineEdit() {
        if (!this.noteState.editingNote) return;
        
        const { id, type, element, input, originalValue } = this.noteState.editingNote;
        const newValue = input.value.trim();
        
        if (newValue === originalValue) {
            this.cancelInlineEdit();
            return;
        }
        
        try {
            // Show saving indicator
            this.showSavingIndicator(element);
            
            // Save to backend
            const updateData = {};
            updateData[type] = newValue;
            
            const response = await fetch(`/api/notes/${id}`, {
                method: 'PATCH',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(updateData)
            });
            
            if (!response.ok) {
                throw new Error(`Failed to save ${type}`);
            }
            
            // Update local state
            const note = this.noteState.notes.get(id);
            if (note) {
                note[type] = newValue;
                note.updated_at = new Date().toISOString();
            }
            
            // Restore element with new value
            element.innerHTML = this.escapeHtml(newValue);
            
            // Show success indicator
            this.showSaveSuccess(element);
            
            // Trigger auto-suggestions if content changed
            if (type === 'content' && this.smartFeatures.smartSuggestions) {
                this.generateSmartSuggestions(id, newValue);
            }
            
        } catch (error) {
            console.error('Failed to save inline edit:', error);
            element.innerHTML = this.escapeHtml(originalValue);
            this.showSaveError(element);
        } finally {
            this.noteState.editingNote = null;
        }
    }

    cancelInlineEdit() {
        if (!this.noteState.editingNote) return;
        
        const { element, originalValue } = this.noteState.editingNote;
        element.innerHTML = this.escapeHtml(originalValue);
        this.noteState.editingNote = null;
    }

    setupNoteActions() {
        // Setup action buttons for notes
        document.addEventListener('click', async (event) => {
            const actionButton = event.target.closest('[data-note-action]');
            if (!actionButton) return;
            
            const action = actionButton.dataset.noteAction;
            const noteId = actionButton.closest('[data-note-id]').dataset.noteId;
            
            await this.handleNoteAction(action, noteId, event);
        });
    }

    async handleNoteAction(action, noteId, event) {
        try {
            switch (action) {
                case 'edit':
                    this.openNoteEditor(noteId);
                    break;
                case 'duplicate':
                    await this.duplicateNote(noteId);
                    break;
                case 'delete':
                    await this.deleteNote(noteId, event.shiftKey); // Hard delete with Shift
                    break;
                case 'archive':
                    await this.archiveNote(noteId);
                    break;
                case 'favorite':
                    await this.toggleFavorite(noteId);
                    break;
                case 'share':
                    this.shareNote(noteId);
                    break;
                case 'export':
                    this.exportNote(noteId);
                    break;
                case 'tag':
                    this.showTagDialog(noteId);
                    break;
                default:
                    console.warn(`Unknown note action: ${action}`);
            }
        } catch (error) {
            console.error(`Failed to execute note action ${action}:`, error);
            this.showNotification(`Failed to ${action} note`, 'error');
        }
    }

    async duplicateNote(noteId) {
        const originalNote = this.noteState.notes.get(noteId);
        if (!originalNote) return;
        
        const duplicateData = {
            title: `${originalNote.title} (Copy)`,
            content: originalNote.content,
            tags: originalNote.tags || []
        };
        
        const response = await fetch('/api/notes', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(duplicateData)
        });
        
        if (response.ok) {
            const newNote = await response.json();
            this.noteState.notes.set(newNote.id.toString(), newNote);
            this.refreshNotesDisplay();
            this.showNotification('Note duplicated successfully', 'success');
        }
    }

    async deleteNote(noteId, hardDelete = false) {
        const note = this.noteState.notes.get(noteId);
        if (!note) return;
        
        const confirmMessage = hardDelete 
            ? 'Permanently delete this note? This cannot be undone.'
            : 'Move this note to trash?';
            
        if (!confirm(confirmMessage)) return;
        
        const endpoint = hardDelete ? `/api/notes/${noteId}?hard=true` : `/api/notes/${noteId}`;
        const response = await fetch(endpoint, { method: 'DELETE' });
        
        if (response.ok) {
            this.noteState.notes.delete(noteId);
            this.noteState.selectedNotes.delete(noteId);
            this.refreshNotesDisplay();
            
            const message = hardDelete ? 'Note permanently deleted' : 'Note moved to trash';
            this.showNotification(message, 'success');
        }
    }

    async archiveNote(noteId) {
        const response = await fetch(`/api/notes/${noteId}`, {
            method: 'PATCH',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ archived: true })
        });
        
        if (response.ok) {
            const note = this.noteState.notes.get(noteId);
            if (note) {
                note.archived = true;
                note.updated_at = new Date().toISOString();
            }
            this.refreshNotesDisplay();
            this.showNotification('Note archived', 'success');
        }
    }

    async toggleFavorite(noteId) {
        const note = this.noteState.notes.get(noteId);
        if (!note) return;
        
        const newFavoriteState = !note.favorite;
        
        const response = await fetch(`/api/notes/${noteId}`, {
            method: 'PATCH',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ favorite: newFavoriteState })
        });
        
        if (response.ok) {
            note.favorite = newFavoriteState;
            note.updated_at = new Date().toISOString();
            this.refreshNotesDisplay();
            
            const message = newFavoriteState ? 'Added to favorites' : 'Removed from favorites';
            this.showNotification(message, 'success');
        }
    }

    /**
     * TAGGING SYSTEM
     */
    async initializeTaggingSystem() {
        await this.loadTags();
        this.initializeTagInput();
        this.initializeTagHierarchy();
        this.initializeTagColors();
    }

    initializeTagInput() {
        // Enhanced tag input with autocomplete
        document.addEventListener('focus', (event) => {
            if (event.target.matches('[data-tag-input]')) {
                this.initializeTagAutocomplete(event.target);
            }
        });
    }

    initializeTagAutocomplete(input) {
        const suggestionContainer = this.createTagSuggestions(input);
        
        input.addEventListener('input', () => {
            this.updateTagSuggestions(input, suggestionContainer);
        });
        
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Tab' || e.key === 'Enter') {
                const activeSuggestion = suggestionContainer.querySelector('.active');
                if (activeSuggestion) {
                    e.preventDefault();
                    this.selectTag(input, activeSuggestion.dataset.tag);
                }
            } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
                this.navigateTagSuggestions(suggestionContainer, e.key === 'ArrowDown');
            }
        });
    }

    createTagSuggestions(input) {
        const container = document.createElement('div');
        container.className = 'tag-suggestions absolute z-50 bg-white border border-slate-300 rounded-lg shadow-lg max-h-48 overflow-y-auto hidden';
        container.style.top = '100%';
        container.style.left = '0';
        container.style.right = '0';
        
        // Position relative to input
        const inputContainer = input.closest('.relative') || input.parentElement;
        inputContainer.style.position = 'relative';
        inputContainer.appendChild(container);
        
        return container;
    }

    updateTagSuggestions(input, container) {
        const query = input.value.toLowerCase().trim();
        
        if (query.length === 0) {
            container.classList.add('hidden');
            return;
        }
        
        // Get matching tags
        const matchingTags = Array.from(this.tagSystem.tags.keys())
            .filter(tag => tag.toLowerCase().includes(query))
            .sort((a, b) => {
                // Prioritize exact matches and recent tags
                const aExact = a.toLowerCase().startsWith(query);
                const bExact = b.toLowerCase().startsWith(query);
                const aRecent = this.tagSystem.recentTags.includes(a);
                const bRecent = this.tagSystem.recentTags.includes(b);
                
                if (aExact && !bExact) return -1;
                if (!aExact && bExact) return 1;
                if (aRecent && !bRecent) return -1;
                if (!aRecent && bRecent) return 1;
                return a.localeCompare(b);
            })
            .slice(0, 10);
        
        // Add smart suggestions based on content
        if (this.smartFeatures.smartSuggestions) {
            const smartTags = this.generateSmartTagSuggestions(input);
            matchingTags.push(...smartTags.slice(0, 3));
        }
        
        // Render suggestions
        container.innerHTML = matchingTags.map((tag, index) => `
            <div class="tag-suggestion px-3 py-2 hover:bg-slate-100 cursor-pointer flex items-center justify-between ${index === 0 ? 'active' : ''}" 
                 data-tag="${tag}">
                <span class="flex items-center">
                    <span class="tag-color w-3 h-3 rounded-full mr-2" 
                          style="background-color: ${this.getTagColor(tag)}"></span>
                    <span class="font-medium">${tag}</span>
                </span>
                <span class="text-xs text-slate-500">
                    ${this.getTagCount(tag)} notes
                </span>
            </div>
        `).join('');
        
        container.classList.remove('hidden');
        
        // Handle selection
        container.querySelectorAll('.tag-suggestion').forEach(suggestion => {
            suggestion.addEventListener('click', () => {
                this.selectTag(input, suggestion.dataset.tag);
            });
        });
    }

    selectTag(input, tag) {
        // Add tag to recent tags
        this.addToRecentTags(tag);
        
        // Get current tags
        const currentTags = input.value.split(',').map(t => t.trim()).filter(t => t.length > 0);
        const lastTag = currentTags[currentTags.length - 1];
        
        // Replace last partial tag or add new tag
        if (lastTag && !this.tagSystem.tags.has(lastTag)) {
            currentTags[currentTags.length - 1] = tag;
        } else {
            currentTags.push(tag);
        }
        
        // Update input
        input.value = currentTags.join(', ') + ', ';
        input.focus();
        
        // Hide suggestions
        const container = input.parentElement.querySelector('.tag-suggestions');
        if (container) {
            container.classList.add('hidden');
        }
        
        // Trigger change event
        input.dispatchEvent(new Event('input'));
    }

    generateSmartTagSuggestions(input) {
        const noteContent = this.getNoteContentForTagging(input);
        if (!noteContent) return [];
        
        const suggestions = [];
        const words = noteContent.toLowerCase().split(/\s+/);
        
        // Extract potential tags from content
        const patterns = [
            /^#(\w+)/g, // Hashtags
            /\b(project|meeting|idea|todo|research)\b/g, // Common tag patterns
            /\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b/g // CamelCase words
        ];
        
        patterns.forEach(pattern => {
            let match;
            while ((match = pattern.exec(noteContent)) !== null) {
                const candidate = match[1] || match[0];
                if (candidate.length > 2 && !this.tagSystem.tags.has(candidate)) {
                    suggestions.push(candidate.toLowerCase());
                }
            }
        });
        
        return [...new Set(suggestions)].slice(0, 3);
    }

    initializeTagHierarchy() {
        // Support for hierarchical tags (e.g., project/web-app, work/meetings)
        this.tagSystem.tagHierarchy.clear();
        
        for (const tag of this.tagSystem.tags.keys()) {
            if (tag.includes('/')) {
                const parts = tag.split('/');
                const parent = parts[0];
                const child = parts.slice(1).join('/');
                
                if (!this.tagSystem.tagHierarchy.has(parent)) {
                    this.tagSystem.tagHierarchy.set(parent, new Set());
                }
                this.tagSystem.tagHierarchy.get(parent).add(child);
            }
        }
    }

    initializeTagColors() {
        // Auto-assign colors to tags based on hash
        for (const tag of this.tagSystem.tags.keys()) {
            if (!this.tagSystem.tagColors.has(tag)) {
                this.tagSystem.tagColors.set(tag, this.generateTagColor(tag));
            }
        }
    }

    generateTagColor(tag) {
        // Generate consistent color based on tag name
        let hash = 0;
        for (let i = 0; i < tag.length; i++) {
            hash = tag.charCodeAt(i) + ((hash << 5) - hash);
        }
        
        const hue = Math.abs(hash % 360);
        return `hsl(${hue}, 60%, 70%)`;
    }

    getTagColor(tag) {
        return this.tagSystem.tagColors.get(tag) || this.generateTagColor(tag);
    }

    getTagCount(tag) {
        return this.tagSystem.tags.get(tag) || 0;
    }

    addToRecentTags(tag) {
        // Add to recent tags and limit to 20
        const recent = this.tagSystem.recentTags.filter(t => t !== tag);
        recent.unshift(tag);
        this.tagSystem.recentTags = recent.slice(0, 20);
        localStorage.setItem('recentTags', JSON.stringify(this.tagSystem.recentTags));
    }

    /**
     * SEARCH SYSTEM
     */
    async initializeSearchSystem() {
        this.initializeGlobalSearch();
        this.initializeSavedSearches();
        this.initializeSearchFilters();
        this.initializeSearchHistory();
    }

    initializeGlobalSearch() {
        // ⌘K shortcut for global search
        document.addEventListener('keydown', (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
                e.preventDefault();
                this.showGlobalSearch();
            }
        });
    }

    showGlobalSearch() {
        if (this.searchSystem.globalSearchVisible) return;
        
        const overlay = document.createElement('div');
        overlay.className = 'fixed inset-0 bg-black bg-opacity-50 z-50 flex items-start justify-center pt-20';
        overlay.innerHTML = `
            <div class="bg-white rounded-lg shadow-2xl w-full max-w-2xl mx-4 max-h-96 flex flex-col">
                <div class="p-4 border-b border-slate-200">
                    <div class="relative">
                        <input type="text" 
                               id="global-search-input"
                               placeholder="Search notes, tags, or content..."
                               class="w-full px-4 py-3 pl-10 text-lg border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-discord-500">
                        <svg class="absolute left-3 top-3.5 h-5 w-5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                        </svg>
                    </div>
                    <div class="flex items-center mt-3 text-sm text-slate-600">
                        <span class="flex items-center mr-4">
                            <kbd class="px-2 py-1 bg-slate-100 rounded text-xs">↵</kbd>
                            <span class="ml-1">to search</span>
                        </span>
                        <span class="flex items-center">
                            <kbd class="px-2 py-1 bg-slate-100 rounded text-xs">Esc</kbd>
                            <span class="ml-1">to close</span>
                        </span>
                    </div>
                </div>
                <div class="flex-1 overflow-hidden">
                    <div id="global-search-results" class="p-4 overflow-y-auto max-h-80">
                        <div class="text-slate-500 text-center py-8">
                            Start typing to search your notes...
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(overlay);
        this.searchSystem.globalSearchVisible = true;
        
        const input = overlay.querySelector('#global-search-input');
        const results = overlay.querySelector('#global-search-results');
        
        // Focus input
        input.focus();
        
        // Setup search functionality
        let searchTimeout;
        input.addEventListener('input', () => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                this.performGlobalSearch(input.value, results);
            }, 200);
        });
        
        // Handle keyboard navigation
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hideGlobalSearch(overlay);
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                this.navigateSearchResults(results, 'down');
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                this.navigateSearchResults(results, 'up');
            } else if (e.key === 'Enter') {
                const activeResult = results.querySelector('.search-result.active');
                if (activeResult) {
                    activeResult.click();
                }
            }
        });
        
        // Close on outside click
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                this.hideGlobalSearch(overlay);
            }
        });
    }

    hideGlobalSearch(overlay) {
        if (overlay) {
            overlay.remove();
        }
        this.searchSystem.globalSearchVisible = false;
    }

    async performGlobalSearch(query, resultsContainer) {
        if (query.trim().length === 0) {
            resultsContainer.innerHTML = `
                <div class="text-slate-500 text-center py-8">
                    Start typing to search your notes...
                </div>
            `;
            return;
        }
        
        try {
            // Show loading state
            resultsContainer.innerHTML = `
                <div class="text-slate-500 text-center py-8">
                    <div class="animate-spin h-6 w-6 border-2 border-discord-500 border-t-transparent rounded-full mx-auto mb-2"></div>
                    Searching...
                </div>
            `;
            
            // Perform search
            const response = await fetch(`/api/search?q=${encodeURIComponent(query)}&limit=20`);
            const data = await response.json();
            
            if (data.results && data.results.length > 0) {
                this.renderSearchResults(data.results, resultsContainer, query);
                
                // Add to search history
                this.addToSearchHistory(query);
            } else {
                resultsContainer.innerHTML = `
                    <div class="text-slate-500 text-center py-8">
                        No results found for "${query}"
                        <div class="mt-4">
                            <button class="text-discord-500 hover:text-discord-600 underline text-sm"
                                    onclick="contentManager.saveSearch('${query}')">
                                Save this search
                            </button>
                        </div>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Global search failed:', error);
            resultsContainer.innerHTML = `
                <div class="text-red-500 text-center py-8">
                    Search failed. Please try again.
                </div>
            `;
        }
    }

    renderSearchResults(results, container, query) {
        container.innerHTML = results.map((result, index) => `
            <div class="search-result p-3 rounded-lg hover:bg-slate-50 cursor-pointer border-l-4 border-transparent ${index === 0 ? 'active bg-slate-50' : ''}"
                 data-note-id="${result.id}">
                <div class="flex items-start justify-between">
                    <div class="flex-1">
                        <h4 class="font-medium text-slate-800 mb-1">
                            ${this.highlightSearchQuery(result.title || 'Untitled', query)}
                        </h4>
                        ${result.snippet ? `
                            <p class="text-sm text-slate-600 mb-2 line-clamp-2">
                                ${this.highlightSearchQuery(result.snippet, query)}
                            </p>
                        ` : ''}
                        <div class="flex items-center space-x-2 text-xs text-slate-500">
                            <span>${new Date(result.updated_at).toLocaleDateString()}</span>
                            ${result.tags ? `
                                <span>•</span>
                                <div class="flex space-x-1">
                                    ${result.tags.slice(0, 3).map(tag => `
                                        <span class="px-2 py-1 bg-slate-200 rounded text-xs">${tag}</span>
                                    `).join('')}
                                    ${result.tags.length > 3 ? `<span>+${result.tags.length - 3}</span>` : ''}
                                </div>
                            ` : ''}
                        </div>
                    </div>
                    <div class="text-xs text-slate-400 ml-4">
                        ${result.score ? `${Math.round(result.score * 100)}% match` : ''}
                    </div>
                </div>
            </div>
        `).join('');
        
        // Add click handlers
        container.querySelectorAll('.search-result').forEach(result => {
            result.addEventListener('click', () => {
                const noteId = result.dataset.noteId;
                this.openNote(noteId);
                this.hideGlobalSearch(result.closest('.fixed'));
            });
        });
    }

    initializeSavedSearches() {
        // Load saved searches from localStorage
        this.searchSystem.savedSearches = JSON.parse(localStorage.getItem('savedSearches') || '[]');
        this.renderSavedSearches();
    }

    saveSearch(query, name = null) {
        if (!name) {
            name = prompt('Name for this search:');
            if (!name) return;
        }
        
        const savedSearch = {
            id: Date.now().toString(),
            name,
            query,
            filters: { ...this.searchSystem.filters },
            createdAt: new Date().toISOString()
        };
        
        this.searchSystem.savedSearches.push(savedSearch);
        localStorage.setItem('savedSearches', JSON.stringify(this.searchSystem.savedSearches));
        
        this.renderSavedSearches();
        this.showNotification(`Search "${name}" saved`, 'success');
    }

    renderSavedSearches() {
        const container = document.getElementById('saved-searches');
        if (!container || this.searchSystem.savedSearches.length === 0) return;
        
        container.innerHTML = this.searchSystem.savedSearches.map(search => `
            <div class="saved-search flex items-center justify-between p-2 rounded hover:bg-slate-50">
                <div class="cursor-pointer flex-1" onclick="contentManager.executeSavedSearch('${search.id}')">
                    <span class="font-medium text-slate-700">${search.name}</span>
                    <span class="text-sm text-slate-500 ml-2">"${search.query}"</span>
                </div>
                <button onclick="contentManager.deleteSavedSearch('${search.id}')" 
                        class="text-slate-400 hover:text-red-500 p-1">
                    <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                    </svg>
                </button>
            </div>
        `).join('');
    }

    /**
     * TEMPLATE SYSTEM
     */
    async initializeTemplateSystem() {
        // Load default templates
        for (const template of this.templateSystem.defaultTemplates) {
            this.templateSystem.templates.set(template.id, template);
        }
        
        // Load custom templates
        await this.loadCustomTemplates();
        
        // Initialize template UI
        this.initializeTemplateUI();
    }

    initializeTemplateUI() {
        // Template selection dropdown
        const createButton = document.querySelector('[data-action="create-note"]');
        if (createButton) {
            // Add dropdown arrow to create button
            createButton.innerHTML += ' <svg class="h-4 w-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>';
            
            // Create dropdown menu
            const dropdown = this.createTemplateDropdown();
            createButton.parentElement.appendChild(dropdown);
            
            // Toggle dropdown
            createButton.addEventListener('click', (e) => {
                e.stopPropagation();
                dropdown.classList.toggle('hidden');
            });
            
            // Close dropdown when clicking outside
            document.addEventListener('click', () => {
                dropdown.classList.add('hidden');
            });
        }
    }

    createTemplateDropdown() {
        const dropdown = document.createElement('div');
        dropdown.className = 'template-dropdown absolute top-full left-0 mt-2 bg-white border border-slate-300 rounded-lg shadow-lg z-10 min-w-48 hidden';
        
        const templates = Array.from(this.templateSystem.templates.values());
        
        dropdown.innerHTML = `
            <div class="p-2">
                <div class="text-xs font-medium text-slate-500 uppercase tracking-wide mb-2">Templates</div>
                ${templates.map(template => `
                    <button class="template-option w-full text-left p-2 rounded hover:bg-slate-50 text-sm"
                            data-template-id="${template.id}">
                        <div class="font-medium text-slate-800">${template.name}</div>
                    </button>
                `).join('')}
                <hr class="my-2">
                <button class="w-full text-left p-2 rounded hover:bg-slate-50 text-sm text-discord-600 font-medium"
                        onclick="contentManager.createBlankNote()">
                    + Blank Note
                </button>
                <button class="w-full text-left p-2 rounded hover:bg-slate-50 text-sm text-slate-600"
                        onclick="contentManager.showTemplateManager()">
                    Manage Templates...
                </button>
            </div>
        `;
        
        // Handle template selection
        dropdown.querySelectorAll('.template-option').forEach(option => {
            option.addEventListener('click', () => {
                const templateId = option.dataset.templateId;
                this.createNoteFromTemplate(templateId);
                dropdown.classList.add('hidden');
            });
        });
        
        return dropdown;
    }

    async createNoteFromTemplate(templateId) {
        const template = this.templateSystem.templates.get(templateId);
        if (!template) return;
        
        // Process template variables
        const processedContent = this.processTemplateVariables(template.content);
        
        // Create note
        const noteData = {
            title: template.name + ' - ' + new Date().toLocaleDateString(),
            content: processedContent,
            template_id: templateId
        };
        
        try {
            const response = await fetch('/api/notes', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(noteData)
            });
            
            if (response.ok) {
                const newNote = await response.json();
                this.openNoteEditor(newNote.id);
                this.showNotification('Note created from template', 'success');
            }
        } catch (error) {
            console.error('Failed to create note from template:', error);
            this.showNotification('Failed to create note', 'error');
        }
    }

    processTemplateVariables(content) {
        let processed = content;
        
        // Process built-in variables
        for (const [variable, getValue] of Object.entries(this.templateSystem.variables)) {
            const regex = new RegExp(`{{${variable}}}`, 'g');
            processed = processed.replace(regex, getValue());
        }
        
        return processed;
    }

    /**
     * BULK OPERATIONS
     */
    initializeBulkOperations() {
        // Initialize multi-select
        this.initializeMultiSelect();
        
        // Initialize bulk action bar
        this.initializeBulkActionBar();
    }

    initializeMultiSelect() {
        document.addEventListener('change', (event) => {
            if (event.target.matches('input[data-note-checkbox]')) {
                const noteId = event.target.value;
                
                if (event.target.checked) {
                    this.noteState.selectedNotes.add(noteId);
                } else {
                    this.noteState.selectedNotes.delete(noteId);
                }
                
                this.updateBulkActionBar();
            }
        });
        
        // Select all functionality
        document.addEventListener('change', (event) => {
            if (event.target.matches('#select-all-notes')) {
                const checkboxes = document.querySelectorAll('input[data-note-checkbox]');
                checkboxes.forEach(checkbox => {
                    checkbox.checked = event.target.checked;
                    if (event.target.checked) {
                        this.noteState.selectedNotes.add(checkbox.value);
                    } else {
                        this.noteState.selectedNotes.delete(checkbox.value);
                    }
                });
                
                this.updateBulkActionBar();
            }
        });
    }

    initializeBulkActionBar() {
        const actionBar = document.getElementById('bulk-action-bar');
        if (!actionBar) {
            this.createBulkActionBar();
        }
    }

    createBulkActionBar() {
        const actionBar = document.createElement('div');
        actionBar.id = 'bulk-action-bar';
        actionBar.className = 'fixed bottom-4 left-1/2 transform -translate-x-1/2 bg-slate-800 text-white rounded-lg shadow-lg p-4 hidden z-40';
        actionBar.innerHTML = `
            <div class="flex items-center space-x-4">
                <span id="selected-count" class="text-sm font-medium">0 notes selected</span>
                <div class="flex space-x-2">
                    <button onclick="contentManager.bulkTag()" class="px-3 py-1 bg-blue-600 rounded hover:bg-blue-700 text-sm">
                        Tag
                    </button>
                    <button onclick="contentManager.bulkArchive()" class="px-3 py-1 bg-yellow-600 rounded hover:bg-yellow-700 text-sm">
                        Archive
                    </button>
                    <button onclick="contentManager.bulkDelete()" class="px-3 py-1 bg-red-600 rounded hover:bg-red-700 text-sm">
                        Delete
                    </button>
                    <button onclick="contentManager.clearSelection()" class="px-3 py-1 bg-slate-600 rounded hover:bg-slate-700 text-sm">
                        Cancel
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(actionBar);
    }

    updateBulkActionBar() {
        const actionBar = document.getElementById('bulk-action-bar');
        const selectedCount = document.getElementById('selected-count');
        
        if (this.noteState.selectedNotes.size > 0) {
            actionBar.classList.remove('hidden');
            selectedCount.textContent = `${this.noteState.selectedNotes.size} notes selected`;
        } else {
            actionBar.classList.add('hidden');
        }
    }

    async bulkTag() {
        const selectedNotes = Array.from(this.noteState.selectedNotes);
        if (selectedNotes.length === 0) return;
        
        const tags = prompt('Enter tags (comma-separated):');
        if (!tags) return;
        
        const tagList = tags.split(',').map(t => t.trim()).filter(t => t.length > 0);
        
        try {
            await Promise.all(selectedNotes.map(noteId => 
                fetch(`/api/notes/${noteId}/tags`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ tags: tagList })
                })
            ));
            
            this.showNotification(`Added tags to ${selectedNotes.length} notes`, 'success');
            this.clearSelection();
            this.refreshNotesDisplay();
        } catch (error) {
            console.error('Bulk tagging failed:', error);
            this.showNotification('Bulk tagging failed', 'error');
        }
    }

    async bulkArchive() {
        const selectedNotes = Array.from(this.noteState.selectedNotes);
        if (selectedNotes.length === 0) return;
        
        if (!confirm(`Archive ${selectedNotes.length} notes?`)) return;
        
        try {
            await Promise.all(selectedNotes.map(noteId => 
                fetch(`/api/notes/${noteId}`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ archived: true })
                })
            ));
            
            this.showNotification(`Archived ${selectedNotes.length} notes`, 'success');
            this.clearSelection();
            this.refreshNotesDisplay();
        } catch (error) {
            console.error('Bulk archiving failed:', error);
            this.showNotification('Bulk archiving failed', 'error');
        }
    }

    async bulkDelete() {
        const selectedNotes = Array.from(this.noteState.selectedNotes);
        if (selectedNotes.length === 0) return;
        
        if (!confirm(`Delete ${selectedNotes.length} notes? This will move them to trash.`)) return;
        
        try {
            await Promise.all(selectedNotes.map(noteId => 
                fetch(`/api/notes/${noteId}`, { method: 'DELETE' })
            ));
            
            // Remove from local state
            selectedNotes.forEach(noteId => {
                this.noteState.notes.delete(noteId);
            });
            
            this.showNotification(`Deleted ${selectedNotes.length} notes`, 'success');
            this.clearSelection();
            this.refreshNotesDisplay();
        } catch (error) {
            console.error('Bulk deletion failed:', error);
            this.showNotification('Bulk deletion failed', 'error');
        }
    }

    clearSelection() {
        this.noteState.selectedNotes.clear();
        document.querySelectorAll('input[data-note-checkbox]').forEach(checkbox => {
            checkbox.checked = false;
        });
        document.getElementById('select-all-notes').checked = false;
        this.updateBulkActionBar();
    }

    /**
     * UTILITY METHODS
     */
    async loadNotes() {
        try {
            const response = await fetch('/api/notes');
            const notes = await response.json();
            
            this.noteState.notes.clear();
            notes.forEach(note => {
                this.noteState.notes.set(note.id.toString(), note);
            });
        } catch (error) {
            console.error('Failed to load notes:', error);
        }
    }

    async loadTags() {
        try {
            const response = await fetch('/api/tags');
            const tags = await response.json();
            
            this.tagSystem.tags.clear();
            tags.forEach(tag => {
                this.tagSystem.tags.set(tag.name, tag.count);
            });
        } catch (error) {
            console.error('Failed to load tags:', error);
        }
    }

    async loadTemplates() {
        try {
            const response = await fetch('/api/templates');
            if (response.ok) {
                const templates = await response.json();
                templates.forEach(template => {
                    this.templateSystem.templates.set(template.id, template);
                });
            }
        } catch (error) {
            console.warn('Failed to load custom templates:', error);
        }
    }

    initializeKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Only handle shortcuts when not in input fields
            if (e.target.matches('input, textarea, [contenteditable]')) return;
            
            if ((e.metaKey || e.ctrlKey) && e.key === 'n') {
                e.preventDefault();
                this.createBlankNote();
            } else if (e.key === 'Delete' || e.key === 'Backspace') {
                if (this.noteState.selectedNotes.size > 0) {
                    this.bulkDelete();
                }
            } else if ((e.metaKey || e.ctrlKey) && e.key === 'a') {
                e.preventDefault();
                this.selectAllNotes();
            }
        });
    }

    // Utility methods
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    highlightSearchQuery(text, query) {
        if (!query || !text) return text;
        
        const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
        return text.replace(regex, '<mark class="bg-yellow-200">$1</mark>');
    }

    autoResizeTextarea(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = textarea.scrollHeight + 'px';
    }

    showSavingIndicator(element) {
        const indicator = document.createElement('span');
        indicator.className = 'saving-indicator text-xs text-slate-500 ml-2';
        indicator.textContent = 'Saving...';
        element.appendChild(indicator);
    }

    showSaveSuccess(element) {
        const indicator = element.querySelector('.saving-indicator');
        if (indicator) {
            indicator.textContent = '✓ Saved';
            indicator.className = 'saving-indicator text-xs text-green-600 ml-2';
            setTimeout(() => indicator.remove(), 2000);
        }
    }

    showSaveError(element) {
        const indicator = element.querySelector('.saving-indicator');
        if (indicator) {
            indicator.textContent = '✗ Failed';
            indicator.className = 'saving-indicator text-xs text-red-600 ml-2';
            setTimeout(() => indicator.remove(), 3000);
        }
    }

    showNotification(message, type = 'info') {
        // Use existing notification system or create simple one
        console.log(`[${type.toUpperCase()}] ${message}`);
        
        // Simple notification implementation
        const notification = document.createElement('div');
        notification.className = `fixed top-4 right-4 px-4 py-2 rounded-lg shadow-lg z-50 ${
            type === 'success' ? 'bg-green-500 text-white' :
            type === 'error' ? 'bg-red-500 text-white' :
            'bg-blue-500 text-white'
        }`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    getCurrentUser() {
        return localStorage.getItem('currentUser') || 'User';
    }

    getWeekNumber() {
        const date = new Date();
        const firstDayOfYear = new Date(date.getFullYear(), 0, 1);
        const pastDaysOfYear = (date - firstDayOfYear) / 86400000;
        return Math.ceil((pastDaysOfYear + firstDayOfYear.getDay() + 1) / 7);
    }

    refreshNotesDisplay() {
        // Trigger refresh of notes display
        const event = new CustomEvent('notes-updated');
        document.dispatchEvent(event);
    }

    openNote(noteId) {
        window.location.href = `/note/${noteId}`;
    }

    openNoteEditor(noteId) {
        window.location.href = `/note/${noteId}/edit`;
    }

    createBlankNote() {
        window.location.href = '/note/new';
    }

    /**
     * MOBILE-SPECIFIC OPTIMIZATIONS
     */
    detectMobile() {
        const userAgent = navigator.userAgent.toLowerCase();
        const mobileKeywords = ['android', 'iphone', 'ipod', 'ipad', 'windows phone', 'blackberry'];
        return mobileKeywords.some(keyword => userAgent.includes(keyword)) || 
               window.innerWidth <= 768;
    }

    setMobileInterface(mobileInterface) {
        this.mobileInterface = mobileInterface;
        this.optimizeForMobile();
    }

    optimizeForMobile() {
        if (!this.isMobile && !this.hasTouch) return;

        // Optimize touch interactions
        this.optimizeTouchInteractions();
        
        // Adjust mobile-specific behaviors
        this.adjustMobileNoteBehaviors();
        
        // Optimize search for mobile
        this.optimizeMobileSearch();
        
        // Optimize tagging for mobile
        this.optimizeMobileTagging();
    }

    optimizeTouchInteractions() {
        // Make note cards swipeable
        document.addEventListener('DOMContentLoaded', () => {
            this.makeNoteCardsSwipeable();
        });

        // Optimize inline editing for touch
        this.optimizeInlineEditingForTouch();
    }

    makeNoteCardsSwipeable() {
        const noteCards = document.querySelectorAll('[data-note-id]');
        noteCards.forEach(card => {
            card.setAttribute('data-swipeable', 'true');
            card.style.touchAction = 'pan-y'; // Allow vertical scrolling but handle horizontal swipes
        });
    }

    optimizeInlineEditingForTouch() {
        if (!this.hasTouch) return;

        // Use longer press for inline editing on mobile
        let pressTimer;
        
        document.addEventListener('touchstart', (e) => {
            const titleElement = e.target.closest('[data-note-title]');
            const contentElement = e.target.closest('[data-note-content]');
            
            if (titleElement || contentElement) {
                pressTimer = setTimeout(() => {
                    if (titleElement) {
                        this.startInlineTitleEdit(titleElement);
                    } else if (contentElement) {
                        this.startInlineContentEdit(contentElement);
                    }
                }, 600); // 600ms press for inline editing on mobile
            }
        });

        document.addEventListener('touchend', () => {
            if (pressTimer) {
                clearTimeout(pressTimer);
                pressTimer = null;
            }
        });

        document.addEventListener('touchmove', () => {
            if (pressTimer) {
                clearTimeout(pressTimer);
                pressTimer = null;
            }
        });
    }

    adjustMobileNoteBehaviors() {
        // Override double-click with single tap for mobile
        if (this.hasTouch) {
            // Remove double-click listeners and replace with mobile-optimized ones
            this.replaceMobileEditingBehavior();
        }

        // Optimize bulk operations for mobile
        this.optimizeBulkOperationsForMobile();
    }

    replaceMobileEditingBehavior() {
        // Override the existing double-click behavior
        const originalInitializeInlineEditing = this.initializeInlineEditing;
        
        this.initializeInlineEditing = () => {
            if (!this.hasTouch) {
                // Use original double-click behavior for non-touch devices
                originalInitializeInlineEditing.call(this);
                return;
            }

            // Mobile-optimized editing: use tap + hold or dedicated edit buttons
            document.addEventListener('click', (event) => {
                const editButton = event.target.closest('[data-mobile-edit]');
                if (editButton) {
                    const noteElement = editButton.closest('[data-note-id]');
                    const titleElement = noteElement.querySelector('[data-note-title]');
                    const contentElement = noteElement.querySelector('[data-note-content]');
                    
                    if (editButton.dataset.mobileEdit === 'title' && titleElement) {
                        this.startInlineTitleEdit(titleElement);
                    } else if (editButton.dataset.mobileEdit === 'content' && contentElement) {
                        this.startInlineContentEdit(contentElement);
                    }
                }
            });

            // Click outside to save (same as original)
            document.addEventListener('click', (event) => {
                if (this.noteState.editingNote && !event.target.closest('.inline-editor')) {
                    this.saveInlineEdit();
                }
            });
        };
    }

    optimizeBulkOperationsForMobile() {
        if (!this.isMobile) return;

        // Override bulk action bar position for mobile
        const originalCreateBulkActionBar = this.createBulkActionBar;
        
        this.createBulkActionBar = () => {
            const actionBar = document.createElement('div');
            actionBar.id = 'bulk-action-bar';
            // Position at bottom for mobile, with safe area insets
            actionBar.className = 'fixed bottom-16 left-4 right-4 bg-slate-800 text-white rounded-lg shadow-lg p-4 hidden z-40';
            actionBar.style.paddingBottom = 'calc(1rem + env(safe-area-inset-bottom))';
            
            actionBar.innerHTML = `
                <div class="flex flex-col space-y-3">
                    <div class="text-center">
                        <span id="selected-count" class="text-sm font-medium">0 notes selected</span>
                    </div>
                    <div class="flex space-x-2 justify-center">
                        <button onclick="contentManager.bulkTag()" class="flex-1 py-2 bg-blue-600 rounded hover:bg-blue-700 text-sm font-medium">
                            Tag
                        </button>
                        <button onclick="contentManager.bulkArchive()" class="flex-1 py-2 bg-yellow-600 rounded hover:bg-yellow-700 text-sm font-medium">
                            Archive
                        </button>
                        <button onclick="contentManager.bulkDelete()" class="flex-1 py-2 bg-red-600 rounded hover:bg-red-700 text-sm font-medium">
                            Delete
                        </button>
                    </div>
                    <button onclick="contentManager.clearSelection()" class="w-full py-2 bg-slate-600 rounded hover:bg-slate-700 text-sm font-medium">
                        Cancel Selection
                    </button>
                </div>
            `;
            
            document.body.appendChild(actionBar);
        };
    }

    optimizeMobileSearch() {
        if (!this.isMobile) return;

        // Override global search to use mobile-optimized layout
        const originalShowGlobalSearch = this.showGlobalSearch;
        
        this.showGlobalSearch = () => {
            if (this.searchSystem.globalSearchVisible) return;
            
            const overlay = document.createElement('div');
            overlay.className = 'fixed inset-0 bg-white z-50 flex flex-col';
            overlay.innerHTML = `
                <div class="flex-shrink-0 bg-white border-b border-slate-200 safe-top">
                    <div class="flex items-center p-4">
                        <button class="mr-3 p-2 -ml-2 text-slate-500 hover:text-slate-700" onclick="mobileInterface.hideGlobalSearch(this.closest('.fixed'))">
                            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path>
                            </svg>
                        </button>
                        <div class="flex-1 relative">
                            <input type="text" 
                                   id="global-search-input"
                                   placeholder="Search notes..."
                                   class="w-full px-4 py-3 pl-10 text-lg bg-slate-100 border-none rounded-full focus:outline-none focus:ring-2 focus:ring-discord-500 focus:bg-white">
                            <svg class="absolute left-3 top-3.5 h-5 w-5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                            </svg>
                        </div>
                        ${this.mobileInterface?.hasTouch ? `
                            <button class="ml-3 p-2 text-slate-500 hover:text-slate-700" onclick="contentManager.startVoiceSearch()">
                                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path>
                                </svg>
                            </button>
                        ` : ''}
                    </div>
                </div>
                <div class="flex-1 overflow-hidden">
                    <div id="global-search-results" class="h-full p-4 overflow-y-auto">
                        <div class="text-slate-500 text-center py-8">
                            Start typing to search your notes...
                        </div>
                    </div>
                </div>
            `;
            
            document.body.appendChild(overlay);
            this.searchSystem.globalSearchVisible = true;
            
            const input = overlay.querySelector('#global-search-input');
            const results = overlay.querySelector('#global-search-results');
            
            // Focus input
            setTimeout(() => input.focus(), 100);
            
            // Setup search functionality with mobile optimizations
            let searchTimeout;
            input.addEventListener('input', () => {
                clearTimeout(searchTimeout);
                searchTimeout = setTimeout(() => {
                    this.performGlobalSearch(input.value, results);
                }, 150); // Faster response for mobile
            });
            
            // Mobile-optimized keyboard handling
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    const activeResult = results.querySelector('.search-result.active');
                    if (activeResult) {
                        activeResult.click();
                    }
                }
            });
        };
    }

    startVoiceSearch() {
        if (!this.mobileInterface || !('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
            this.showNotification('Voice search not supported', 'error');
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';
        
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            const searchInput = document.getElementById('global-search-input');
            if (searchInput) {
                searchInput.value = transcript;
                searchInput.dispatchEvent(new Event('input'));
            }
        };
        
        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.showNotification('Voice search failed', 'error');
        };
        
        this.showNotification('Listening...', 'info');
        recognition.start();
    }

    optimizeMobileTagging() {
        if (!this.isMobile) return;

        // Override tag suggestions to be more touch-friendly
        const originalCreateTagSuggestions = this.createTagSuggestions;
        
        this.createTagSuggestions = (input) => {
            const container = document.createElement('div');
            container.className = 'tag-suggestions fixed inset-x-4 bg-white border border-slate-300 rounded-lg shadow-xl max-h-48 overflow-y-auto hidden z-50';
            
            // Position above keyboard if possible
            const rect = input.getBoundingClientRect();
            const spaceAbove = rect.top;
            const spaceBelow = window.innerHeight - rect.bottom;
            
            if (spaceAbove > 200 && spaceAbove > spaceBelow) {
                container.style.bottom = `${window.innerHeight - rect.top + 8}px`;
            } else {
                container.style.top = `${rect.bottom + 8}px`;
            }
            
            document.body.appendChild(container);
            return container;
        };

        // Make tag suggestions more touch-friendly
        const originalUpdateTagSuggestions = this.updateTagSuggestions;
        
        this.updateTagSuggestions = (input, container) => {
            const query = input.value.toLowerCase().trim();
            
            if (query.length === 0) {
                container.classList.add('hidden');
                return;
            }
            
            // Get matching tags (same logic)
            const matchingTags = Array.from(this.tagSystem.tags.keys())
                .filter(tag => tag.toLowerCase().includes(query))
                .sort((a, b) => {
                    const aExact = a.toLowerCase().startsWith(query);
                    const bExact = b.toLowerCase().startsWith(query);
                    const aRecent = this.tagSystem.recentTags.includes(a);
                    const bRecent = this.tagSystem.recentTags.includes(b);
                    
                    if (aExact && !bExact) return -1;
                    if (!aExact && bExact) return 1;
                    if (aRecent && !bRecent) return -1;
                    if (!aRecent && bRecent) return 1;
                    return a.localeCompare(b);
                })
                .slice(0, 8); // Show fewer suggestions on mobile
            
            // Mobile-optimized suggestion rendering with larger touch targets
            container.innerHTML = matchingTags.map((tag, index) => `
                <div class="tag-suggestion px-4 py-3 hover:bg-slate-100 cursor-pointer flex items-center justify-between border-b border-slate-100 last:border-b-0 ${index === 0 ? 'active bg-slate-50' : ''}" 
                     data-tag="${tag}">
                    <span class="flex items-center">
                        <span class="tag-color w-4 h-4 rounded-full mr-3" 
                              style="background-color: ${this.getTagColor(tag)}"></span>
                        <span class="font-medium text-base">${tag}</span>
                    </span>
                    <span class="text-sm text-slate-500">
                        ${this.getTagCount(tag)}
                    </span>
                </div>
            `).join('');
            
            container.classList.remove('hidden');
            
            // Handle selection with touch-friendly events
            container.querySelectorAll('.tag-suggestion').forEach(suggestion => {
                suggestion.addEventListener('click', () => {
                    this.selectTag(input, suggestion.dataset.tag);
                    this.mobileInterface?.triggerHapticFeedback('light');
                });
            });
        };
    }

    /**
     * MOBILE UI ENHANCEMENTS
     */
    showMobileContextMenu(noteId) {
        // This will be called by the mobile interface
        if (!this.mobileInterface) return;
        
        this.mobileInterface.showMobileContextMenu(
            document.querySelector(`[data-note-id="${noteId}"]`), 
            noteId
        );
    }

    // Override notification system for mobile
    showNotification(message, type = 'info') {
        if (this.mobileInterface && this.isMobile) {
            this.mobileInterface.showNotification(message, type);
        } else {
            // Use original notification system
            console.log(`[${type.toUpperCase()}] ${message}`);
            
            const notification = document.createElement('div');
            notification.className = `fixed top-4 right-4 px-4 py-2 rounded-lg shadow-lg z-50 ${
                type === 'success' ? 'bg-green-500 text-white' :
                type === 'error' ? 'bg-red-500 text-white' :
                'bg-blue-500 text-white'
            }`;
            notification.textContent = message;
            
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.remove();
            }, 3000);
        }
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.contentManager = new ContentManager(window.dashboardPerformance);
        
        // Connect with mobile interface when available
        if (window.mobileInterface) {
            window.contentManager.setMobileInterface(window.mobileInterface);
        } else {
            // Wait for mobile interface to initialize
            document.addEventListener('mobileInterfaceReady', () => {
                window.contentManager.setMobileInterface(window.mobileInterface);
            });
        }
    });
} else {
    window.contentManager = new ContentManager(window.dashboardPerformance);
    
    // Connect with mobile interface when available
    if (window.mobileInterface) {
        window.contentManager.setMobileInterface(window.mobileInterface);
    } else {
        // Wait for mobile interface to initialize
        document.addEventListener('mobileInterfaceReady', () => {
            window.contentManager.setMobileInterface(window.mobileInterface);
        });
    }
}