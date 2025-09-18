/**
 * Dashboard v3 Help & Documentation System
 * Interactive onboarding, contextual help, and user guidance
 * 
 * Features:
 * - First-time user guided tour
 * - Progressive feature discovery
 * - Contextual help tooltips
 * - Interactive tutorials
 * - Searchable help overlay
 * - Keyboard shortcuts reference
 * - Troubleshooting guides
 * - Achievement system for feature mastery
 * - Smart feature suggestions
 */

class DashboardHelp {
    constructor() {
        this.isVisible = false;
        this.currentTour = null;
        this.tourStep = 0;
        this.helpOverlayVisible = false;
        
        // User progress tracking
        this.userProgress = {
            onboardingCompleted: false,
            featuresUsed: new Set(),
            tourCompleted: new Set(),
            achievements: new Set(),
            helpSearchHistory: [],
            lastHelpAccessed: null
        };

        // Help content database
        this.helpContent = new Map();
        this.tutorials = new Map();
        this.achievements = new Map();
        this.contextualTips = new Map();
        
        // Feature discovery system
        this.featureDiscovery = {
            suggestions: [],
            dismissedSuggestions: new Set(),
            lastSuggestion: null
        };

        // Tour configuration
        this.tours = new Map();
        this.tourElements = new Map();

        // Help search index
        this.searchIndex = [];
        this.searchResults = [];

        // Settings
        this.settings = {
            showContextualTips: true,
            enableFeatureSuggestions: true,
            tourAnimationDuration: 300,
            tooltipDelay: 500,
            achievementNotifications: true,
            helpSearchResultsLimit: 10
        };

        this.init();
    }

    /**
     * INITIALIZATION
     */
    async init() {
        console.log('📚 Initializing Dashboard Help System');
        
        try {
            // Load user progress
            this.loadUserProgress();
            
            // Initialize help content
            this.initHelpContent();
            
            // Setup tours
            this.initTours();
            
            // Setup achievements
            this.initAchievements();
            
            // Setup contextual help
            this.initContextualHelp();
            
            // Setup keyboard shortcuts
            this.initHelpKeyboardShortcuts();
            
            // Create help UI elements
            this.createHelpUI();
            
            // Setup feature discovery
            this.initFeatureDiscovery();
            
            // Setup help search
            this.initHelpSearch();
            
            // Check if first-time user
            if (!this.userProgress.onboardingCompleted) {
                setTimeout(() => this.startOnboarding(), 2000);
            } else {
                // Show smart suggestions for returning users
                setTimeout(() => this.showSmartSuggestions(), 5000);
            }
            
            console.log('✅ Dashboard Help System initialized');
            
        } catch (error) {
            console.error('❌ Help system initialization failed:', error);
        }
    }

    loadUserProgress() {
        try {
            const stored = localStorage.getItem('dashboard_help_progress');
            if (stored) {
                const progress = JSON.parse(stored);
                this.userProgress = {
                    ...this.userProgress,
                    ...progress,
                    featuresUsed: new Set(progress.featuresUsed || []),
                    tourCompleted: new Set(progress.tourCompleted || []),
                    achievements: new Set(progress.achievements || [])
                };
            }

            const settings = localStorage.getItem('dashboard_help_settings');
            if (settings) {
                this.settings = { ...this.settings, ...JSON.parse(settings) };
            }
        } catch (error) {
            console.warn('Could not load help progress:', error);
        }
    }

    saveUserProgress() {
        try {
            const progress = {
                ...this.userProgress,
                featuresUsed: Array.from(this.userProgress.featuresUsed),
                tourCompleted: Array.from(this.userProgress.tourCompleted),
                achievements: Array.from(this.userProgress.achievements)
            };
            localStorage.setItem('dashboard_help_progress', JSON.stringify(progress));
            localStorage.setItem('dashboard_help_settings', JSON.stringify(this.settings));
        } catch (error) {
            console.warn('Could not save help progress:', error);
        }
    }

    /**
     * HELP CONTENT INITIALIZATION
     */
    initHelpContent() {
        // Main features help
        this.helpContent.set('search', {
            title: 'Search Your Notes',
            description: 'Find any note, document, or piece of information instantly',
            sections: [
                {
                    title: 'Global Search',
                    content: 'Use Cmd/Ctrl+K to access global search from anywhere. Start typing to see instant results.',
                    shortcuts: ['⌘/Ctrl + K'],
                    tips: [
                        'Use quotes for exact phrase matching: "meeting notes"',
                        'Search by date: "yesterday", "last week", "2023"',
                        'Filter by type: type:audio, type:pdf, type:image'
                    ]
                },
                {
                    title: 'Advanced Search',
                    content: 'Access advanced search filters to narrow down results by date, tags, content type, and more.',
                    shortcuts: ['⌘/Ctrl + Shift + F'],
                    tips: [
                        'Use boolean operators: AND, OR, NOT',
                        'Search within specific time ranges',
                        'Combine multiple filters for precise results'
                    ]
                }
            ],
            videoUrl: '/static/videos/search-tutorial.mp4',
            relatedTopics: ['navigation', 'shortcuts', 'organization']
        });

        this.helpContent.set('note-creation', {
            title: 'Creating Notes',
            description: 'Capture thoughts, ideas, and information in multiple formats',
            sections: [
                {
                    title: 'Quick Capture',
                    content: 'The fastest way to save a note. Just type and press Enter or Cmd/Ctrl+S to save.',
                    shortcuts: ['⌘/Ctrl + N', '⌘/Ctrl + S'],
                    tips: [
                        'Notes auto-save as you type',
                        'Use # for headings and - for bullet points',
                        'Paste images directly into notes'
                    ]
                },
                {
                    title: 'Rich Content',
                    content: 'Add images, audio recordings, PDFs, and other files to your notes.',
                    tips: [
                        'Drag and drop files directly onto the note area',
                        'Click the microphone icon to record audio',
                        'Use @ to mention other notes'
                    ]
                }
            ],
            videoUrl: '/static/videos/note-creation-tutorial.mp4',
            relatedTopics: ['organization', 'shortcuts', 'sync']
        });

        this.helpContent.set('navigation', {
            title: 'Dashboard Navigation',
            description: 'Move efficiently between different views and sections',
            sections: [
                {
                    title: 'View Switching',
                    content: 'Switch between Dashboard, Search, Notes, and Analytics views.',
                    shortcuts: ['G H', 'G S', 'G N', 'G A'],
                    tips: [
                        'Use keyboard shortcuts for quick navigation',
                        'Click view names in the sidebar',
                        'Use browser back/forward buttons'
                    ]
                },
                {
                    title: 'Quick Actions',
                    content: 'Access common actions without navigating away from your current view.',
                    shortcuts: ['A', 'R', 'Escape'],
                    tips: [
                        'Press A to add a new note from anywhere',
                        'Press R to refresh the current view',
                        'Press Escape to cancel any action'
                    ]
                }
            ],
            relatedTopics: ['shortcuts', 'search', 'productivity']
        });

        this.helpContent.set('shortcuts', {
            title: 'Keyboard Shortcuts',
            description: 'Work faster with keyboard shortcuts for every action',
            sections: [
                {
                    title: 'Essential Shortcuts',
                    content: 'The most important shortcuts to learn first.',
                    shortcuts: ['⌘/Ctrl + K', '⌘/Ctrl + N', '⌘/Ctrl + S', '⌘/Ctrl + /'],
                    tips: [
                        'Learn one new shortcut each day',
                        'Use ⌘/Ctrl+/ to see all shortcuts',
                        'Shortcuts work from any view'
                    ]
                },
                {
                    title: 'Advanced Shortcuts',
                    content: 'Power-user shortcuts for advanced functionality.',
                    shortcuts: ['G H', 'G S', 'A', 'R'],
                    tips: [
                        'G shortcuts are for "Go to" actions',
                        'Single letters work when not typing',
                        'Customize shortcuts in settings'
                    ]
                }
            ],
            relatedTopics: ['navigation', 'productivity', 'customization']
        });

        this.helpContent.set('integrations', {
            title: 'Integrations & Sync',
            description: 'Connect with Obsidian, Discord, and Apple Shortcuts',
            sections: [
                {
                    title: 'Obsidian Sync',
                    content: 'Seamlessly sync with your Obsidian vault for cross-platform access.',
                    tips: [
                        'Changes sync automatically in both directions',
                        'YAML frontmatter is preserved',
                        'Images and attachments are synced'
                    ]
                },
                {
                    title: 'Discord Integration',
                    content: 'Capture notes and ideas directly from Discord.',
                    tips: [
                        'Use slash commands to create notes',
                        'Share notes with Discord communities',
                        'Archive important Discord conversations'
                    ]
                },
                {
                    title: 'Apple Shortcuts',
                    content: 'Create notes from iOS/macOS with custom shortcuts.',
                    tips: [
                        'Voice recordings transcribe automatically',
                        'Location data is captured with notes',
                        'Share from any iOS app'
                    ]
                }
            ],
            relatedTopics: ['sync', 'mobile', 'automation']
        });

        this.helpContent.set('organization', {
            title: 'Organizing Your Knowledge',
            description: 'Keep your notes and information well-organized and discoverable',
            sections: [
                {
                    title: 'Tags and Categories',
                    content: 'Use tags to categorize and find related content.',
                    tips: [
                        'Use consistent tag naming conventions',
                        'Create tag hierarchies with slashes: work/meetings',
                        'Tags auto-complete as you type'
                    ]
                },
                {
                    title: 'Linking Notes',
                    content: 'Create connections between related notes and ideas.',
                    tips: [
                        'Use [[note title]] to link to other notes',
                        'Backlinks show connections automatically',
                        'Use aliases for different ways to reference notes'
                    ]
                }
            ],
            relatedTopics: ['search', 'productivity', 'knowledge-management']
        });

        this.buildSearchIndex();
    }

    buildSearchIndex() {
        this.searchIndex = [];
        
        for (const [key, content] of this.helpContent.entries()) {
            // Index main content
            this.searchIndex.push({
                id: key,
                type: 'help',
                title: content.title,
                description: content.description,
                content: content.description,
                keywords: [content.title, ...content.relatedTopics],
                searchText: `${content.title} ${content.description} ${content.relatedTopics.join(' ')}`
            });

            // Index sections
            content.sections.forEach((section, index) => {
                this.searchIndex.push({
                    id: `${key}-section-${index}`,
                    type: 'section',
                    parentId: key,
                    title: section.title,
                    content: section.content,
                    keywords: section.tips || [],
                    searchText: `${section.title} ${section.content} ${(section.tips || []).join(' ')}`
                });
            });
        }
    }

    /**
     * TOURS AND ONBOARDING
     */
    initTours() {
        // Welcome tour for new users
        this.tours.set('welcome', {
            title: 'Welcome to Your Second Brain',
            description: 'Let\'s take a quick tour to get you started',
            steps: [
                {
                    target: '#globalSearch',
                    title: 'Global Search',
                    content: 'Search across all your notes and documents instantly. Use ⌘/Ctrl+K to access from anywhere.',
                    position: 'bottom',
                    showNext: true,
                    highlightClass: 'help-highlight-pulse'
                },
                {
                    target: '#note',
                    title: 'Quick Note Capture',
                    content: 'Capture thoughts quickly here. Notes auto-save as you type, and you can add rich content like images and audio.',
                    position: 'top',
                    showNext: true,
                    actions: [
                        {
                            text: 'Try typing a note',
                            action: () => document.getElementById('note')?.focus()
                        }
                    ]
                },
                {
                    target: '.nav-dashboard',
                    title: 'Navigation',
                    content: 'Switch between views using the sidebar or keyboard shortcuts like G+H for home.',
                    position: 'right',
                    showNext: true
                },
                {
                    target: '.recent-notes',
                    title: 'Recent Activity',
                    content: 'Your recent notes and activity appear here. Click any item to open it.',
                    position: 'left',
                    showNext: true
                },
                {
                    target: 'body',
                    title: 'You\'re All Set!',
                    content: 'That\'s the basics! Press ⌘/Ctrl+/ anytime to get help, or explore the advanced features as you go.',
                    position: 'center',
                    showNext: false,
                    final: true
                }
            ]
        });

        // Advanced features tour
        this.tours.set('advanced-features', {
            title: 'Advanced Features Tour',
            description: 'Discover powerful features to enhance your workflow',
            steps: [
                {
                    target: '#advancedSearchInput',
                    title: 'Advanced Search',
                    content: 'Use filters, date ranges, and boolean operators for precise search results.',
                    position: 'bottom',
                    showNext: true
                },
                {
                    target: '.drag-drop-zone',
                    title: 'Drag & Drop',
                    content: 'Drag files directly onto the dashboard to add them to your knowledge base.',
                    position: 'top',
                    showNext: true
                },
                {
                    target: '.analytics-section',
                    title: 'Analytics',
                    content: 'Track your knowledge growth and discover patterns in your notes.',
                    position: 'left',
                    showNext: true
                }
            ]
        });

        // Integration setup tour
        this.tours.set('integrations', {
            title: 'Setup Integrations',
            description: 'Connect with Obsidian, Discord, and more',
            steps: [
                {
                    target: '.obsidian-sync-status',
                    title: 'Obsidian Integration',
                    content: 'Connect your Obsidian vault for seamless sync across devices.',
                    position: 'bottom',
                    showNext: true
                },
                {
                    target: '.discord-status',
                    title: 'Discord Bot',
                    content: 'Use Discord slash commands to capture notes on the go.',
                    position: 'top',
                    showNext: true
                }
            ]
        });
    }

    async startOnboarding() {
        if (this.userProgress.onboardingCompleted) {
            return;
        }

        const shouldStart = await this.showOnboardingPrompt();
        if (shouldStart) {
            this.startTour('welcome');
        }
    }

    showOnboardingPrompt() {
        return new Promise((resolve) => {
            const modal = document.createElement('div');
            modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 animate-fade-in';
            
            modal.innerHTML = `
                <div class="bg-white dark:bg-slate-800 rounded-xl p-8 max-w-md mx-4 shadow-2xl transform animate-scale-in">
                    <div class="text-center">
                        <div class="text-6xl mb-4">👋</div>
                        <h2 class="text-2xl font-bold mb-4 text-slate-900 dark:text-slate-100">
                            Welcome to Your Second Brain!
                        </h2>
                        <p class="text-slate-600 dark:text-slate-300 mb-6">
                            Would you like a quick tour to get started? It only takes 2 minutes.
                        </p>
                        <div class="flex space-x-4">
                            <button id="skipTour" class="flex-1 px-6 py-3 border border-slate-300 dark:border-slate-600 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors">
                                Skip for now
                            </button>
                            <button id="startTour" class="flex-1 px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors">
                                Start Tour
                            </button>
                        </div>
                    </div>
                </div>
            `;

            document.body.appendChild(modal);

            document.getElementById('startTour').addEventListener('click', () => {
                document.body.removeChild(modal);
                resolve(true);
            });

            document.getElementById('skipTour').addEventListener('click', () => {
                document.body.removeChild(modal);
                this.userProgress.onboardingCompleted = true;
                this.saveUserProgress();
                resolve(false);
            });
        });
    }

    startTour(tourId) {
        const tour = this.tours.get(tourId);
        if (!tour) return;

        this.currentTour = tour;
        this.tourStep = 0;
        this.showTourStep();
    }

    showTourStep() {
        if (!this.currentTour || this.tourStep >= this.currentTour.steps.length) {
            this.completeTour();
            return;
        }

        const step = this.currentTour.steps[this.tourStep];
        const targetElement = step.target === 'body' ? document.body : document.querySelector(step.target);

        if (!targetElement && step.target !== 'body') {
            console.warn(`Tour target not found: ${step.target}`);
            this.nextTourStep();
            return;
        }

        this.showTourTooltip(targetElement, step);
    }

    showTourTooltip(targetElement, step) {
        // Remove existing tour elements
        this.removeTourElements();

        // Create overlay
        const overlay = document.createElement('div');
        overlay.className = 'help-tour-overlay fixed inset-0 bg-black bg-opacity-60 z-40';
        document.body.appendChild(overlay);

        // Highlight target element
        if (targetElement !== document.body) {
            const rect = targetElement.getBoundingClientRect();
            const highlight = document.createElement('div');
            highlight.className = `help-tour-highlight fixed z-50 rounded-lg ${step.highlightClass || 'help-highlight-default'}`;
            highlight.style.left = `${rect.left - 8}px`;
            highlight.style.top = `${rect.top - 8}px`;
            highlight.style.width = `${rect.width + 16}px`;
            highlight.style.height = `${rect.height + 16}px`;
            document.body.appendChild(highlight);
            
            this.tourElements.set('highlight', highlight);
        }

        // Create tooltip
        const tooltip = this.createTourTooltip(step, targetElement);
        document.body.appendChild(tooltip);

        this.tourElements.set('overlay', overlay);
        this.tourElements.set('tooltip', tooltip);

        // Execute any step actions
        if (step.actions) {
            step.actions.forEach(action => {
                setTimeout(() => action.action(), 500);
            });
        }
    }

    createTourTooltip(step, targetElement) {
        const tooltip = document.createElement('div');
        tooltip.className = 'help-tour-tooltip fixed z-50 bg-white dark:bg-slate-800 rounded-xl shadow-2xl p-6 max-w-sm animate-scale-in';

        // Position tooltip
        if (targetElement !== document.body) {
            const rect = targetElement.getBoundingClientRect();
            const tooltipPosition = this.calculateTooltipPosition(rect, step.position);
            tooltip.style.left = `${tooltipPosition.x}px`;
            tooltip.style.top = `${tooltipPosition.y}px`;
        } else {
            // Center on screen
            tooltip.className += ' top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2';
        }

        tooltip.innerHTML = `
            <div class="mb-4">
                <h3 class="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">
                    ${step.title}
                </h3>
                <p class="text-slate-600 dark:text-slate-300">
                    ${step.content}
                </p>
            </div>
            
            ${step.actions ? `
                <div class="mb-4 space-y-2">
                    ${step.actions.map(action => `
                        <button class="help-tour-action w-full px-4 py-2 bg-blue-50 dark:bg-blue-900 text-blue-600 dark:text-blue-300 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-800 transition-colors text-sm">
                            ${action.text}
                        </button>
                    `).join('')}
                </div>
            ` : ''}
            
            <div class="flex justify-between items-center">
                <div class="flex space-x-2">
                    <span class="text-sm text-slate-500">
                        ${this.tourStep + 1} of ${this.currentTour.steps.length}
                    </span>
                </div>
                <div class="flex space-x-2">
                    ${this.tourStep > 0 ? `
                        <button id="tourPrev" class="px-4 py-2 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 transition-colors">
                            Previous
                        </button>
                    ` : ''}
                    
                    ${step.showNext ? `
                        <button id="tourNext" class="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors">
                            ${step.final ? 'Complete' : 'Next'}
                        </button>
                    ` : ''}
                    
                    <button id="tourSkip" class="px-4 py-2 text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 transition-colors">
                        ${step.final ? 'Done' : 'Skip'}
                    </button>
                </div>
            </div>
        `;

        // Bind event handlers
        const nextBtn = tooltip.querySelector('#tourNext');
        const prevBtn = tooltip.querySelector('#tourPrev');
        const skipBtn = tooltip.querySelector('#tourSkip');

        if (nextBtn) {
            nextBtn.addEventListener('click', () => this.nextTourStep());
        }
        if (prevBtn) {
            prevBtn.addEventListener('click', () => this.prevTourStep());
        }
        if (skipBtn) {
            skipBtn.addEventListener('click', () => this.completeTour());
        }

        // Bind action buttons
        const actionButtons = tooltip.querySelectorAll('.help-tour-action');
        actionButtons.forEach((button, index) => {
            if (step.actions && step.actions[index]) {
                button.addEventListener('click', step.actions[index].action);
            }
        });

        return tooltip;
    }

    calculateTooltipPosition(targetRect, position) {
        const tooltipWidth = 384; // max-w-sm
        const tooltipHeight = 200; // estimated
        const margin = 16;

        let x, y;

        switch (position) {
            case 'top':
                x = targetRect.left + (targetRect.width / 2) - (tooltipWidth / 2);
                y = targetRect.top - tooltipHeight - margin;
                break;
            case 'bottom':
                x = targetRect.left + (targetRect.width / 2) - (tooltipWidth / 2);
                y = targetRect.bottom + margin;
                break;
            case 'left':
                x = targetRect.left - tooltipWidth - margin;
                y = targetRect.top + (targetRect.height / 2) - (tooltipHeight / 2);
                break;
            case 'right':
                x = targetRect.right + margin;
                y = targetRect.top + (targetRect.height / 2) - (tooltipHeight / 2);
                break;
            default:
                x = targetRect.left + (targetRect.width / 2) - (tooltipWidth / 2);
                y = targetRect.bottom + margin;
        }

        // Keep tooltip within viewport
        x = Math.max(margin, Math.min(x, window.innerWidth - tooltipWidth - margin));
        y = Math.max(margin, Math.min(y, window.innerHeight - tooltipHeight - margin));

        return { x, y };
    }

    nextTourStep() {
        this.tourStep++;
        this.showTourStep();
    }

    prevTourStep() {
        if (this.tourStep > 0) {
            this.tourStep--;
            this.showTourStep();
        }
    }

    completeTour() {
        this.removeTourElements();
        
        if (this.currentTour) {
            const tourId = Array.from(this.tours.entries())
                .find(([_, tour]) => tour === this.currentTour)?.[0];
                
            if (tourId) {
                this.userProgress.tourCompleted.add(tourId);
                
                if (tourId === 'welcome') {
                    this.userProgress.onboardingCompleted = true;
                    this.showOnboardingComplete();
                }
                
                this.saveUserProgress();
            }
        }

        this.currentTour = null;
        this.tourStep = 0;
    }

    removeTourElements() {
        for (const [name, element] of this.tourElements.entries()) {
            if (element.parentNode) {
                element.parentNode.removeChild(element);
            }
        }
        this.tourElements.clear();
    }

    showOnboardingComplete() {
        const notification = document.createElement('div');
        notification.className = 'fixed top-4 right-4 bg-green-500 text-white p-4 rounded-lg shadow-lg z-50 max-w-sm animate-slide-in';
        
        notification.innerHTML = `
            <div class="flex items-center">
                <span class="text-2xl mr-3">🎉</span>
                <div>
                    <div class="font-semibold">Welcome aboard!</div>
                    <div class="text-sm opacity-90">You've completed the onboarding tour. Press ⌘/Ctrl+/ anytime for help.</div>
                </div>
            </div>
        `;

        document.body.appendChild(notification);

        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);

        // Unlock achievement
        this.unlockAchievement('onboarding-complete');
    }

    /**
     * CONTEXTUAL HELP AND TOOLTIPS
     */
    initContextualHelp() {
        console.log('💡 Initializing contextual help');

        // Setup contextual tips for UI elements
        this.setupContextualTips();
        
        // Monitor user interactions for help opportunities
        this.setupHelpOpportunityDetection();
        
        // Setup progressive disclosure
        this.setupProgressiveDisclosure();
    }

    setupContextualTips() {
        // Register contextual tips for various UI elements
        this.contextualTips.set('#globalSearch', {
            title: 'Global Search',
            content: 'Search across all your notes instantly. Try typing to see suggestions.',
            trigger: 'focus',
            position: 'bottom',
            showOnce: false
        });

        this.contextualTips.set('#note', {
            title: 'Quick Note',
            content: 'Type here to create a note. It auto-saves as you type. Use Markdown for formatting.',
            trigger: 'focus',
            position: 'top',
            showOnce: true
        });

        this.contextualTips.set('.drag-drop-zone', {
            title: 'Drag & Drop Files',
            content: 'Drop files here to add them to your knowledge base. Supports images, PDFs, audio, and more.',
            trigger: 'dragenter',
            position: 'center',
            showOnce: true
        });

        this.contextualTips.set('.analytics-section', {
            title: 'Knowledge Analytics',
            content: 'Track your learning progress and discover patterns in your notes.',
            trigger: 'mouseenter',
            position: 'left',
            showOnce: true
        });

        // Bind contextual tips to elements
        this.bindContextualTips();
    }

    bindContextualTips() {
        for (const [selector, tip] of this.contextualTips.entries()) {
            const elements = document.querySelectorAll(selector);
            
            elements.forEach(element => {
                const tipId = `tip-${selector.replace(/[^a-zA-Z0-9]/g, '-')}`;
                const hasShown = localStorage.getItem(tipId);
                
                if (tip.showOnce && hasShown) return;

                const showTip = () => {
                    if (!this.settings.showContextualTips) return;
                    
                    setTimeout(() => {
                        this.showContextualTip(element, tip, tipId);
                    }, this.settings.tooltipDelay);
                };

                const hideTip = () => {
                    this.hideContextualTip(tipId);
                };

                switch (tip.trigger) {
                    case 'focus':
                        element.addEventListener('focus', showTip);
                        element.addEventListener('blur', hideTip);
                        break;
                    case 'mouseenter':
                        element.addEventListener('mouseenter', showTip);
                        element.addEventListener('mouseleave', hideTip);
                        break;
                    case 'dragenter':
                        element.addEventListener('dragenter', showTip);
                        element.addEventListener('dragleave', hideTip);
                        break;
                }
            });
        }
    }

    showContextualTip(element, tip, tipId) {
        // Don't show if help overlay is visible
        if (this.helpOverlayVisible || this.currentTour) return;

        // Remove existing tip
        this.hideContextualTip(tipId);

        const tooltip = document.createElement('div');
        tooltip.id = tipId;
        tooltip.className = 'help-contextual-tip absolute z-50 bg-slate-900 text-white text-sm rounded-lg p-3 max-w-xs shadow-xl animate-fade-in';
        
        tooltip.innerHTML = `
            <div class="flex items-start justify-between">
                <div>
                    <div class="font-medium mb-1">${tip.title}</div>
                    <div class="text-slate-300">${tip.content}</div>
                </div>
                <button class="ml-2 text-slate-400 hover:text-white opacity-70 hover:opacity-100 transition-opacity">
                    <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                    </svg>
                </button>
            </div>
        `;

        // Position tooltip
        document.body.appendChild(tooltip);
        const rect = element.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        const position = this.calculateTooltipPosition(rect, tip.position);
        
        tooltip.style.left = `${position.x}px`;
        tooltip.style.top = `${position.y}px`;

        // Bind close button
        const closeBtn = tooltip.querySelector('button');
        closeBtn.addEventListener('click', () => {
            this.hideContextualTip(tipId);
            if (tip.showOnce) {
                localStorage.setItem(tipId, 'shown');
            }
        });

        // Auto-hide after delay
        setTimeout(() => {
            this.hideContextualTip(tipId);
        }, 8000);
    }

    hideContextualTip(tipId) {
        const tooltip = document.getElementById(tipId);
        if (tooltip) {
            tooltip.remove();
        }
    }

    setupHelpOpportunityDetection() {
        // Detect when users might need help
        let consecutiveFailedSearches = 0;
        let noActivityTime = 0;
        let stuckOnSameView = 0;
        let currentView = null;

        // Monitor search failures
        document.addEventListener('search-no-results', () => {
            consecutiveFailedSearches++;
            if (consecutiveFailedSearches >= 3) {
                this.suggestSearchHelp();
                consecutiveFailedSearches = 0;
            }
        });

        // Monitor view changes
        document.addEventListener('view-changed', (event) => {
            if (currentView === event.detail.view) {
                stuckOnSameView++;
                if (stuckOnSameView >= 10) {
                    this.suggestNavigationHelp();
                    stuckOnSameView = 0;
                }
            } else {
                stuckOnSameView = 0;
                currentView = event.detail.view;
            }
        });

        // Monitor user activity
        let lastActivity = Date.now();
        ['click', 'keydown', 'scroll'].forEach(event => {
            document.addEventListener(event, () => {
                lastActivity = Date.now();
                noActivityTime = 0;
            }, { passive: true });
        });

        // Check for inactivity
        setInterval(() => {
            noActivityTime = Date.now() - lastActivity;
            if (noActivityTime > 60000) { // 1 minute of inactivity
                this.suggestNextSteps();
            }
        }, 30000);
    }

    suggestSearchHelp() {
        if (!this.settings.enableFeatureSuggestions) return;

        this.showSmartSuggestion({
            title: 'Having trouble finding what you need?',
            content: 'Try using advanced search with filters, or check out our search tips.',
            actions: [
                {
                    text: 'Search Tips',
                    action: () => this.showHelp('search')
                },
                {
                    text: 'Advanced Search',
                    action: () => this.focusAdvancedSearch()
                }
            ],
            type: 'help-suggestion',
            timeout: 10000
        });
    }

    suggestNavigationHelp() {
        this.showSmartSuggestion({
            title: 'Try keyboard shortcuts for faster navigation',
            content: 'Use G+H for Dashboard, G+S for Search, or ⌘/Ctrl+K for global search.',
            actions: [
                {
                    text: 'Show All Shortcuts',
                    action: () => this.showKeyboardShortcuts()
                }
            ],
            type: 'productivity-tip',
            timeout: 8000
        });
    }

    suggestNextSteps() {
        const hour = new Date().getHours();
        let suggestion;

        if (hour < 12) {
            suggestion = {
                title: 'Good morning! Ready to capture some ideas?',
                content: 'Start your day by jotting down your goals or reviewing yesterday\'s notes.',
                actions: [
                    {
                        text: 'New Note',
                        action: () => document.getElementById('note')?.focus()
                    },
                    {
                        text: 'Search Recent',
                        action: () => this.searchRecent()
                    }
                ]
            };
        } else if (hour >= 17) {
            suggestion = {
                title: 'End of day reflection',
                content: 'Consider capturing key insights from today or planning for tomorrow.',
                actions: [
                    {
                        text: 'Daily Reflection',
                        action: () => this.createDailyReflection()
                    }
                ]
            };
        } else {
            return; // No suggestions during work hours unless specifically triggered
        }

        this.showSmartSuggestion({
            ...suggestion,
            type: 'daily-suggestion',
            timeout: 12000
        });
    }

    /**
     * HELP OVERLAY AND SEARCH
     */
    initHelpSearch() {
        this.searchResults = [];
    }

    createHelpUI() {
        // Create help button
        const helpButton = document.createElement('button');
        helpButton.id = 'helpButton';
        helpButton.className = 'fixed bottom-4 left-4 bg-blue-500 hover:bg-blue-600 text-white rounded-full p-3 shadow-lg z-30 transition-all duration-200';
        helpButton.innerHTML = `
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
        `;
        
        helpButton.addEventListener('click', () => this.toggleHelpOverlay());
        document.body.appendChild(helpButton);

        // Create help overlay (hidden by default)
        this.createHelpOverlay();
    }

    createHelpOverlay() {
        const overlay = document.createElement('div');
        overlay.id = 'helpOverlay';
        overlay.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 hidden';
        
        overlay.innerHTML = `
            <div class="bg-white dark:bg-slate-800 rounded-xl shadow-2xl w-full max-w-4xl h-5/6 flex flex-col mx-4">
                <!-- Header -->
                <div class="flex items-center justify-between p-6 border-b border-slate-200 dark:border-slate-700">
                    <h2 class="text-2xl font-bold text-slate-900 dark:text-slate-100">
                        Help & Documentation
                    </h2>
                    <button id="closeHelp" class="text-slate-500 hover:text-slate-700 dark:hover:text-slate-300">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                        </svg>
                    </button>
                </div>

                <!-- Search -->
                <div class="p-6 border-b border-slate-200 dark:border-slate-700">
                    <div class="relative">
                        <input 
                            type="text" 
                            id="helpSearch" 
                            placeholder="Search help articles, tutorials, and shortcuts..."
                            class="w-full px-4 py-3 pl-12 bg-slate-100 dark:bg-slate-700 border-0 rounded-lg focus:ring-2 focus:ring-blue-500 text-slate-900 dark:text-slate-100"
                        />
                        <svg class="absolute left-4 top-4 w-5 h-5 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                        </svg>
                    </div>
                </div>

                <div class="flex flex-1 overflow-hidden">
                    <!-- Sidebar -->
                    <div class="w-1/3 border-r border-slate-200 dark:border-slate-700 overflow-y-auto">
                        <div id="helpSidebar" class="p-4">
                            <!-- Navigation will be populated here -->
                        </div>
                    </div>

                    <!-- Content -->
                    <div class="flex-1 overflow-y-auto">
                        <div id="helpContent" class="p-6">
                            <!-- Content will be populated here -->
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(overlay);

        // Bind event handlers
        this.bindHelpOverlayEvents();
        
        // Populate initial content
        this.populateHelpSidebar();
        this.showHelpHome();
    }

    bindHelpOverlayEvents() {
        const overlay = document.getElementById('helpOverlay');
        const closeBtn = document.getElementById('closeHelp');
        const searchInput = document.getElementById('helpSearch');

        // Close help overlay
        closeBtn.addEventListener('click', () => this.hideHelpOverlay());
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                this.hideHelpOverlay();
            }
        });

        // Search functionality
        let searchTimeout;
        searchInput.addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                this.performHelpSearch(e.target.value);
            }, 300);
        });

        // Keyboard shortcuts in help overlay
        overlay.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hideHelpOverlay();
            }
        });
    }

    toggleHelpOverlay() {
        if (this.helpOverlayVisible) {
            this.hideHelpOverlay();
        } else {
            this.showHelpOverlay();
        }
    }

    showHelpOverlay() {
        const overlay = document.getElementById('helpOverlay');
        overlay.classList.remove('hidden');
        this.helpOverlayVisible = true;
        
        // Focus search input
        setTimeout(() => {
            document.getElementById('helpSearch')?.focus();
        }, 100);

        // Track help usage
        this.userProgress.lastHelpAccessed = Date.now();
        this.saveUserProgress();
    }

    hideHelpOverlay() {
        const overlay = document.getElementById('helpOverlay');
        overlay.classList.add('hidden');
        this.helpOverlayVisible = false;
    }

    populateHelpSidebar() {
        const sidebar = document.getElementById('helpSidebar');
        
        const categories = [
            { title: 'Getting Started', icon: '🚀', items: ['welcome', 'search', 'note-creation'] },
            { title: 'Core Features', icon: '⚡', items: ['navigation', 'shortcuts', 'organization'] },
            { title: 'Integrations', icon: '🔗', items: ['integrations'] },
            { title: 'Advanced', icon: '🛠️', items: ['customization', 'troubleshooting'] }
        ];

        let sidebarHTML = `
            <div class="space-y-1 mb-6">
                <button class="help-nav-item w-full text-left px-3 py-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 flex items-center space-x-3 text-slate-700 dark:text-slate-300" data-section="home">
                    <span class="text-xl">🏠</span>
                    <span class="font-medium">Home</span>
                </button>
                <button class="help-nav-item w-full text-left px-3 py-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 flex items-center space-x-3 text-slate-700 dark:text-slate-300" data-section="tours">
                    <span class="text-xl">🎯</span>
                    <span class="font-medium">Tours & Tutorials</span>
                </button>
                <button class="help-nav-item w-full text-left px-3 py-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 flex items-center space-x-3 text-slate-700 dark:text-slate-300" data-section="shortcuts">
                    <span class="text-xl">⌨️</span>
                    <span class="font-medium">Keyboard Shortcuts</span>
                </button>
            </div>
        `;

        categories.forEach(category => {
            sidebarHTML += `
                <div class="mb-4">
                    <div class="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2 flex items-center">
                        <span class="mr-2">${category.icon}</span>
                        ${category.title}
                    </div>
                    <div class="space-y-1">
                        ${category.items.map(item => {
                            const content = this.helpContent.get(item);
                            if (!content) return '';
                            return `
                                <button class="help-nav-item w-full text-left px-3 py-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300" data-section="${item}">
                                    ${content.title}
                                </button>
                            `;
                        }).join('')}
                    </div>
                </div>
            `;
        });

        sidebar.innerHTML = sidebarHTML;

        // Bind navigation events
        sidebar.addEventListener('click', (e) => {
            const button = e.target.closest('.help-nav-item');
            if (button) {
                const section = button.dataset.section;
                this.showHelpSection(section);
                
                // Update active state
                sidebar.querySelectorAll('.help-nav-item').forEach(item => {
                    item.classList.remove('bg-blue-100', 'dark:bg-blue-900', 'text-blue-600', 'dark:text-blue-300');
                });
                button.classList.add('bg-blue-100', 'dark:bg-blue-900', 'text-blue-600', 'dark:text-blue-300');
            }
        });
    }

    showHelpHome() {
        const content = document.getElementById('helpContent');
        content.innerHTML = `
            <div class="max-w-none">
                <h1 class="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-6">
                    Welcome to Your Second Brain
                </h1>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                    <div class="bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl p-6 text-white">
                        <h3 class="text-xl font-semibold mb-2">Quick Start</h3>
                        <p class="mb-4 opacity-90">New to the dashboard? Take a guided tour to learn the basics.</p>
                        <button id="startWelcomeTour" class="bg-white bg-opacity-20 hover:bg-opacity-30 px-4 py-2 rounded-lg transition-colors">
                            Start Tour
                        </button>
                    </div>
                    
                    <div class="bg-gradient-to-br from-green-500 to-teal-600 rounded-xl p-6 text-white">
                        <h3 class="text-xl font-semibold mb-2">Keyboard Shortcuts</h3>
                        <p class="mb-4 opacity-90">Work faster with keyboard shortcuts for every action.</p>
                        <button id="showShortcuts" class="bg-white bg-opacity-20 hover:bg-opacity-30 px-4 py-2 rounded-lg transition-colors">
                            View Shortcuts
                        </button>
                    </div>
                </div>

                <div class="mb-8">
                    <h2 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">Popular Help Topics</h2>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                        ${['search', 'note-creation', 'navigation'].map(topic => {
                            const content = this.helpContent.get(topic);
                            return `
                                <div class="help-topic-card border border-slate-200 dark:border-slate-700 rounded-lg p-4 hover:bg-slate-50 dark:hover:bg-slate-700 cursor-pointer transition-colors" data-topic="${topic}">
                                    <h4 class="font-medium text-slate-900 dark:text-slate-100 mb-2">${content?.title}</h4>
                                    <p class="text-sm text-slate-600 dark:text-slate-400">${content?.description}</p>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>

                <div class="mb-8">
                    <h2 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">Your Progress</h2>
                    <div class="bg-slate-100 dark:bg-slate-700 rounded-lg p-4">
                        <div class="flex items-center justify-between mb-2">
                            <span class="text-slate-700 dark:text-slate-300">Features Explored</span>
                            <span class="font-medium text-slate-900 dark:text-slate-100">${this.userProgress.featuresUsed.size} / 10</span>
                        </div>
                        <div class="w-full bg-slate-300 dark:bg-slate-600 rounded-full h-2">
                            <div class="bg-blue-500 h-2 rounded-full" style="width: ${(this.userProgress.featuresUsed.size / 10) * 100}%"></div>
                        </div>
                        <div class="mt-2 text-sm text-slate-600 dark:text-slate-400">
                            Tours completed: ${this.userProgress.tourCompleted.size}
                        </div>
                    </div>
                </div>

                <div>
                    <h2 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">Need More Help?</h2>
                    <div class="flex flex-wrap gap-2">
                        <button class="help-action-btn px-4 py-2 bg-slate-200 dark:bg-slate-700 rounded-lg hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors" data-action="feedback">
                            Send Feedback
                        </button>
                        <button class="help-action-btn px-4 py-2 bg-slate-200 dark:bg-slate-700 rounded-lg hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors" data-action="reset-progress">
                            Reset Progress
                        </button>
                        <button class="help-action-btn px-4 py-2 bg-slate-200 dark:bg-slate-700 rounded-lg hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors" data-action="export-data">
                            Export Help Data
                        </button>
                    </div>
                </div>
            </div>
        `;

        // Bind event handlers
        content.querySelector('#startWelcomeTour')?.addEventListener('click', () => {
            this.hideHelpOverlay();
            this.startTour('welcome');
        });

        content.querySelector('#showShortcuts')?.addEventListener('click', () => {
            this.showHelpSection('shortcuts');
        });

        content.querySelectorAll('.help-topic-card').forEach(card => {
            card.addEventListener('click', () => {
                const topic = card.dataset.topic;
                this.showHelpSection(topic);
            });
        });

        content.querySelectorAll('.help-action-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const action = btn.dataset.action;
                this.handleHelpAction(action);
            });
        });
    }

    showHelpSection(sectionId) {
        const content = document.getElementById('helpContent');
        
        if (sectionId === 'home') {
            this.showHelpHome();
            return;
        }

        if (sectionId === 'shortcuts') {
            this.showKeyboardShortcutsHelp();
            return;
        }

        if (sectionId === 'tours') {
            this.showToursHelp();
            return;
        }

        const helpItem = this.helpContent.get(sectionId);
        if (!helpItem) return;

        content.innerHTML = `
            <div class="max-w-none">
                <div class="mb-6">
                    <h1 class="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-2">
                        ${helpItem.title}
                    </h1>
                    <p class="text-xl text-slate-600 dark:text-slate-400">
                        ${helpItem.description}
                    </p>
                </div>

                ${helpItem.videoUrl ? `
                    <div class="mb-8">
                        <div class="bg-slate-100 dark:bg-slate-700 rounded-lg p-4 text-center">
                            <p class="text-slate-600 dark:text-slate-400">Video tutorial would be embedded here</p>
                            <p class="text-sm text-slate-500 mt-2">${helpItem.videoUrl}</p>
                        </div>
                    </div>
                ` : ''}

                <div class="space-y-8">
                    ${helpItem.sections.map(section => `
                        <div>
                            <h2 class="text-2xl font-semibold text-slate-900 dark:text-slate-100 mb-4">
                                ${section.title}
                            </h2>
                            <p class="text-slate-700 dark:text-slate-300 mb-4">
                                ${section.content}
                            </p>

                            ${section.shortcuts && section.shortcuts.length > 0 ? `
                                <div class="mb-4">
                                    <h4 class="font-medium text-slate-900 dark:text-slate-100 mb-2">Keyboard Shortcuts:</h4>
                                    <div class="flex flex-wrap gap-2">
                                        ${section.shortcuts.map(shortcut => `
                                            <kbd class="px-2 py-1 bg-slate-200 dark:bg-slate-700 rounded text-sm font-mono">${shortcut}</kbd>
                                        `).join('')}
                                    </div>
                                </div>
                            ` : ''}

                            ${section.tips && section.tips.length > 0 ? `
                                <div class="bg-blue-50 dark:bg-blue-900 rounded-lg p-4">
                                    <h4 class="font-medium text-blue-900 dark:text-blue-100 mb-2">💡 Tips:</h4>
                                    <ul class="space-y-1 text-blue-800 dark:text-blue-200">
                                        ${section.tips.map(tip => `<li>• ${tip}</li>`).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                        </div>
                    `).join('')}
                </div>

                ${helpItem.relatedTopics && helpItem.relatedTopics.length > 0 ? `
                    <div class="mt-8 pt-8 border-t border-slate-200 dark:border-slate-700">
                        <h3 class="font-semibold text-slate-900 dark:text-slate-100 mb-4">Related Topics:</h3>
                        <div class="flex flex-wrap gap-2">
                            ${helpItem.relatedTopics.map(topic => {
                                const relatedContent = this.helpContent.get(topic);
                                return relatedContent ? `
                                    <button class="help-related-topic px-3 py-2 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 rounded-lg text-sm transition-colors" data-topic="${topic}">
                                        ${relatedContent.title}
                                    </button>
                                ` : '';
                            }).join('')}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;

        // Bind related topic links
        content.querySelectorAll('.help-related-topic').forEach(link => {
            link.addEventListener('click', () => {
                const topic = link.dataset.topic;
                this.showHelpSection(topic);
            });
        });

        // Track feature usage
        this.trackFeatureUsage(`help-${sectionId}`);
    }

    showKeyboardShortcutsHelp() {
        const content = document.getElementById('helpContent');
        
        // Get shortcuts from dashboard performance system if available
        let shortcuts = [];
        if (window.dashboardPerformance && window.dashboardPerformance.shortcuts) {
            shortcuts = Array.from(window.dashboardPerformance.shortcuts.entries());
        }

        content.innerHTML = `
            <div class="max-w-none">
                <h1 class="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-6">
                    Keyboard Shortcuts
                </h1>

                ${shortcuts.length > 0 ? `
                    <div class="space-y-6">
                        ${this.groupShortcutsByCategory(shortcuts).map(category => `
                            <div>
                                <h2 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">
                                    ${category.name}
                                </h2>
                                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    ${category.shortcuts.map(([shortcut, config]) => `
                                        <div class="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-700 rounded-lg">
                                            <span class="text-slate-700 dark:text-slate-300">${config.description}</span>
                                            <kbd class="px-2 py-1 bg-slate-200 dark:bg-slate-600 rounded text-sm font-mono">
                                                ${shortcut.split(',')[0].replace(/ctrl|cmd/g, '⌘').replace(/shift/g, '⇧').replace(/\+/g, ' + ').toUpperCase()}
                                            </kbd>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        `).join('')}
                    </div>
                ` : `
                    <div class="text-center py-8">
                        <p class="text-slate-600 dark:text-slate-400 mb-4">Keyboard shortcuts system not loaded yet.</p>
                        <button id="reloadShortcuts" class="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors">
                            Reload Shortcuts
                        </button>
                    </div>
                `}

                <div class="mt-8 p-4 bg-yellow-50 dark:bg-yellow-900 rounded-lg">
                    <h3 class="font-medium text-yellow-900 dark:text-yellow-100 mb-2">💡 Pro Tips:</h3>
                    <ul class="space-y-1 text-yellow-800 dark:text-yellow-200 text-sm">
                        <li>• Learn one new shortcut each day for gradual mastery</li>
                        <li>• Shortcuts work from any view in the dashboard</li>
                        <li>• Use ⌘/Ctrl+/ to quickly access this shortcuts reference</li>
                        <li>• Single-letter shortcuts only work when not typing in input fields</li>
                    </ul>
                </div>
            </div>
        `;

        // Bind reload button if present
        content.querySelector('#reloadShortcuts')?.addEventListener('click', () => {
            setTimeout(() => this.showKeyboardShortcutsHelp(), 1000);
        });
    }

    groupShortcutsByCategory(shortcuts) {
        const categories = {
            navigation: { name: '🧭 Navigation', shortcuts: [] },
            search: { name: '🔍 Search', shortcuts: [] },
            actions: { name: '⚡ Actions', shortcuts: [] },
            views: { name: '👁️ Views', shortcuts: [] },
            other: { name: '🔧 Other', shortcuts: [] }
        };

        shortcuts.forEach(([shortcut, config]) => {
            const description = config.description.toLowerCase();
            
            if (description.includes('search')) {
                categories.search.shortcuts.push([shortcut, config]);
            } else if (description.includes('go to') || description.includes('navigation')) {
                categories.navigation.shortcuts.push([shortcut, config]);
            } else if (description.includes('view') || description.includes('show')) {
                categories.views.shortcuts.push([shortcut, config]);
            } else if (description.includes('new') || description.includes('save') || description.includes('add')) {
                categories.actions.shortcuts.push([shortcut, config]);
            } else {
                categories.other.shortcuts.push([shortcut, config]);
            }
        });

        return Object.values(categories).filter(cat => cat.shortcuts.length > 0);
    }

    showToursHelp() {
        const content = document.getElementById('helpContent');
        
        content.innerHTML = `
            <div class="max-w-none">
                <h1 class="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-6">
                    Tours & Tutorials
                </h1>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    ${Array.from(this.tours.entries()).map(([tourId, tour]) => {
                        const isCompleted = this.userProgress.tourCompleted.has(tourId);
                        return `
                            <div class="border border-slate-200 dark:border-slate-700 rounded-lg p-6 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors">
                                <div class="flex items-center justify-between mb-3">
                                    <h3 class="text-lg font-semibold text-slate-900 dark:text-slate-100">
                                        ${tour.title}
                                    </h3>
                                    ${isCompleted ? `
                                        <span class="px-2 py-1 bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 text-xs font-medium rounded-full">
                                            ✓ Completed
                                        </span>
                                    ` : ''}
                                </div>
                                <p class="text-slate-600 dark:text-slate-400 mb-4">
                                    ${tour.description}
                                </p>
                                <div class="flex items-center justify-between">
                                    <span class="text-sm text-slate-500">
                                        ${tour.steps.length} steps
                                    </span>
                                    <button class="help-start-tour px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors text-sm" data-tour="${tourId}">
                                        ${isCompleted ? 'Replay' : 'Start'} Tour
                                    </button>
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>

                <div class="mt-8 p-4 bg-slate-100 dark:bg-slate-700 rounded-lg">
                    <h3 class="font-medium text-slate-900 dark:text-slate-100 mb-2">🎯 Tour Tips:</h3>
                    <ul class="space-y-1 text-slate-700 dark:text-slate-300 text-sm">
                        <li>• Tours can be skipped or paused at any time by pressing Escape</li>
                        <li>• You can replay any tour from this page</li>
                        <li>• Tours adapt to your current dashboard state</li>
                        <li>• Complete all tours to unlock the "Tour Master" achievement</li>
                    </ul>
                </div>
            </div>
        `;

        // Bind tour start buttons
        content.querySelectorAll('.help-start-tour').forEach(btn => {
            btn.addEventListener('click', () => {
                const tourId = btn.dataset.tour;
                this.hideHelpOverlay();
                setTimeout(() => this.startTour(tourId), 300);
            });
        });
    }

    performHelpSearch(query) {
        if (!query.trim()) {
            this.showHelpHome();
            return;
        }

        const results = this.searchHelpContent(query);
        this.displayHelpSearchResults(query, results);
        
        // Track search
        this.userProgress.helpSearchHistory.push({
            query,
            results: results.length,
            timestamp: Date.now()
        });
        
        // Keep only last 50 searches
        if (this.userProgress.helpSearchHistory.length > 50) {
            this.userProgress.helpSearchHistory = this.userProgress.helpSearchHistory.slice(-25);
        }
        
        this.saveUserProgress();
    }

    searchHelpContent(query) {
        const queryLower = query.toLowerCase();
        const results = [];

        this.searchIndex.forEach(item => {
            const searchText = item.searchText.toLowerCase();
            
            // Exact phrase match (highest score)
            if (searchText.includes(queryLower)) {
                const score = queryLower.length / searchText.length;
                results.push({ ...item, score: score + 0.5, matchType: 'exact' });
            }
            
            // Keyword matches
            const queryWords = queryLower.split(/\s+/);
            const matchCount = queryWords.reduce((count, word) => {
                return count + (searchText.includes(word) ? 1 : 0);
            }, 0);
            
            if (matchCount > 0) {
                const score = matchCount / queryWords.length;
                if (score >= 0.5) { // At least half the words match
                    const existing = results.find(r => r.id === item.id);
                    if (!existing || existing.score < score) {
                        if (existing) {
                            existing.score = score;
                            existing.matchType = 'keywords';
                        } else {
                            results.push({ ...item, score, matchType: 'keywords' });
                        }
                    }
                }
            }
        });

        // Sort by score (descending) and limit results
        return results
            .sort((a, b) => b.score - a.score)
            .slice(0, this.settings.helpSearchResultsLimit);
    }

    displayHelpSearchResults(query, results) {
        const content = document.getElementById('helpContent');
        
        content.innerHTML = `
            <div class="max-w-none">
                <h1 class="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-2">
                    Search Results
                </h1>
                <p class="text-slate-600 dark:text-slate-400 mb-6">
                    Found ${results.length} result${results.length !== 1 ? 's' : ''} for "${query}"
                </p>

                ${results.length > 0 ? `
                    <div class="space-y-4">
                        ${results.map(result => `
                            <div class="help-search-result border border-slate-200 dark:border-slate-700 rounded-lg p-4 hover:bg-slate-50 dark:hover:bg-slate-700 cursor-pointer transition-colors" data-section="${result.parentId || result.id}">
                                <div class="flex items-start justify-between">
                                    <div class="flex-1">
                                        <h3 class="font-semibold text-slate-900 dark:text-slate-100 mb-1">
                                            ${this.highlightSearchTerm(result.title, query)}
                                        </h3>
                                        <p class="text-slate-600 dark:text-slate-400 text-sm mb-2">
                                            ${this.highlightSearchTerm(result.content, query)}
                                        </p>
                                        <div class="flex items-center space-x-2 text-xs">
                                            <span class="px-2 py-1 bg-slate-100 dark:bg-slate-600 rounded-full text-slate-600 dark:text-slate-300">
                                                ${result.type}
                                            </span>
                                            <span class="px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-300 rounded-full">
                                                ${Math.round(result.score * 100)}% match
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                ` : `
                    <div class="text-center py-12">
                        <div class="text-6xl mb-4">🔍</div>
                        <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-2">
                            No results found
                        </h3>
                        <p class="text-slate-600 dark:text-slate-400 mb-6">
                            Try searching with different terms or browse our help topics.
                        </p>
                        <button id="browseHelpTopics" class="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors">
                            Browse Help Topics
                        </button>
                    </div>
                `}
            </div>
        `;

        // Bind result click handlers
        content.querySelectorAll('.help-search-result').forEach(result => {
            result.addEventListener('click', () => {
                const section = result.dataset.section;
                this.showHelpSection(section);
            });
        });

        // Bind browse button
        content.querySelector('#browseHelpTopics')?.addEventListener('click', () => {
            this.showHelpHome();
            document.getElementById('helpSearch').value = '';
        });
    }

    highlightSearchTerm(text, query) {
        if (!query.trim()) return text;
        
        const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
        return text.replace(regex, '<mark class="bg-yellow-200 dark:bg-yellow-800 px-1 rounded">$1</mark>');
    }

    /**
     * KEYBOARD SHORTCUTS FOR HELP SYSTEM
     */
    initHelpKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Cmd/Ctrl + / to toggle help
            if ((e.ctrlKey || e.metaKey) && e.key === '/') {
                e.preventDefault();
                this.toggleHelpOverlay();
            }
            
            // Escape to close help
            if (e.key === 'Escape' && this.helpOverlayVisible) {
                e.preventDefault();
                this.hideHelpOverlay();
            }

            // F1 for context-sensitive help
            if (e.key === 'F1') {
                e.preventDefault();
                this.showContextualHelp();
            }
        });
    }

    showContextualHelp() {
        // Determine current context and show relevant help
        const activeElement = document.activeElement;
        const currentView = document.querySelector('.view:not(.hidden)')?.id;

        let helpTopic = 'navigation'; // default

        if (activeElement?.id === 'globalSearch' || activeElement?.id === 'advancedSearchInput') {
            helpTopic = 'search';
        } else if (activeElement?.id === 'note') {
            helpTopic = 'note-creation';
        } else if (currentView?.includes('analytics')) {
            helpTopic = 'analytics';
        }

        this.showHelpOverlay();
        setTimeout(() => this.showHelpSection(helpTopic), 100);
    }

    /**
     * ACHIEVEMENTS SYSTEM
     */
    initAchievements() {
        // Define achievements
        this.achievements.set('onboarding-complete', {
            title: 'Welcome Aboard!',
            description: 'Completed the welcome tour',
            icon: '🎉',
            category: 'getting-started'
        });

        this.achievements.set('first-search', {
            title: 'Knowledge Seeker',
            description: 'Performed your first search',
            icon: '🔍',
            category: 'search'
        });

        this.achievements.set('shortcut-master', {
            title: 'Shortcut Master',
            description: 'Used 10 different keyboard shortcuts',
            icon: '⌨️',
            category: 'productivity'
        });

        this.achievements.set('tour-master', {
            title: 'Tour Master',
            description: 'Completed all available tours',
            icon: '🎯',
            category: 'exploration'
        });

        this.achievements.set('help-explorer', {
            title: 'Help Explorer',
            description: 'Accessed 5 different help topics',
            icon: '📚',
            category: 'learning'
        });

        this.achievements.set('feature-discoverer', {
            title: 'Feature Discoverer',
            description: 'Discovered 8 different features',
            icon: '🧭',
            category: 'exploration'
        });
    }

    unlockAchievement(achievementId) {
        if (this.userProgress.achievements.has(achievementId)) return;

        const achievement = this.achievements.get(achievementId);
        if (!achievement) return;

        this.userProgress.achievements.add(achievementId);
        this.saveUserProgress();

        if (this.settings.achievementNotifications) {
            this.showAchievementNotification(achievement);
        }

        console.log(`🏆 Achievement unlocked: ${achievement.title}`);
    }

    showAchievementNotification(achievement) {
        const notification = document.createElement('div');
        notification.className = 'fixed top-4 right-4 bg-gradient-to-r from-purple-500 to-pink-500 text-white p-4 rounded-lg shadow-lg z-50 max-w-sm animate-slide-in';
        
        notification.innerHTML = `
            <div class="flex items-center">
                <span class="text-3xl mr-3">${achievement.icon}</span>
                <div>
                    <div class="font-semibold">Achievement Unlocked!</div>
                    <div class="text-sm">${achievement.title}</div>
                    <div class="text-xs opacity-90 mt-1">${achievement.description}</div>
                </div>
                <button class="ml-2 opacity-70 hover:opacity-100" onclick="this.parentElement.parentElement.remove()">
                    <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                    </svg>
                </button>
            </div>
        `;

        document.body.appendChild(notification);

        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 8000);
    }

    /**
     * SMART SUGGESTIONS AND FEATURE DISCOVERY
     */
    initFeatureDiscovery() {
        // Monitor feature usage
        this.trackFeatureUsage = this.trackFeatureUsage.bind(this);
        
        // Setup feature usage tracking
        this.setupFeatureTracking();
    }

    setupFeatureTracking() {
        // Track search usage
        document.addEventListener('search-performed', () => {
            this.trackFeatureUsage('search');
            if (!this.userProgress.achievements.has('first-search')) {
                this.unlockAchievement('first-search');
            }
        });

        // Track keyboard shortcuts
        if (window.dashboardPerformance) {
            const originalHandler = window.dashboardPerformance.handleKeyboardShortcut;
            if (originalHandler) {
                window.dashboardPerformance.handleKeyboardShortcut = function(e) {
                    const result = originalHandler.call(this, e);
                    if (result !== false) { // Shortcut was handled
                        window.dashboardHelp?.trackFeatureUsage('keyboard-shortcut');
                    }
                    return result;
                };
            }
        }

        // Track note creation
        document.addEventListener('note-created', () => {
            this.trackFeatureUsage('note-creation');
        });

        // Track view changes
        document.addEventListener('view-changed', (e) => {
            this.trackFeatureUsage(`view-${e.detail.view}`);
        });
    }

    trackFeatureUsage(feature) {
        this.userProgress.featuresUsed.add(feature);
        this.saveUserProgress();

        // Check for achievements
        if (this.userProgress.featuresUsed.size >= 8 && !this.userProgress.achievements.has('feature-discoverer')) {
            this.unlockAchievement('feature-discoverer');
        }

        // Check for shortcut mastery
        const shortcutFeatures = Array.from(this.userProgress.featuresUsed).filter(f => f === 'keyboard-shortcut');
        if (shortcutFeatures.length >= 10 && !this.userProgress.achievements.has('shortcut-master')) {
            this.unlockAchievement('shortcut-master');
        }
    }

    showSmartSuggestion(suggestion) {
        if (!this.settings.enableFeatureSuggestions) return;
        if (this.featureDiscovery.dismissedSuggestions.has(suggestion.title)) return;

        // Ensure only one smart suggestion is visible at a time (singleton behavior)
        try {
            document.querySelectorAll('.sb-smart-suggestion').forEach(el => el.remove());
        } catch (e) { /* ignore */ }

        const suggestionEl = document.createElement('div');
        suggestionEl.className = 'sb-smart-suggestion fixed bottom-20 left-4 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg shadow-lg p-4 max-w-sm z-40 animate-slide-up';
        if (suggestion.type) suggestionEl.dataset.suggestionType = suggestion.type;
        
        suggestionEl.innerHTML = `
            <div class="flex items-start justify-between mb-2">
                <h4 class="font-medium text-slate-900 dark:text-slate-100">${suggestion.title}</h4>
                <button class="text-slate-400 hover:text-slate-600 ml-2" onclick="this.closest('.fixed').remove()">
                    <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                    </svg>
                </button>
            </div>
            <p class="text-sm text-slate-600 dark:text-slate-400 mb-3">${suggestion.content}</p>
            <div class="flex space-x-2">
                ${suggestion.actions?.map(action => `
                    <button class="help-suggestion-action px-3 py-1 bg-blue-500 text-white text-sm rounded hover:bg-blue-600 transition-colors" data-action-text="${action.text}">
                        ${action.text}
                    </button>
                `).join('') || ''}
                <button class="help-dismiss-suggestion px-3 py-1 text-slate-500 hover:text-slate-700 text-sm transition-colors">
                    Dismiss
                </button>
            </div>
        `;

        document.body.appendChild(suggestionEl);

        // Bind action buttons
        suggestionEl.querySelectorAll('.help-suggestion-action').forEach((btn, index) => {
            btn.addEventListener('click', () => {
                if (suggestion.actions && suggestion.actions[index]) {
                    suggestion.actions[index].action();
                }
                suggestionEl.remove();
            });
        });

        // Bind dismiss button
        suggestionEl.querySelector('.help-dismiss-suggestion').addEventListener('click', () => {
            this.featureDiscovery.dismissedSuggestions.add(suggestion.title);
            suggestionEl.remove();
        });

        // Auto-remove after timeout
        if (suggestion.timeout) {
            setTimeout(() => {
                if (suggestionEl.parentNode) {
                    suggestionEl.remove();
                }
            }, suggestion.timeout);
        }
    }

    showSmartSuggestions() {
        // Show contextual suggestions based on user progress
        const suggestions = this.generateSmartSuggestions();
        
        if (suggestions.length > 0) {
            // Show the most relevant suggestion
            this.showSmartSuggestion(suggestions[0]);
        }
    }

    generateSmartSuggestions() {
        const suggestions = [];
        
        // Suggest keyboard shortcuts if user hasn't used many
        const shortcutUsage = Array.from(this.userProgress.featuresUsed).filter(f => f === 'keyboard-shortcut').length;
        if (shortcutUsage < 3) {
            suggestions.push({
                title: 'Speed up with keyboard shortcuts',
                content: 'Learn essential keyboard shortcuts to work faster. Try ⌘/Ctrl+K for global search!',
                actions: [
                    {
                        text: 'Show Shortcuts',
                        action: () => this.showKeyboardShortcuts()
                    }
                ],
                priority: 8
            });
        }

        // Suggest advanced search if user only uses basic search
        if (this.userProgress.featuresUsed.has('search') && !this.userProgress.featuresUsed.has('advanced-search')) {
            suggestions.push({
                title: 'Try advanced search features',
                content: 'Use filters, date ranges, and boolean operators for more precise results.',
                actions: [
                    {
                        text: 'Learn More',
                        action: () => this.showHelp('search')
                    }
                ],
                priority: 6
            });
        }

        // Suggest integrations if user hasn't explored them
        if (!this.userProgress.featuresUsed.has('integrations')) {
            suggestions.push({
                title: 'Connect with your favorite tools',
                content: 'Set up Obsidian sync, Discord integration, and Apple Shortcuts for seamless workflow.',
                actions: [
                    {
                        text: 'Setup Integrations',
                        action: () => this.showHelp('integrations')
                    }
                ],
                priority: 4
            });
        }

        // Sort by priority (higher first)
        return suggestions.sort((a, b) => (b.priority || 0) - (a.priority || 0));
    }

    /**
     * UTILITY METHODS
     */
    handleHelpAction(action) {
        switch (action) {
            case 'feedback':
                this.showFeedbackForm();
                break;
            case 'reset-progress':
                this.resetProgress();
                break;
            case 'export-data':
                this.exportHelpData();
                break;
        }
    }

    showFeedbackForm() {
        // Create feedback modal
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        modal.innerHTML = `
            <div class="bg-white dark:bg-slate-800 rounded-lg p-6 max-w-md w-full mx-4">
                <h3 class="text-lg font-semibold mb-4 text-slate-900 dark:text-slate-100">Send Feedback</h3>
                <textarea id="feedbackText" placeholder="Tell us about your experience with the help system..." 
                          class="w-full h-32 p-3 border border-slate-300 dark:border-slate-600 rounded-lg resize-none bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100" maxlength="500"></textarea>
                <div class="flex justify-between items-center mt-4">
                    <span class="text-sm text-slate-500"><span id="charCount">0</span>/500</span>
                    <div class="space-x-2">
                        <button id="cancelFeedback" class="px-4 py-2 text-slate-600 hover:text-slate-800 transition-colors">Cancel</button>
                        <button id="sendFeedback" class="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors">Send</button>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        const textarea = modal.querySelector('#feedbackText');
        const charCount = modal.querySelector('#charCount');
        const sendBtn = modal.querySelector('#sendFeedback');
        const cancelBtn = modal.querySelector('#cancelFeedback');

        textarea.addEventListener('input', () => {
            charCount.textContent = textarea.value.length;
        });

        sendBtn.addEventListener('click', () => {
            const feedback = textarea.value.trim();
            if (feedback) {
                // Store feedback locally (in real app, would send to server)
                const feedbacks = JSON.parse(localStorage.getItem('dashboard_feedback') || '[]');
                feedbacks.push({
                    feedback,
                    timestamp: Date.now(),
                    context: 'help-system'
                });
                localStorage.setItem('dashboard_feedback', JSON.stringify(feedbacks));
                
                this.showNotification('Thank you for your feedback!', 'success');
            }
            modal.remove();
        });

        cancelBtn.addEventListener('click', () => modal.remove());
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.remove();
        });
    }

    resetProgress() {
        if (confirm('Are you sure you want to reset all help progress? This will clear your achievements and tour completion status.')) {
            this.userProgress = {
                onboardingCompleted: false,
                featuresUsed: new Set(),
                tourCompleted: new Set(),
                achievements: new Set(),
                helpSearchHistory: [],
                lastHelpAccessed: null
            };
            this.saveUserProgress();
            this.showNotification('Help progress reset successfully', 'success');
            this.showHelpHome();
        }
    }

    exportHelpData() {
        const data = {
            progress: {
                ...this.userProgress,
                featuresUsed: Array.from(this.userProgress.featuresUsed),
                tourCompleted: Array.from(this.userProgress.tourCompleted),
                achievements: Array.from(this.userProgress.achievements)
            },
            settings: this.settings,
            timestamp: Date.now()
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `dashboard-help-data-${new Date().toISOString().split('T')[0]}.json`;
        link.click();
        URL.revokeObjectURL(url);

        this.showNotification('Help data exported successfully', 'success');
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `
            fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 max-w-sm animate-slide-in
            ${type === 'success' ? 'bg-green-500 text-white' :
              type === 'error' ? 'bg-red-500 text-white' :
              type === 'warning' ? 'bg-yellow-500 text-black' :
              'bg-blue-500 text-white'}
        `;
        
        notification.innerHTML = `
            <div class="flex items-center justify-between">
                <span>${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" class="ml-2 opacity-70 hover:opacity-100">
                    <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                    </svg>
                </button>
            </div>
        `;

        document.body.appendChild(notification);

        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 4000);
    }

    // Public API methods
    showHelp(topic) {
        this.showHelpOverlay();
        if (topic) {
            setTimeout(() => this.showHelpSection(topic), 100);
        }
    }

    showKeyboardShortcuts() {
        this.showHelpOverlay();
        setTimeout(() => this.showHelpSection('shortcuts'), 100);
    }

    focusAdvancedSearch() {
        const advancedSearch = document.getElementById('advancedSearchInput');
        if (advancedSearch) {
            advancedSearch.focus();
        }
    }

    searchRecent() {
        const searchInput = document.getElementById('globalSearch');
        if (searchInput) {
            searchInput.focus();
            // Could populate with recent terms
        }
    }

    createDailyReflection() {
        const noteInput = document.getElementById('note');
        if (noteInput) {
            const template = `# Daily Reflection - ${new Date().toLocaleDateString()}

## What went well today?


## What could be improved?


## Key insights or learnings:


## Tomorrow's priorities:
- [ ] 
- [ ] 
- [ ] `;

            noteInput.value = template;
            noteInput.focus();
        }
    }

    cleanup() {
        console.log('🧹 Cleaning up help system');
        
        // Remove UI elements
        const helpButton = document.getElementById('helpButton');
        const helpOverlay = document.getElementById('helpOverlay');
        
        if (helpButton) helpButton.remove();
        if (helpOverlay) helpOverlay.remove();
        
        // Clean up tours
        this.removeTourElements();
        
        console.log('✅ Help system cleanup completed');
    }
}

// Initialize help system when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.dashboardHelp = new DashboardHelp();
    });
} else {
    window.dashboardHelp = new DashboardHelp();
}

// Export for debugging and manual access
window.DashboardHelp = DashboardHelp;
