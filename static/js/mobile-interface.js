/**
 * Mobile Interface Revolution for Dashboard v3
 * 
 * Transform Second Brain into a world-class mobile-first knowledge management platform
 * 
 * Features:
 * - Touch-first navigation with swipe gestures
 * - Mobile-optimized layout with responsive breakpoints
 * - Native mobile interactions (iOS/Android styles)
 * - Floating Action Button (FAB) system
 * - Bottom navigation bar
 * - Pull-to-refresh functionality
 * - Haptic feedback simulation
 * - Performance optimizations for mobile
 */

class MobileInterface {
    constructor(contentManager = null, performanceSystem = null) {
        this.contentManager = contentManager;
        this.performanceSystem = performanceSystem;
        
        // Mobile detection and capabilities
        this.isMobile = this.detectMobile();
        this.isTablet = this.detectTablet();
        this.isAndroid = /Android/i.test(navigator.userAgent);
        this.isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
        this.hasTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
        
        // Touch gesture state
        this.gestureState = {
            touchStart: null,
            touchMove: null,
            touchEnd: null,
            isGesturing: false,
            gestureType: null,
            targetElement: null,
            startTime: 0,
            threshold: {
                swipe: 50,        // Minimum distance for swipe
                velocity: 0.3,    // Minimum velocity for gesture
                tap: 10,          // Maximum distance for tap
                longPress: 500    // Time for long press
            }
        };
        
        // Swipe actions configuration
        this.swipeActions = {
            left: { action: 'delete', color: '#ef4444', icon: '🗑️', text: 'Delete' },
            right: { action: 'archive', color: '#f59e0b', icon: '📦', text: 'Archive' },
            up: { action: 'favorite', color: '#10b981', icon: '⭐', text: 'Favorite' },
            down: { action: 'share', color: '#3b82f6', icon: '📤', text: 'Share' }
        };
        
        // Mobile layout state
        this.layoutState = {
            bottomNavVisible: true,
            fabVisible: true,
            currentScreen: 'dashboard',
            isKeyboardVisible: false,
            safeAreaInsets: this.getSafeAreaInsets(),
            orientation: this.getOrientation(),
            density: this.getContentDensity()
        };
        
        // Pull-to-refresh state
        this.pullToRefresh = {
            enabled: true,
            threshold: 60,
            isRefreshing: false,
            pullDistance: 0,
            indicator: null
        };
        
        // Virtual keyboard handling
        this.keyboard = {
            isVisible: false,
            height: 0,
            initialViewportHeight: window.innerHeight
        };
        
        this.init();
    }
    
    /**
     * INITIALIZATION
     */
    async init() {
        console.log('📱 Initializing Mobile Interface Revolution');
        
        if (!this.isMobile && !this.isTablet && !this.hasTouch) {
            console.log('📱 Desktop detected - enabling mobile preview mode');
        }
        
        try {
            // Initialize all mobile systems
            this.initializeMobileDetection();
            this.initializeTouchGestures();
            this.initializeMobileLayout();
            this.initializeBottomNavigation();
            this.initializeFloatingActionButton();
            this.initializePullToRefresh();
            this.initializeVirtualKeyboard();
            this.initializeHapticFeedback();
            this.initializePerformanceOptimizations();
            this.initializeMobileAccessibility();
            
            // Setup responsive breakpoints
            this.setupResponsiveBreakpoints();
            
            // Initialize platform-specific features
            if (this.isIOS) {
                this.initializeIOSFeatures();
            } else if (this.isAndroid) {
                this.initializeAndroidFeatures();
            }
            
            // Apply mobile styles
            this.applyMobileStyles();
            
            console.log('✅ Mobile Interface Revolution initialized');
            
            // Dispatch ready event for content manager integration
            document.dispatchEvent(new CustomEvent('mobileInterfaceReady'));
        } catch (error) {
            console.error('❌ Mobile Interface initialization failed:', error);
            if (this.performanceSystem) {
                this.performanceSystem.handleError('mobile-interface-init', error);
            }
        }
    }
    
    /**
     * MOBILE DETECTION & CAPABILITIES
     */
    detectMobile() {
        const userAgent = navigator.userAgent.toLowerCase();
        const mobileKeywords = ['android', 'iphone', 'ipod', 'ipad', 'windows phone', 'blackberry'];
        return mobileKeywords.some(keyword => userAgent.includes(keyword)) || 
               window.innerWidth <= 768;
    }
    
    detectTablet() {
        const userAgent = navigator.userAgent.toLowerCase();
        return (userAgent.includes('ipad') || 
                (userAgent.includes('android') && !userAgent.includes('mobile')) ||
                (window.innerWidth >= 768 && window.innerWidth <= 1024));
    }
    
    getSafeAreaInsets() {
        // Get safe area insets for modern devices (iPhone X+, Android with notches)
        const style = getComputedStyle(document.documentElement);
        return {
            top: parseInt(style.getPropertyValue('--sat') || style.getPropertyValue('env(safe-area-inset-top)') || '0'),
            bottom: parseInt(style.getPropertyValue('--sab') || style.getPropertyValue('env(safe-area-inset-bottom)') || '0'),
            left: parseInt(style.getPropertyValue('--sal') || style.getPropertyValue('env(safe-area-inset-left)') || '0'),
            right: parseInt(style.getPropertyValue('--sar') || style.getPropertyValue('env(safe-area-inset-right)') || '0')
        };
    }
    
    getOrientation() {
        return window.innerWidth > window.innerHeight ? 'landscape' : 'portrait';
    }
    
    getContentDensity() {
        // Determine optimal content density based on screen size
        const width = window.innerWidth;
        if (width < 375) return 'compact';      // Small phones
        if (width < 414) return 'comfortable';  // Standard phones
        if (width < 768) return 'spacious';     // Large phones
        return 'desktop';                       // Tablets and up
    }
    
    /**
     * TOUCH GESTURES SYSTEM
     */
    initializeTouchGestures() {
        if (!this.hasTouch) {
            // Enable mouse simulation for desktop testing
            this.enableMouseSimulation();
        }
        
        // Add touch event listeners
        document.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: false });
        document.addEventListener('touchmove', this.handleTouchMove.bind(this), { passive: false });
        document.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: false });
        document.addEventListener('touchcancel', this.handleTouchCancel.bind(this), { passive: false });
        
        // Add gesture recognition for specific elements
        this.initializeSwipeGestures();
        this.initializeLongPressGestures();
        this.initializePinchGestures();
    }
    
    handleTouchStart(event) {
        const touch = event.touches[0];
        this.gestureState.touchStart = {
            x: touch.clientX,
            y: touch.clientY,
            time: Date.now()
        };
        this.gestureState.targetElement = event.target.closest('[data-swipeable], [data-note-id]');
        this.gestureState.startTime = Date.now();
        this.gestureState.isGesturing = false;
        
        // Cancel any existing swipe animations
        this.cancelSwipeAnimation();
        
        // Start long press timer
        this.startLongPressTimer();
    }
    
    handleTouchMove(event) {
        if (!this.gestureState.touchStart) return;
        
        const touch = event.touches[0];
        this.gestureState.touchMove = {
            x: touch.clientX,
            y: touch.clientY,
            time: Date.now()
        };
        
        // Calculate gesture metrics
        const deltaX = this.gestureState.touchMove.x - this.gestureState.touchStart.x;
        const deltaY = this.gestureState.touchMove.y - this.gestureState.touchStart.y;
        const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
        
        // Cancel long press if moving
        if (distance > this.gestureState.threshold.tap) {
            this.cancelLongPress();
        }
        
        // Handle swipe gestures
        if (this.gestureState.targetElement && this.isSwipeGesture(deltaX, deltaY, distance)) {
            this.gestureState.isGesturing = true;
            this.handleSwipePreview(deltaX, deltaY);
            
            // Prevent default scrolling during swipe
            if (Math.abs(deltaX) > Math.abs(deltaY)) {
                event.preventDefault();
            }
        }
    }
    
    handleTouchEnd(event) {
        if (!this.gestureState.touchStart) return;
        
        const endTime = Date.now();
        const touch = event.changedTouches[0];
        this.gestureState.touchEnd = {
            x: touch.clientX,
            y: touch.clientY,
            time: endTime
        };
        
        // Calculate final gesture metrics
        const deltaX = this.gestureState.touchEnd.x - this.gestureState.touchStart.x;
        const deltaY = this.gestureState.touchEnd.y - this.gestureState.touchStart.y;
        const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
        const duration = endTime - this.gestureState.startTime;
        const velocity = distance / duration;
        
        // Cancel long press
        this.cancelLongPress();
        
        // Handle completed gestures
        if (this.gestureState.isGesturing && this.gestureState.targetElement) {
            this.handleSwipeComplete(deltaX, deltaY, velocity);
        } else if (distance <= this.gestureState.threshold.tap) {
            this.handleTap(this.gestureState.targetElement);
        }
        
        // Reset gesture state
        this.resetGestureState();
    }
    
    handleTouchCancel(event) {
        this.cancelLongPress();
        this.cancelSwipeAnimation();
        this.resetGestureState();
    }
    
    isSwipeGesture(deltaX, deltaY, distance) {
        return distance > this.gestureState.threshold.swipe && 
               (Math.abs(deltaX) > Math.abs(deltaY) * 0.7 || Math.abs(deltaY) > Math.abs(deltaX) * 0.7);
    }
    
    handleSwipePreview(deltaX, deltaY) {
        const element = this.gestureState.targetElement;
        if (!element) return;
        
        // Determine swipe direction
        let direction = null;
        if (Math.abs(deltaX) > Math.abs(deltaY)) {
            direction = deltaX > 0 ? 'right' : 'left';
        } else {
            direction = deltaY > 0 ? 'down' : 'up';
        }
        
        // Apply preview transformation
        const progress = Math.min(Math.abs(deltaX > deltaY ? deltaX : deltaY) / 100, 1);
        this.showSwipePreview(element, direction, progress);
    }
    
    showSwipePreview(element, direction, progress) {
        const action = this.swipeActions[direction];
        if (!action) return;
        
        // Apply visual feedback
        const translateX = direction === 'left' ? -progress * 80 : 
                          direction === 'right' ? progress * 80 : 0;
        const translateY = direction === 'up' ? -progress * 80 : 
                          direction === 'down' ? progress * 80 : 0;
        
        element.style.transform = `translate3d(${translateX}px, ${translateY}px, 0) scale(${1 - progress * 0.1})`;
        element.style.opacity = 1 - progress * 0.3;
        
        // Show action indicator
        this.showSwipeActionIndicator(element, direction, action, progress);
        
        // Haptic feedback
        if (progress > 0.7) {
            this.triggerHapticFeedback('medium');
        }
    }
    
    showSwipeActionIndicator(element, direction, action, progress) {
        let indicator = element.querySelector('.swipe-indicator');
        
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.className = 'swipe-indicator fixed inset-0 flex items-center justify-center pointer-events-none z-50';
            element.appendChild(indicator);
        }
        
        const opacity = Math.min(progress, 1);
        indicator.style.backgroundColor = `${action.color}${Math.round(opacity * 0.3 * 255).toString(16).padStart(2, '0')}`;
        indicator.innerHTML = `
            <div class="flex flex-col items-center text-white">
                <div class="text-3xl mb-2" style="transform: scale(${0.5 + progress * 0.5})">${action.icon}</div>
                <div class="text-sm font-medium">${action.text}</div>
            </div>
        `;
        
        indicator.style.opacity = opacity;
    }
    
    handleSwipeComplete(deltaX, deltaY, velocity) {
        const element = this.gestureState.targetElement;
        if (!element) return;
        
        // Determine swipe direction and strength
        let direction = null;
        let strength = 0;
        
        if (Math.abs(deltaX) > Math.abs(deltaY)) {
            direction = deltaX > 0 ? 'right' : 'left';
            strength = Math.abs(deltaX);
        } else {
            direction = deltaY > 0 ? 'down' : 'up';
            strength = Math.abs(deltaY);
        }
        
        // Check if swipe meets threshold
        const minSwipe = this.gestureState.threshold.swipe;
        const minVelocity = this.gestureState.threshold.velocity;
        
        if (strength >= minSwipe && velocity >= minVelocity) {
            this.executeSwipeAction(element, direction);
        } else {
            this.cancelSwipeAnimation(element);
        }
    }
    
    async executeSwipeAction(element, direction) {
        const action = this.swipeActions[direction];
        if (!action || !this.contentManager) {
            this.cancelSwipeAnimation(element);
            return;
        }
        
        // Get note ID
        const noteId = element.closest('[data-note-id]')?.dataset.noteId;
        if (!noteId) {
            this.cancelSwipeAnimation(element);
            return;
        }
        
        // Trigger haptic feedback
        this.triggerHapticFeedback('heavy');
        
        // Animate swipe completion
        await this.animateSwipeComplete(element, direction);
        
        // Execute action
        try {
            switch (action.action) {
                case 'delete':
                    await this.contentManager.deleteNote(noteId);
                    break;
                case 'archive':
                    await this.contentManager.archiveNote(noteId);
                    break;
                case 'favorite':
                    await this.contentManager.toggleFavorite(noteId);
                    break;
                case 'share':
                    this.shareNote(noteId);
                    break;
            }
        } catch (error) {
            console.error('Swipe action failed:', error);
            this.cancelSwipeAnimation(element);
        }
    }
    
    async animateSwipeComplete(element, direction) {
        return new Promise(resolve => {
            const finalTransform = direction === 'left' ? 'translateX(-100%)' :
                                 direction === 'right' ? 'translateX(100%)' :
                                 direction === 'up' ? 'translateY(-100%)' :
                                 'translateY(100%)';
            
            element.style.transition = 'transform 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
            element.style.transform = finalTransform;
            
            setTimeout(() => {
                element.style.opacity = '0';
                setTimeout(resolve, 100);
            }, 200);
        });
    }
    
    cancelSwipeAnimation(element = null) {
        const target = element || this.gestureState.targetElement;
        if (!target) return;
        
        // Remove any swipe indicators
        const indicator = target.querySelector('.swipe-indicator');
        if (indicator) indicator.remove();
        
        // Reset transforms
        target.style.transform = '';
        target.style.opacity = '';
        target.style.transition = 'transform 0.2s ease-out';
        
        setTimeout(() => {
            target.style.transition = '';
        }, 200);
    }
    
    /**
     * LONG PRESS GESTURES
     */
    initializeLongPressGestures() {
        this.longPressTimer = null;
    }
    
    startLongPressTimer() {
        this.cancelLongPress();
        this.longPressTimer = setTimeout(() => {
            if (this.gestureState.targetElement && !this.gestureState.isGesturing) {
                this.handleLongPress(this.gestureState.targetElement);
            }
        }, this.gestureState.threshold.longPress);
    }
    
    cancelLongPress() {
        if (this.longPressTimer) {
            clearTimeout(this.longPressTimer);
            this.longPressTimer = null;
        }
    }
    
    handleLongPress(element) {
        // Trigger haptic feedback
        this.triggerHapticFeedback('heavy');
        
        // Show context menu
        const noteId = element.closest('[data-note-id]')?.dataset.noteId;
        if (noteId) {
            this.showMobileContextMenu(element, noteId);
        }
    }
    
    showMobileContextMenu(element, noteId) {
        // Remove existing context menus
        document.querySelectorAll('.mobile-context-menu').forEach(menu => menu.remove());
        
        const menu = document.createElement('div');
        menu.className = 'mobile-context-menu fixed inset-0 bg-black bg-opacity-50 z-50 flex items-end justify-center';
        
        const actions = [
            { id: 'edit', icon: '✏️', text: 'Edit', action: () => this.contentManager.openNoteEditor(noteId) },
            { id: 'duplicate', icon: '📋', text: 'Duplicate', action: () => this.contentManager.duplicateNote(noteId) },
            { id: 'tag', icon: '🏷️', text: 'Add Tags', action: () => this.contentManager.showTagDialog(noteId) },
            { id: 'share', icon: '📤', text: 'Share', action: () => this.shareNote(noteId) },
            { id: 'favorite', icon: '⭐', text: 'Favorite', action: () => this.contentManager.toggleFavorite(noteId) },
            { id: 'archive', icon: '📦', text: 'Archive', action: () => this.contentManager.archiveNote(noteId) },
            { id: 'delete', icon: '🗑️', text: 'Delete', action: () => this.contentManager.deleteNote(noteId), danger: true }
        ];
        
        menu.innerHTML = `
            <div class="bg-white rounded-t-xl w-full max-w-md mx-4 pb-${this.layoutState.safeAreaInsets.bottom + 4} animate-slide-up">
                <div class="flex justify-center py-2">
                    <div class="w-12 h-1 bg-slate-300 rounded-full"></div>
                </div>
                <div class="px-4 py-2">
                    <h3 class="font-semibold text-slate-800 mb-4">Note Actions</h3>
                    <div class="space-y-2">
                        ${actions.map(action => `
                            <button class="context-menu-action w-full flex items-center space-x-3 p-3 rounded-lg hover:bg-slate-100 text-left transition-colors ${action.danger ? 'text-red-600' : 'text-slate-700'}"
                                    data-action="${action.id}">
                                <span class="text-xl">${action.icon}</span>
                                <span class="font-medium">${action.text}</span>
                            </button>
                        `).join('')}
                        <button class="w-full p-3 mt-4 bg-slate-100 text-slate-700 rounded-lg font-medium"
                                onclick="this.closest('.mobile-context-menu').remove()">
                            Cancel
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(menu);
        
        // Handle action clicks
        menu.querySelectorAll('.context-menu-action').forEach((button, index) => {
            button.addEventListener('click', () => {
                actions[index].action();
                menu.remove();
                this.triggerHapticFeedback('light');
            });
        });
        
        // Close on backdrop click
        menu.addEventListener('click', (e) => {
            if (e.target === menu) {
                menu.remove();
            }
        });
    }
    
    /**
     * PINCH GESTURES (ZOOM)
     */
    initializePinchGestures() {
        let initialDistance = 0;
        let lastScale = 1;
        
        document.addEventListener('touchstart', (e) => {
            if (e.touches.length === 2) {
                initialDistance = this.getDistance(e.touches[0], e.touches[1]);
            }
        });
        
        document.addEventListener('touchmove', (e) => {
            if (e.touches.length === 2) {
                e.preventDefault();
                
                const currentDistance = this.getDistance(e.touches[0], e.touches[1]);
                const scale = currentDistance / initialDistance;
                
                // Apply zoom to content
                if (Math.abs(scale - lastScale) > 0.1) {
                    this.handlePinchZoom(scale);
                    lastScale = scale;
                }
            }
        });
        
        document.addEventListener('touchend', () => {
            initialDistance = 0;
            lastScale = 1;
        });
    }
    
    getDistance(touch1, touch2) {
        const dx = touch1.clientX - touch2.clientX;
        const dy = touch1.clientY - touch2.clientY;
        return Math.sqrt(dx * dx + dy * dy);
    }
    
    handlePinchZoom(scale) {
        // Adjust content density based on pinch
        let newDensity = this.layoutState.density;
        
        if (scale > 1.2) {
            // Zoom in - make content more spacious
            newDensity = this.layoutState.density === 'compact' ? 'comfortable' :
                        this.layoutState.density === 'comfortable' ? 'spacious' : 'spacious';
        } else if (scale < 0.8) {
            // Zoom out - make content more compact
            newDensity = this.layoutState.density === 'spacious' ? 'comfortable' :
                        this.layoutState.density === 'comfortable' ? 'compact' : 'compact';
        }
        
        if (newDensity !== this.layoutState.density) {
            this.layoutState.density = newDensity;
            this.applyContentDensity(newDensity);
            this.triggerHapticFeedback('light');
        }
    }
    
    applyContentDensity(density) {
        const body = document.body;
        body.classList.remove('density-compact', 'density-comfortable', 'density-spacious');
        body.classList.add(`density-${density}`);
        
        // Save preference
        localStorage.setItem('contentDensity', density);
    }
    
    /**
     * MOBILE LAYOUT SYSTEM
     */
    initializeMobileLayout() {
        this.setupResponsiveBreakpoints();
        this.initializeSafeAreas();
        this.handleOrientationChange();
        
        // Listen for orientation changes
        window.addEventListener('orientationchange', () => {
            setTimeout(() => {
                this.handleOrientationChange();
            }, 100);
        });
        
        // Listen for resize events
        window.addEventListener('resize', this.debounce(() => {
            this.handleResize();
        }, 150));
    }
    
    setupResponsiveBreakpoints() {
        const width = window.innerWidth;
        let breakpoint = 'desktop';
        
        if (width < 768) {
            breakpoint = 'mobile';
        } else if (width < 1024) {
            breakpoint = 'tablet';
        } else if (width < 1200) {
            breakpoint = 'laptop';
        }
        
        document.body.classList.remove('bp-mobile', 'bp-tablet', 'bp-laptop', 'bp-desktop');
        document.body.classList.add(`bp-${breakpoint}`);
        
        // Update layout state
        this.layoutState.currentBreakpoint = breakpoint;
    }
    
    initializeSafeAreas() {
        // Add CSS custom properties for safe areas
        const root = document.documentElement;
        const insets = this.layoutState.safeAreaInsets;
        
        root.style.setProperty('--safe-top', `${insets.top}px`);
        root.style.setProperty('--safe-bottom', `${insets.bottom}px`);
        root.style.setProperty('--safe-left', `${insets.left}px`);
        root.style.setProperty('--safe-right', `${insets.right}px`);
    }
    
    handleOrientationChange() {
        const newOrientation = this.getOrientation();
        
        if (newOrientation !== this.layoutState.orientation) {
            this.layoutState.orientation = newOrientation;
            
            // Update body class
            document.body.classList.remove('orientation-portrait', 'orientation-landscape');
            document.body.classList.add(`orientation-${newOrientation}`);
            
            // Adjust layout for orientation
            this.adjustLayoutForOrientation(newOrientation);
            
            // Update safe areas (they might change)
            this.layoutState.safeAreaInsets = this.getSafeAreaInsets();
            this.initializeSafeAreas();
        }
    }
    
    adjustLayoutForOrientation(orientation) {
        if (orientation === 'landscape' && this.isMobile) {
            // Hide bottom navigation in landscape mode on phones
            this.layoutState.bottomNavVisible = false;
            this.updateBottomNavigation();
        } else {
            // Show bottom navigation in portrait mode
            this.layoutState.bottomNavVisible = true;
            this.updateBottomNavigation();
        }
    }
    
    handleResize() {
        this.setupResponsiveBreakpoints();
        this.layoutState.safeAreaInsets = this.getSafeAreaInsets();
        this.initializeSafeAreas();
    }
    
    /**
     * BOTTOM NAVIGATION
     */
    initializeBottomNavigation() {
        if (!this.isMobile && !this.isTablet) return;
        
        this.createBottomNavigation();
        this.updateBottomNavigation();
    }
    
    createBottomNavigation() {
        // Remove existing bottom nav
        const existing = document.getElementById('mobile-bottom-nav');
        if (existing) existing.remove();
        
        const bottomNav = document.createElement('nav');
        bottomNav.id = 'mobile-bottom-nav';
        bottomNav.className = `fixed bottom-0 left-0 right-0 bg-white border-t border-slate-200 z-40 transition-transform duration-300`;
        bottomNav.style.paddingBottom = `${this.layoutState.safeAreaInsets.bottom}px`;
        
        const tabs = [
            { id: 'dashboard', icon: '🏠', label: 'Home', active: true },
            { id: 'search', icon: '🔍', label: 'Search', active: false },
            { id: 'voice', icon: '🎤', label: 'Voice', active: false },
            { id: 'settings', icon: '⚙️', label: 'Settings', active: false }
        ];
        
        bottomNav.innerHTML = `
            <div class="flex items-center justify-around py-2 px-1">
                ${tabs.map(tab => `
                    <button class="bottom-nav-tab flex-1 flex flex-col items-center py-2 px-1 transition-colors ${tab.active ? 'text-discord-500' : 'text-slate-400'}"
                            data-tab="${tab.id}">
                        <span class="text-xl mb-1">${tab.icon}</span>
                        <span class="text-xs font-medium">${tab.label}</span>
                    </button>
                `).join('')}
            </div>
        `;
        
        document.body.appendChild(bottomNav);
        
        // Handle tab clicks
        bottomNav.querySelectorAll('.bottom-nav-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                const tabId = tab.dataset.tab;
                this.switchToTab(tabId);
                this.triggerHapticFeedback('light');
            });
        });
        
        // Add bottom padding to main content
        const mainContent = document.querySelector('main, .main-content, #app');
        if (mainContent) {
            mainContent.style.paddingBottom = `${60 + this.layoutState.safeAreaInsets.bottom}px`;
        }
    }
    
    updateBottomNavigation() {
        const bottomNav = document.getElementById('mobile-bottom-nav');
        if (!bottomNav) return;
        
        if (this.layoutState.bottomNavVisible && (this.isMobile || this.isTablet)) {
            bottomNav.classList.remove('translate-y-full');
        } else {
            bottomNav.classList.add('translate-y-full');
        }
    }
    
    switchToTab(tabId) {
        // Update active tab
        document.querySelectorAll('.bottom-nav-tab').forEach(tab => {
            const isActive = tab.dataset.tab === tabId;
            tab.classList.toggle('text-discord-500', isActive);
            tab.classList.toggle('text-slate-400', !isActive);
        });
        
        this.layoutState.currentScreen = tabId;
        
        // Handle tab-specific actions
        switch (tabId) {
            case 'dashboard':
                if (window.location.pathname !== '/') {
                    window.location.href = '/';
                }
                break;
            case 'search':
                if (this.contentManager) {
                    this.contentManager.showGlobalSearch();
                }
                break;
            case 'voice':
                this.showVoiceRecording();
                break;
            case 'settings':
                this.showMobileSettings();
                break;
        }
    }
    
    /**
     * FLOATING ACTION BUTTON (FAB)
     */
    initializeFloatingActionButton() {
        if (!this.isMobile && !this.isTablet) return;
        
        this.createFloatingActionButton();
        this.updateFloatingActionButton();
    }
    
    createFloatingActionButton() {
        // Remove existing FAB
        const existing = document.getElementById('mobile-fab');
        if (existing) existing.remove();
        
        const fab = document.createElement('button');
        fab.id = 'mobile-fab';
        fab.className = `fixed z-50 w-14 h-14 bg-discord-500 hover:bg-discord-600 text-white rounded-full shadow-lg flex items-center justify-center transition-all duration-300 active:scale-95`;
        fab.style.bottom = `${80 + this.layoutState.safeAreaInsets.bottom}px`;
        fab.style.right = '16px';
        
        fab.innerHTML = `
            <svg class="w-6 h-6 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path>
            </svg>
        `;
        
        document.body.appendChild(fab);
        
        // Handle FAB clicks
        fab.addEventListener('click', () => {
            this.handleFabClick();
            this.triggerHapticFeedback('medium');
        });
        
        // Handle scroll to hide/show FAB
        this.setupFabScrollBehavior();
    }
    
    updateFloatingActionButton() {
        const fab = document.getElementById('mobile-fab');
        if (!fab) return;
        
        if (this.layoutState.fabVisible && (this.isMobile || this.isTablet)) {
            fab.classList.remove('scale-0');
        } else {
            fab.classList.add('scale-0');
        }
    }
    
    handleFabClick() {
        // Show FAB action menu
        this.showFabActionMenu();
    }
    
    showFabActionMenu() {
        // Remove existing menu
        document.querySelectorAll('.fab-menu').forEach(menu => menu.remove());
        
        const menu = document.createElement('div');
        menu.className = 'fab-menu fixed inset-0 bg-black bg-opacity-50 z-50 flex items-end justify-center';
        
        const actions = [
            { id: 'text-note', icon: '📝', text: 'Text Note', action: () => this.createTextNote() },
            { id: 'voice-note', icon: '🎤', text: 'Voice Note', action: () => this.showVoiceRecording() },
            { id: 'photo-note', icon: '📷', text: 'Photo Note', action: () => this.capturePhoto() },
            { id: 'quick-note', icon: '⚡', text: 'Quick Note', action: () => this.showQuickNoteDialog() }
        ];
        
        menu.innerHTML = `
            <div class="bg-white rounded-t-xl w-full max-w-md mx-4 pb-${this.layoutState.safeAreaInsets.bottom + 4} animate-slide-up">
                <div class="flex justify-center py-2">
                    <div class="w-12 h-1 bg-slate-300 rounded-full"></div>
                </div>
                <div class="px-4 py-2">
                    <h3 class="font-semibold text-slate-800 mb-4">Create New</h3>
                    <div class="grid grid-cols-2 gap-3">
                        ${actions.map(action => `
                            <button class="fab-action flex flex-col items-center p-4 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors"
                                    data-action="${action.id}">
                                <span class="text-2xl mb-2">${action.icon}</span>
                                <span class="text-sm font-medium text-slate-700">${action.text}</span>
                            </button>
                        `).join('')}
                    </div>
                    <button class="w-full p-3 mt-4 bg-slate-100 text-slate-700 rounded-lg font-medium"
                            onclick="this.closest('.fab-menu').remove()">
                        Cancel
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(menu);
        
        // Handle action clicks
        menu.querySelectorAll('.fab-action').forEach((button, index) => {
            button.addEventListener('click', () => {
                actions[index].action();
                menu.remove();
                this.triggerHapticFeedback('light');
            });
        });
        
        // Close on backdrop click
        menu.addEventListener('click', (e) => {
            if (e.target === menu) {
                menu.remove();
            }
        });
    }
    
    setupFabScrollBehavior() {
        let lastScrollY = window.scrollY;
        let scrollTimeout;
        
        window.addEventListener('scroll', () => {
            const currentScrollY = window.scrollY;
            const scrollingDown = currentScrollY > lastScrollY;
            
            // Hide FAB when scrolling down, show when scrolling up
            if (scrollingDown && currentScrollY > 100) {
                this.layoutState.fabVisible = false;
            } else {
                this.layoutState.fabVisible = true;
            }
            
            this.updateFloatingActionButton();
            lastScrollY = currentScrollY;
            
            // Show FAB after scroll stops
            clearTimeout(scrollTimeout);
            scrollTimeout = setTimeout(() => {
                this.layoutState.fabVisible = true;
                this.updateFloatingActionButton();
            }, 1000);
        });
    }
    
    /**
     * PULL TO REFRESH
     */
    initializePullToRefresh() {
        if (!this.hasTouch) return;
        
        let startY = 0;
        let isRefreshing = false;
        
        const mainContent = document.querySelector('main, .main-content, #notes-container');
        if (!mainContent) return;
        
        mainContent.addEventListener('touchstart', (e) => {
            if (mainContent.scrollTop === 0) {
                startY = e.touches[0].clientY;
            }
        }, { passive: true });
        
        mainContent.addEventListener('touchmove', (e) => {
            if (isRefreshing || mainContent.scrollTop > 0) return;
            
            const currentY = e.touches[0].clientY;
            const pullDistance = currentY - startY;
            
            if (pullDistance > 0) {
                e.preventDefault();
                this.handlePullToRefresh(pullDistance);
            }
        }, { passive: false });
        
        mainContent.addEventListener('touchend', () => {
            if (this.pullToRefresh.pullDistance >= this.pullToRefresh.threshold) {
                this.triggerRefresh();
            } else {
                this.cancelPullToRefresh();
            }
        }, { passive: true });
    }
    
    handlePullToRefresh(distance) {
        this.pullToRefresh.pullDistance = Math.min(distance, this.pullToRefresh.threshold * 1.5);
        
        if (!this.pullToRefresh.indicator) {
            this.createPullToRefreshIndicator();
        }
        
        this.updatePullToRefreshIndicator();
    }
    
    createPullToRefreshIndicator() {
        const indicator = document.createElement('div');
        indicator.id = 'pull-to-refresh-indicator';
        indicator.className = 'fixed top-0 left-1/2 transform -translate-x-1/2 bg-white rounded-b-lg shadow-lg px-4 py-2 z-50 transition-all duration-300';
        indicator.style.top = `-${this.layoutState.safeAreaInsets.top + 60}px`;
        
        indicator.innerHTML = `
            <div class="flex items-center space-x-2 text-slate-700">
                <div class="refresh-icon transition-transform duration-300">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                    </svg>
                </div>
                <span class="refresh-text text-sm font-medium">Pull to refresh</span>
            </div>
        `;
        
        document.body.appendChild(indicator);
        this.pullToRefresh.indicator = indicator;
    }
    
    updatePullToRefreshIndicator() {
        const indicator = this.pullToRefresh.indicator;
        if (!indicator) return;
        
        const progress = this.pullToRefresh.pullDistance / this.pullToRefresh.threshold;
        const translateY = Math.min(this.pullToRefresh.pullDistance * 0.5, this.pullToRefresh.threshold * 0.5);
        
        indicator.style.transform = `translate(-50%, ${translateY}px)`;
        
        const icon = indicator.querySelector('.refresh-icon');
        const text = indicator.querySelector('.refresh-text');
        
        if (progress >= 1) {
            icon.style.transform = 'rotate(180deg)';
            text.textContent = 'Release to refresh';
            this.triggerHapticFeedback('medium');
        } else {
            icon.style.transform = `rotate(${progress * 180}deg)`;
            text.textContent = 'Pull to refresh';
        }
    }
    
    async triggerRefresh() {
        this.pullToRefresh.isRefreshing = true;
        
        const indicator = this.pullToRefresh.indicator;
        if (indicator) {
            indicator.querySelector('.refresh-text').textContent = 'Refreshing...';
            indicator.querySelector('.refresh-icon').style.animation = 'spin 1s linear infinite';
        }
        
        // Trigger haptic feedback
        this.triggerHapticFeedback('heavy');
        
        try {
            // Refresh data
            if (this.contentManager) {
                await this.contentManager.loadNotes();
                await this.contentManager.loadTags();
            }
            
            // Refresh UI
            document.dispatchEvent(new CustomEvent('notes-updated'));
            
            // Show success
            this.showNotification('Notes refreshed', 'success');
        } catch (error) {
            console.error('Refresh failed:', error);
            this.showNotification('Refresh failed', 'error');
        } finally {
            setTimeout(() => {
                this.cancelPullToRefresh();
            }, 1000);
        }
    }
    
    cancelPullToRefresh() {
        if (this.pullToRefresh.indicator) {
            this.pullToRefresh.indicator.remove();
            this.pullToRefresh.indicator = null;
        }
        
        this.pullToRefresh.pullDistance = 0;
        this.pullToRefresh.isRefreshing = false;
    }
    
    /**
     * VIRTUAL KEYBOARD HANDLING
     */
    initializeVirtualKeyboard() {
        // Detect virtual keyboard visibility
        let initialViewportHeight = window.innerHeight;
        
        window.addEventListener('resize', () => {
            const currentHeight = window.innerHeight;
            const heightDifference = initialViewportHeight - currentHeight;
            
            if (heightDifference > 150) {
                // Keyboard is likely visible
                this.keyboard.isVisible = true;
                this.keyboard.height = heightDifference;
                this.handleKeyboardShow();
            } else {
                // Keyboard is likely hidden
                this.keyboard.isVisible = false;
                this.keyboard.height = 0;
                this.handleKeyboardHide();
            }
        });
        
        // Handle focus events
        document.addEventListener('focusin', (e) => {
            if (e.target.matches('input, textarea, [contenteditable]')) {
                this.handleInputFocus(e.target);
            }
        });
        
        document.addEventListener('focusout', (e) => {
            if (e.target.matches('input, textarea, [contenteditable]')) {
                this.handleInputBlur(e.target);
            }
        });
    }
    
    handleKeyboardShow() {
        document.body.classList.add('keyboard-visible');
        
        // Hide bottom navigation when keyboard is visible
        this.layoutState.bottomNavVisible = false;
        this.updateBottomNavigation();
        
        // Hide FAB when keyboard is visible
        this.layoutState.fabVisible = false;
        this.updateFloatingActionButton();
    }
    
    handleKeyboardHide() {
        document.body.classList.remove('keyboard-visible');
        
        // Show bottom navigation when keyboard is hidden
        this.layoutState.bottomNavVisible = true;
        this.updateBottomNavigation();
        
        // Show FAB when keyboard is hidden
        this.layoutState.fabVisible = true;
        this.updateFloatingActionButton();
    }
    
    handleInputFocus(input) {
        // Scroll input into view if needed
        setTimeout(() => {
            if (this.keyboard.isVisible) {
                const rect = input.getBoundingClientRect();
                const viewportHeight = window.innerHeight;
                const keyboardTop = viewportHeight - this.keyboard.height;
                
                if (rect.bottom > keyboardTop - 20) {
                    input.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }
        }, 300);
    }
    
    handleInputBlur(input) {
        // Optional: Handle input blur if needed
    }
    
    /**
     * HAPTIC FEEDBACK
     */
    initializeHapticFeedback() {
        this.hapticSupported = 'vibrate' in navigator;
    }
    
    triggerHapticFeedback(type = 'light') {
        if (!this.hapticSupported) return;
        
        const patterns = {
            light: [10],
            medium: [30],
            heavy: [50],
            double: [30, 10, 30],
            success: [10, 10, 10]
        };
        
        const pattern = patterns[type] || patterns.light;
        navigator.vibrate(pattern);
    }
    
    /**
     * PERFORMANCE OPTIMIZATIONS
     */
    initializePerformanceOptimizations() {
        // Throttle touch events
        this.throttledTouchMove = this.throttle(this.handleTouchMove.bind(this), 16);
        
        // Debounce resize events
        this.debouncedResize = this.debounce(this.handleResize.bind(this), 150);
        
        // Use passive listeners where possible
        this.enablePassiveListeners();
        
        // Enable GPU acceleration for animations
        this.enableHardwareAcceleration();
    }
    
    enablePassiveListeners() {
        // Add passive listeners for better scrolling performance
        document.addEventListener('wheel', () => {}, { passive: true });
        document.addEventListener('touchstart', () => {}, { passive: true });
        document.addEventListener('touchmove', () => {}, { passive: true });
    }
    
    enableHardwareAcceleration() {
        // Enable GPU acceleration for animated elements
        const animatedElements = document.querySelectorAll('.animate-slide-up, .animate-fade-in, [data-swipeable]');
        animatedElements.forEach(element => {
            element.style.transform = 'translateZ(0)';
            element.style.backfaceVisibility = 'hidden';
        });
    }
    
    /**
     * MOBILE ACCESSIBILITY
     */
    initializeMobileAccessibility() {
        // Increase touch targets for better accessibility
        this.improveTouchTargets();
        
        // Add high contrast mode support
        this.initializeHighContrastMode();
        
        // Add voice control support
        this.initializeVoiceControl();
    }
    
    improveTouchTargets() {
        const smallTargets = document.querySelectorAll('button, a, [role="button"]');
        smallTargets.forEach(target => {
            const rect = target.getBoundingClientRect();
            if (rect.width < 44 || rect.height < 44) {
                target.style.minWidth = '44px';
                target.style.minHeight = '44px';
                target.style.display = 'inline-flex';
                target.style.alignItems = 'center';
                target.style.justifyContent = 'center';
            }
        });
    }
    
    initializeHighContrastMode() {
        // Detect high contrast preference
        if (window.matchMedia('(prefers-contrast: high)').matches) {
            document.body.classList.add('high-contrast');
        }
    }
    
    initializeVoiceControl() {
        // Basic voice control for mobile
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            this.voiceControlEnabled = true;
        }
    }
    
    /**
     * PLATFORM-SPECIFIC FEATURES
     */
    initializeIOSFeatures() {
        document.body.classList.add('ios');
        
        // Add iOS-specific styles and behaviors
        this.addIOSScrollBehavior();
        this.addIOSNavigationStyles();
    }
    
    initializeAndroidFeatures() {
        document.body.classList.add('android');
        
        // Add Android-specific styles and behaviors
        this.addMaterialDesignEffects();
    }
    
    addIOSScrollBehavior() {
        // Add elastic scroll behavior
        document.body.style.webkitOverflowScrolling = 'touch';
    }
    
    addIOSNavigationStyles() {
        // Add iOS-style navigation animations
        const style = document.createElement('style');
        style.textContent = `
            .ios .animate-slide-up {
                animation: ios-slide-up 0.4s cubic-bezier(0.36, 0.66, 0.04, 1);
            }
            
            @keyframes ios-slide-up {
                from {
                    transform: translateY(100%);
                    opacity: 0.8;
                }
                to {
                    transform: translateY(0);
                    opacity: 1;
                }
            }
        `;
        document.head.appendChild(style);
    }
    
    addMaterialDesignEffects() {
        // Add ripple effect to buttons
        document.addEventListener('touchstart', (e) => {
            const button = e.target.closest('button, [role="button"]');
            if (button && document.body.classList.contains('android')) {
                this.createRippleEffect(button, e);
            }
        });
    }
    
    createRippleEffect(element, event) {
        const ripple = document.createElement('span');
        const rect = element.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = event.touches[0].clientX - rect.left - size / 2;
        const y = event.touches[0].clientY - rect.top - size / 2;
        
        ripple.style.cssText = `
            position: absolute;
            width: ${size}px;
            height: ${size}px;
            left: ${x}px;
            top: ${y}px;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            transform: scale(0);
            animation: ripple 0.6s ease-out;
            pointer-events: none;
        `;
        
        element.style.position = 'relative';
        element.style.overflow = 'hidden';
        element.appendChild(ripple);
        
        setTimeout(() => ripple.remove(), 600);
    }
    
    /**
     * MOBILE STYLES APPLICATION
     */
    applyMobileStyles() {
        const style = document.createElement('style');
        style.id = 'mobile-interface-styles';
        style.textContent = `
            /* Mobile Interface Revolution Styles */
            
            /* Responsive Breakpoints */
            .bp-mobile {
                --content-max-width: 100%;
                --sidebar-width: 0;
                --nav-height: 60px;
            }
            
            .bp-tablet {
                --content-max-width: 768px;
                --sidebar-width: 280px;
                --nav-height: 64px;
            }
            
            /* Content Density */
            .density-compact .note-card {
                padding: 0.75rem;
                margin-bottom: 0.5rem;
            }
            
            .density-comfortable .note-card {
                padding: 1rem;
                margin-bottom: 0.75rem;
            }
            
            .density-spacious .note-card {
                padding: 1.5rem;
                margin-bottom: 1rem;
            }
            
            /* Touch-optimized elements */
            @media (pointer: coarse) {
                button, a, [role="button"] {
                    min-height: 44px;
                    min-width: 44px;
                }
                
                input, textarea, select {
                    min-height: 44px;
                    font-size: 16px; /* Prevent zoom on iOS */
                }
            }
            
            /* Swipe animations */
            @keyframes swipe-left {
                from { transform: translateX(0); }
                to { transform: translateX(-100%); }
            }
            
            @keyframes swipe-right {
                from { transform: translateX(0); }
                to { transform: translateX(100%); }
            }
            
            @keyframes slide-up {
                from {
                    transform: translateY(100%);
                    opacity: 0;
                }
                to {
                    transform: translateY(0);
                    opacity: 1;
                }
            }
            
            @keyframes ripple {
                to {
                    transform: scale(2);
                    opacity: 0;
                }
            }
            
            /* Safe area support */
            .safe-top { padding-top: env(safe-area-inset-top); }
            .safe-bottom { padding-bottom: env(safe-area-inset-bottom); }
            .safe-left { padding-left: env(safe-area-inset-left); }
            .safe-right { padding-right: env(safe-area-inset-right); }
            
            /* Keyboard adjustments */
            .keyboard-visible .main-content {
                padding-bottom: 0 !important;
            }
            
            /* Mobile-specific hiding */
            .bp-mobile .desktop-only {
                display: none !important;
            }
            
            .bp-desktop .mobile-only {
                display: none !important;
            }
            
            /* High contrast mode */
            .high-contrast {
                --tw-bg-opacity: 1;
                --tw-text-opacity: 1;
            }
            
            .high-contrast .note-card {
                border-width: 2px;
            }
            
            /* Performance optimizations */
            .gpu-accelerated {
                transform: translateZ(0);
                backface-visibility: hidden;
            }
        `;
        
        document.head.appendChild(style);
    }
    
    /**
     * UTILITY METHODS
     */
    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
    
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    resetGestureState() {
        this.gestureState.touchStart = null;
        this.gestureState.touchMove = null;
        this.gestureState.touchEnd = null;
        this.gestureState.isGesturing = false;
        this.gestureState.gestureType = null;
        this.gestureState.targetElement = null;
        this.gestureState.startTime = 0;
    }
    
    handleTap(element) {
        // Handle single tap events
        if (element && element.closest('[data-note-id]')) {
            this.triggerHapticFeedback('light');
        }
    }
    
    enableMouseSimulation() {
        // Convert mouse events to touch events for desktop testing
        document.addEventListener('mousedown', (e) => {
            const touchEvent = new TouchEvent('touchstart', {
                touches: [new Touch({
                    identifier: 0,
                    target: e.target,
                    clientX: e.clientX,
                    clientY: e.clientY
                })]
            });
            this.handleTouchStart(touchEvent);
        });
        
        document.addEventListener('mousemove', (e) => {
            if (this.gestureState.touchStart) {
                const touchEvent = new TouchEvent('touchmove', {
                    touches: [new Touch({
                        identifier: 0,
                        target: e.target,
                        clientX: e.clientX,
                        clientY: e.clientY
                    })]
                });
                this.handleTouchMove(touchEvent);
            }
        });
        
        document.addEventListener('mouseup', (e) => {
            if (this.gestureState.touchStart) {
                const touchEvent = new TouchEvent('touchend', {
                    changedTouches: [new Touch({
                        identifier: 0,
                        target: e.target,
                        clientX: e.clientX,
                        clientY: e.clientY
                    })]
                });
                this.handleTouchEnd(touchEvent);
            }
        });
    }
    
    /**
     * ACTION METHODS
     */
    createTextNote() {
        if (this.contentManager) {
            this.contentManager.createBlankNote();
        } else {
            window.location.href = '/note/new';
        }
    }
    
    showVoiceRecording() {
        // Trigger voice recording interface
        const voiceButton = document.querySelector('[data-action="voice"], .voice-record-btn');
        if (voiceButton) {
            voiceButton.click();
        }
    }
    
    capturePhoto() {
        // Trigger photo capture
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.capture = 'environment';
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                this.processPhotoCapture(file);
            }
        };
        input.click();
    }
    
    processPhotoCapture(file) {
        // Process captured photo
        console.log('Processing photo capture:', file);
        // TODO: Implement photo processing and note creation
    }
    
    showQuickNoteDialog() {
        // Show quick note input dialog
        const dialog = document.createElement('div');
        dialog.className = 'fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4';
        
        dialog.innerHTML = `
            <div class="bg-white rounded-lg w-full max-w-md p-6">
                <h3 class="text-lg font-semibold mb-4">Quick Note</h3>
                <textarea class="w-full h-32 p-3 border border-slate-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-discord-500"
                          placeholder="What's on your mind?"
                          id="quick-note-text"></textarea>
                <div class="flex justify-end space-x-3 mt-4">
                    <button class="px-4 py-2 text-slate-600 hover:text-slate-800"
                            onclick="this.closest('.fixed').remove()">
                        Cancel
                    </button>
                    <button class="px-4 py-2 bg-discord-500 text-white rounded-lg hover:bg-discord-600"
                            onclick="mobileInterface.saveQuickNote()">
                        Save
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(dialog);
        
        const textarea = dialog.querySelector('#quick-note-text');
        textarea.focus();
    }
    
    async saveQuickNote() {
        const text = document.getElementById('quick-note-text').value.trim();
        if (!text) return;
        
        try {
            const response = await fetch('/api/notes', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    title: 'Quick Note',
                    content: text
                })
            });
            
            if (response.ok) {
                this.showNotification('Quick note saved', 'success');
                document.querySelector('.fixed').remove();
                
                // Refresh notes if content manager is available
                if (this.contentManager) {
                    this.contentManager.loadNotes();
                }
            }
        } catch (error) {
            console.error('Failed to save quick note:', error);
            this.showNotification('Failed to save note', 'error');
        }
    }
    
    shareNote(noteId) {
        if (navigator.share) {
            // Use native sharing if available
            const note = this.contentManager?.noteState.notes.get(noteId);
            if (note) {
                navigator.share({
                    title: note.title,
                    text: note.content,
                    url: `${window.location.origin}/note/${noteId}`
                });
            }
        } else {
            // Fallback to clipboard
            const url = `${window.location.origin}/note/${noteId}`;
            navigator.clipboard.writeText(url).then(() => {
                this.showNotification('Link copied to clipboard', 'success');
            });
        }
    }
    
    showMobileSettings() {
        // Show mobile-optimized settings screen
        console.log('Showing mobile settings...');
        // TODO: Implement mobile settings interface
    }
    
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `fixed top-4 left-4 right-4 mx-auto max-w-sm bg-white border-l-4 rounded-lg shadow-lg p-4 z-50 transition-all duration-300 ${
            type === 'success' ? 'border-green-500' :
            type === 'error' ? 'border-red-500' :
            'border-blue-500'
        }`;
        
        notification.innerHTML = `
            <div class="flex items-center">
                <div class="flex-shrink-0">
                    ${type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️'}
                </div>
                <div class="ml-3">
                    <p class="text-sm font-medium text-slate-800">${message}</p>
                </div>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Animate in
        requestAnimationFrame(() => {
            notification.style.transform = 'translateY(0)';
        });
        
        // Auto remove
        setTimeout(() => {
            notification.style.transform = 'translateY(-100%)';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.mobileInterface = new MobileInterface(
            window.contentManager, 
            window.dashboardPerformance
        );
    });
} else {
    window.mobileInterface = new MobileInterface(
        window.contentManager, 
        window.dashboardPerformance
    );
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MobileInterface;
}