/**
 * HTMX Helper Functions for Second Brain
 *
 * Minimal JavaScript utilities to enhance HTMX functionality.
 * Keeps JavaScript to a minimum - most interactivity is handled by HTMX + Alpine.js
 */

// ============================================================================
// Toast Notifications
// ============================================================================

/**
 * Show a toast notification
 * @param {string} message - Main message
 * @param {string} type - Type: success, error, warning, info
 * @param {string} detail - Optional detail text
 * @param {number} duration - Duration in ms (0 = don't auto-dismiss)
 */
function showToast(message, type = 'info', detail = '', duration = 3000) {
    window.dispatchEvent(new CustomEvent('show-toast', {
        detail: { message, type, detail, duration }
    }));
}

// Make showToast globally available
window.showToast = showToast;


// ============================================================================
// HTMX Event Handlers
// ============================================================================

// Global error handler for HTMX requests
document.body.addEventListener('htmx:responseError', function(event) {
    console.error('HTMX Error:', event.detail);

    const xhr = event.detail.xhr;
    let errorMessage = 'An error occurred';

    try {
        const response = JSON.parse(xhr.responseText);
        errorMessage = response.detail || response.message || errorMessage;
    } catch (e) {
        errorMessage = xhr.statusText || errorMessage;
    }

    showToast('Request Failed', 'error', errorMessage, 5000);
});

// Handle network errors
document.body.addEventListener('htmx:sendError', function(event) {
    console.error('HTMX Network Error:', event.detail);
    showToast('Network Error', 'error', 'Unable to reach the server. Please check your connection.', 5000);
});

// Handle timeouts
document.body.addEventListener('htmx:timeout', function(event) {
    console.error('HTMX Timeout:', event.detail);
    showToast('Request Timeout', 'warning', 'The request took too long. Please try again.', 5000);
});

// Log successful requests in development
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    document.body.addEventListener('htmx:afterRequest', function(event) {
        if (event.detail.successful) {
            console.log('✅ HTMX Success:', event.detail);
        }
    });
}


// ============================================================================
// Form Helpers
// ============================================================================

/**
 * Reset form after successful HTMX submission
 * Usage: hx-on::after-request="resetFormOnSuccess(event)"
 */
window.resetFormOnSuccess = function(event) {
    if (event.detail.successful) {
        event.target.reset();
    }
};


// ============================================================================
// Confirmation Dialogs
// ============================================================================

// Custom confirmation dialog for delete actions
document.body.addEventListener('htmx:confirm', function(event) {
    const question = event.detail.question;

    // You can customize this with a prettier modal later
    if (!confirm(question)) {
        event.preventDefault();
    }
});


// ============================================================================
// Loading States
// ============================================================================

// Add loading class to elements during HTMX requests
document.body.addEventListener('htmx:beforeRequest', function(event) {
    const elt = event.detail.elt;

    // Add loading state to buttons
    if (elt.tagName === 'BUTTON') {
        elt.classList.add('htmx-loading');
        elt.disabled = true;
    }
});

document.body.addEventListener('htmx:afterRequest', function(event) {
    const elt = event.detail.elt;

    // Remove loading state from buttons
    if (elt.tagName === 'BUTTON') {
        elt.classList.remove('htmx-loading');
        elt.disabled = false;
    }
});


// ============================================================================
// URL Helpers
// ============================================================================

/**
 * Update browser URL without reload (for search, filters, etc.)
 */
document.body.addEventListener('htmx:pushedIntoHistory', function(event) {
    console.log('URL updated:', event.detail.path);
});


// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 HTMX Helpers Loaded');

    // Configure HTMX globally
    if (typeof htmx !== 'undefined') {
        // Set default timeout to 30 seconds
        htmx.config.timeout = 30000;

        // Enable history support
        htmx.config.historyCacheSize = 10;

        console.log('✅ HTMX configured');
    }

    // Show welcome message (only in development)
    if (window.location.hostname === 'localhost') {
        console.log('%c Second Brain - HTMX Edition ', 'background: #6366f1; color: white; font-size: 14px; padding: 4px 8px; border-radius: 4px;');
        console.log('Using HTMX for dynamic interactions with zero build step!');
    }
});
