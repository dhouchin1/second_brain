/**
 * Simple Service Worker for Second Brain
 * Optimized for performance without complexity
 */

const CACHE_NAME = 'second-brain-v1';
const STATIC_CACHE = [
    '/static/js/dashboard-core.js',
    '/static/js/dashboard-utilities.js',
    '/static/css/design-system.css',
    '/static/css/components.css',
    '/static/css/dashboard-enhanced.css',
    '/static/manifest.json'
];

// Cache static assets on install
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => cache.addAll(STATIC_CACHE))
            .then(() => self.skipWaiting())
    );
});

// Clean up old caches on activate
self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.map(cacheName => {
                    if (cacheName !== CACHE_NAME) {
                        return caches.delete(cacheName);
                    }
                })
            );
        }).then(() => self.clients.claim())
    );
});

// Network first for API, cache first for static assets
self.addEventListener('fetch', event => {
    const { request } = event;
    const url = new URL(request.url);

    // Skip cross-origin requests
    if (url.origin !== location.origin) {
        return;
    }

    // API requests - network first with fallback
    if (url.pathname.startsWith('/api/')) {
        event.respondWith(
            fetch(request)
                .then(response => {
                    // Cache successful GET requests
                    if (request.method === 'GET' && response.status === 200) {
                        const responseClone = response.clone();
                        caches.open(CACHE_NAME).then(cache => {
                            cache.put(request, responseClone);
                        });
                    }
                    return response;
                })
                .catch(() => {
                    // Fallback to cache for offline access
                    return caches.match(request);
                })
        );
        return;
    }

    // Static assets - cache first
    if (url.pathname.startsWith('/static/')) {
        event.respondWith(
            caches.match(request)
                .then(response => {
                    return response || fetch(request).then(fetchResponse => {
                        const responseClone = fetchResponse.clone();
                        caches.open(CACHE_NAME).then(cache => {
                            cache.put(request, responseClone);
                        });
                        return fetchResponse;
                    });
                })
        );
        return;
    }

    // For everything else, try network first
    event.respondWith(
        fetch(request).catch(() => caches.match(request))
    );
});