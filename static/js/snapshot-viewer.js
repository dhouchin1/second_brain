/**
 * Snapshot Viewer Component - Clean URL Implementation
 * Displays web ingestion snapshots without exposing sensitive data in URLs
 */

class SnapshotViewer {
    constructor(stateManager, navigationManager) {
        this.stateManager = stateManager;
        this.navigationManager = navigationManager;
        this.isVisible = false;

        this.init();
    }

    init() {
        // Listen for snapshot state changes
        this.stateManager.subscribe('snapshotViewOpen', (isOpen) => {
            this.toggleView(isOpen);
        });

        this.stateManager.subscribe('selectedSnapshot', (snapshot) => {
            if (snapshot) {
                this.displaySnapshot(snapshot);
            }
        });

        this.stateManager.subscribe('selectedArtifact', (artifact) => {
            if (artifact) {
                this.displayArtifact(artifact);
            }
        });

        // Create snapshot viewer container
        this.createContainer();
    }

    createContainer() {
        const container = document.createElement('div');
        container.id = 'snapshotViewer';
        container.className = 'fixed inset-0 bg-gray-900 bg-opacity-50 z-50 hidden overflow-y-auto';
        container.innerHTML = `
            <div class="min-h-screen px-4 py-8">
                <div class="max-w-5xl mx-auto bg-white rounded-lg shadow-xl">
                    <!-- Header -->
                    <div class="flex items-center justify-between p-6 border-b border-gray-200">
                        <div>
                            <h1 class="text-2xl font-semibold text-gray-900" id="snapshotTitle">Snapshot View</h1>
                            <p class="text-sm text-gray-600" id="snapshotDate"></p>
                        </div>
                        <div class="flex items-center space-x-3">
                            <button onclick="window.snapshotViewer.openOriginalUrl()"
                                    id="originalUrlBtn"
                                    class="inline-flex items-center px-3 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200">
                                <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path>
                                </svg>
                                Open Original
                            </button>
                            <button onclick="window.snapshotViewer.close()"
                                    class="text-gray-400 hover:text-gray-600">
                                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                                </svg>
                            </button>
                        </div>
                    </div>

                    <!-- Content -->
                    <div class="p-6 space-y-6" id="snapshotContent">
                        <!-- Summary Section -->
                        <div id="summarySection" class="hidden">
                            <h3 class="text-lg font-medium text-gray-900 mb-3">Summary</h3>
                            <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
                                <p id="snapshotSummary" class="text-gray-800"></p>
                            </div>
                        </div>

                        <!-- Metadata Section -->
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div>
                                <h3 class="text-lg font-medium text-gray-900 mb-3">Metadata</h3>
                                <div id="metadataContent" class="space-y-2 text-sm text-gray-600">
                                    <!-- Metadata will be populated here -->
                                </div>
                            </div>
                            <div>
                                <h3 class="text-lg font-medium text-gray-900 mb-3">Configuration</h3>
                                <div id="configContent" class="space-y-2 text-sm text-gray-600">
                                    <!-- Config will be populated here -->
                                </div>
                            </div>
                        </div>

                        <!-- Artifacts Section -->
                        <div>
                            <h3 class="text-lg font-medium text-gray-900 mb-3">Artifacts</h3>
                            <div id="artifactsContent" class="space-y-4">
                                <!-- Artifacts will be populated here -->
                            </div>
                        </div>

                        <!-- Raw Content Section -->
                        <div>
                            <h3 class="text-lg font-medium text-gray-900 mb-3">Raw Content</h3>
                            <div class="bg-gray-50 border border-gray-200 rounded-lg p-4 max-h-96 overflow-y-auto">
                                <pre id="rawContent" class="text-sm text-gray-800 whitespace-pre-wrap"></pre>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(container);
        this.container = container;

        // Add keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (this.isVisible && e.key === 'Escape') {
                this.close();
            }
        });

        // Close on background click
        container.addEventListener('click', (e) => {
            if (e.target === container) {
                this.close();
            }
        });
    }

    toggleView(isOpen) {
        this.isVisible = isOpen;
        this.container.classList.toggle('hidden', !isOpen);

        if (isOpen) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = '';
        }
    }

    displaySnapshot(snapshot) {
        // Update title and metadata
        document.getElementById('snapshotTitle').textContent =
            snapshot.manifest?.title || snapshot.note?.title || 'Snapshot View';

        document.getElementById('snapshotDate').textContent =
            `Captured on ${snapshot.manifest?.generated_at || snapshot.note?.created_at || 'Unknown date'}`;

        // Display summary if available
        const summarySection = document.getElementById('summarySection');
        const summaryText = document.getElementById('snapshotSummary');

        if (snapshot.manifest?.ai_summary || snapshot.note?.summary) {
            summaryText.textContent = snapshot.manifest?.ai_summary || snapshot.note?.summary;
            summarySection.classList.remove('hidden');
        } else {
            summarySection.classList.add('hidden');
        }

        // Display metadata
        this.displayMetadata(snapshot.manifest?.metadata || {}, snapshot.file_metadata || {});

        // Display configuration
        this.displayConfiguration(snapshot.manifest?.config || {});

        // Display artifacts
        this.displayArtifacts(snapshot.artifacts || []);

        // Display raw content
        document.getElementById('rawContent').textContent = snapshot.note?.content || 'No content available';

        // Update original URL button
        const originalUrlBtn = document.getElementById('originalUrlBtn');
        if (snapshot.manifest?.url || snapshot.file_metadata?.source_url) {
            this.originalUrl = snapshot.manifest?.url || snapshot.file_metadata?.source_url;
            originalUrlBtn.style.display = 'inline-flex';
        } else {
            originalUrlBtn.style.display = 'none';
        }
    }

    displayMetadata(manifest_meta, file_meta) {
        const container = document.getElementById('metadataContent');
        container.innerHTML = '';

        const metadata = {
            'Domain': manifest_meta.domain || file_meta.domain,
            'Handler': manifest_meta.handler,
            'Tags': manifest_meta.ai_tags ? manifest_meta.ai_tags.join(', ') : null,
            'Repository Stars': manifest_meta.repo?.stars,
            'Content Type': file_meta.content_type,
            'File Size': file_meta.file_size ? this.formatFileSize(file_meta.file_size) : null
        };

        Object.entries(metadata).forEach(([key, value]) => {
            if (value) {
                const item = document.createElement('div');
                item.innerHTML = `<span class="font-medium">${key}:</span> ${value}`;
                container.appendChild(item);
            }
        });

        if (container.children.length === 0) {
            container.innerHTML = '<div class="text-gray-400">No metadata available</div>';
        }
    }

    displayConfiguration(config) {
        const container = document.getElementById('configContent');
        container.innerHTML = '';

        if (Object.keys(config).length === 0) {
            container.innerHTML = '<div class="text-gray-400">No configuration data</div>';
            return;
        }

        Object.entries(config).forEach(([key, value]) => {
            const item = document.createElement('div');
            const displayKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            item.innerHTML = `<span class="font-medium">${displayKey}:</span> ${value}`;
            container.appendChild(item);
        });
    }

    displayArtifacts(artifacts) {
        const container = document.getElementById('artifactsContent');
        container.innerHTML = '';

        if (artifacts.length === 0) {
            container.innerHTML = '<div class="text-gray-400">No artifacts captured</div>';
            return;
        }

        artifacts.forEach((artifact, index) => {
            const artifactElement = this.createArtifactElement(artifact, index);
            container.appendChild(artifactElement);
        });
    }

    createArtifactElement(artifact, index) {
        const div = document.createElement('div');
        div.className = 'border border-gray-200 rounded-lg p-4 bg-gray-50';

        const isImage = artifact.mime_type?.startsWith('image/');
        const isVideo = artifact.mime_type?.startsWith('video/');
        const isAudio = artifact.mime_type?.startsWith('audio/');
        const isPdf = artifact.mime_type?.includes('pdf');

        div.innerHTML = `
            <div class="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-3">
                <div>
                    <p class="font-medium text-gray-900">${artifact.label || artifact.type || 'Artifact'}</p>
                    <p class="text-xs text-gray-500">
                        ${artifact.mime_type || 'application/octet-stream'} •
                        ${artifact.size ? this.formatFileSize(artifact.size) : '0 bytes'}
                    </p>
                </div>
                <div class="flex items-center gap-2">
                    ${artifact.inline_url ? `
                        <button onclick="window.snapshotViewer.viewArtifact(${index})"
                                class="inline-flex items-center px-3 py-1.5 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700">
                            <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path>
                            </svg>
                            Preview
                        </button>
                    ` : ''}
                    ${artifact.view_url ? `
                        <button onclick="window.snapshotViewer.downloadArtifact(${index})"
                                class="inline-flex items-center px-3 py-1.5 text-sm font-medium text-gray-700 bg-gray-200 rounded-md hover:bg-gray-300">
                            <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                            </svg>
                            Download
                        </button>
                    ` : ''}
                </div>
            </div>
        `;

        // Add preview if applicable
        if (isImage && artifact.view_url) {
            div.innerHTML += `<img src="${artifact.view_url}" alt="Snapshot image" class="mt-3 rounded-md border border-gray-300 max-h-64 object-contain" />`;
        } else if (isVideo && artifact.view_url) {
            div.innerHTML += `
                <video controls preload="metadata" class="mt-3 w-full rounded-lg border border-gray-300 max-h-64">
                    <source src="${artifact.view_url}" type="${artifact.mime_type}">
                    Your browser does not support embedded video.
                </video>
            `;
        } else if (isAudio && artifact.view_url) {
            div.innerHTML += `
                <audio controls class="mt-3 w-full">
                    <source src="${artifact.view_url}" type="${artifact.mime_type}">
                    Your browser does not support audio playback.
                </audio>
            `;
        } else if (isPdf && artifact.view_url) {
            div.innerHTML += `<iframe src="${artifact.view_url}" class="mt-3 w-full h-64 border border-gray-300 rounded-lg" title="PDF preview"></iframe>`;
        }

        return div;
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Public methods for interaction
    async viewSnapshot(noteId) {
        try {
            // Fetch snapshot data without exposing ID in URL
            const snapshotData = await this.fetchSnapshotData(noteId);
            this.stateManager.selectSnapshot(snapshotData);
            this.navigationManager.navigateTo('/snapshot');
        } catch (error) {
            console.error('Failed to load snapshot:', error);
            this.showError('Failed to load snapshot');
        }
    }

    async showSnapshotList() {
        try {
            const snapshots = await this.fetchSnapshotList();
            this.displaySnapshotList(snapshots);
        } catch (error) {
            console.error('Failed to load snapshots:', error);
            this.showError('Failed to load snapshots');
        }
    }

    async fetchSnapshotList() {
        const response = await fetch('/api/snapshots', {
            credentials: 'same-origin'
        });

        if (!response.ok) {
            throw new Error(`Failed to fetch snapshots: ${response.status}`);
        }

        return await response.json();
    }

    displaySnapshotList(snapshots) {
        // Create snapshot list modal
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-gray-900 bg-opacity-50 z-50 overflow-y-auto';
        modal.innerHTML = `
            <div class="min-h-screen px-4 py-8">
                <div class="max-w-4xl mx-auto bg-white rounded-lg shadow-xl">
                    <div class="flex items-center justify-between p-6 border-b border-gray-200">
                        <h1 class="text-2xl font-semibold text-gray-900">Web Snapshots</h1>
                        <button onclick="this.closest('.fixed').remove()"
                                class="text-gray-400 hover:text-gray-600">
                            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                            </svg>
                        </button>
                    </div>
                    <div class="p-6">
                        ${snapshots.length > 0 ? this.renderSnapshotGrid(snapshots) : this.renderEmptyState()}
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Close on background click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });

        // Close on escape
        const escapeHandler = (e) => {
            if (e.key === 'Escape') {
                modal.remove();
                document.removeEventListener('keydown', escapeHandler);
            }
        };
        document.addEventListener('keydown', escapeHandler);
    }

    renderSnapshotGrid(snapshots) {
        return `
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                ${snapshots.map(snapshot => this.renderSnapshotCard(snapshot)).join('')}
            </div>
        `;
    }

    renderSnapshotCard(snapshot) {
        const metadata = snapshot.file_metadata ? JSON.parse(snapshot.file_metadata) : {};
        const hasScreenshot = metadata.artifacts?.some(a => a.type === 'screenshot');
        const domain = metadata.source_url ? new URL(metadata.source_url).hostname : 'Unknown';

        return `
            <div class="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
                 onclick="window.snapshotViewer.viewSnapshot(${snapshot.id}); this.closest('.fixed').remove();">
                <div class="flex items-start justify-between mb-3">
                    <div class="flex-1">
                        <h3 class="font-medium text-gray-900 line-clamp-2">${snapshot.title || 'Untitled Snapshot'}</h3>
                        <p class="text-sm text-gray-500 mt-1">${domain}</p>
                    </div>
                    ${hasScreenshot ? '<div class="ml-2 text-blue-500">📸</div>' : ''}
                </div>

                <div class="flex items-center justify-between text-xs text-gray-400">
                    <span>${new Date(snapshot.created_at).toLocaleDateString()}</span>
                    <span>${metadata.size_bytes ? this.formatFileSize(metadata.size_bytes) : ''}</span>
                </div>

                ${snapshot.summary ? `
                    <p class="text-sm text-gray-600 mt-2 line-clamp-2">${snapshot.summary}</p>
                ` : ''}

                <div class="mt-3 flex items-center text-xs text-gray-500">
                    <svg class="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    Click to view snapshot
                </div>
            </div>
        `;
    }

    renderEmptyState() {
        return `
            <div class="text-center py-12">
                <div class="w-16 h-16 mx-auto mb-4 text-gray-300">
                    <svg fill="currentColor" viewBox="0 0 20 20">
                        <path d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z"></path>
                    </svg>
                </div>
                <h3 class="text-lg font-medium text-gray-900 mb-2">No snapshots yet</h3>
                <p class="text-gray-500 mb-6">Web content snapshots will appear here when you capture web pages.</p>
                <button onclick="this.closest('.fixed').remove();"
                        class="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700">
                    Got it
                </button>
            </div>
        `;
    }

    async fetchSnapshotData(noteId) {
        const response = await fetch(`/api/snapshot/${noteId}`, {
            credentials: 'same-origin'
        });

        if (!response.ok) {
            throw new Error(`Failed to fetch snapshot: ${response.status}`);
        }

        return await response.json();
    }

    viewArtifact(index) {
        const snapshot = this.stateManager.getState('selectedSnapshot');
        if (snapshot && snapshot.artifacts && snapshot.artifacts[index]) {
            const artifact = snapshot.artifacts[index];
            this.stateManager.selectArtifact(artifact);
            // Open artifact in new tab to avoid URL pollution
            if (artifact.inline_url) {
                window.open(artifact.inline_url, '_blank', 'noopener,noreferrer');
            }
        }
    }

    downloadArtifact(index) {
        const snapshot = this.stateManager.getState('selectedSnapshot');
        if (snapshot && snapshot.artifacts && snapshot.artifacts[index]) {
            const artifact = snapshot.artifacts[index];
            if (artifact.view_url) {
                // Create temporary download link
                const a = document.createElement('a');
                a.href = artifact.view_url;
                a.download = artifact.label || `artifact_${index}`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }
        }
    }

    openOriginalUrl() {
        if (this.originalUrl) {
            window.open(this.originalUrl, '_blank', 'noopener,noreferrer');
        }
    }

    close() {
        this.stateManager.closeSnapshot();
        this.navigationManager.navigateTo('/');
    }

    showError(message) {
        if (window.showToast) {
            window.showToast(message, 'error');
        } else {
            alert(message);
        }
    }
}

// Initialize snapshot viewer when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (window.stateManager && window.navigationManager) {
        window.snapshotViewer = new SnapshotViewer(window.stateManager, window.navigationManager);
        console.log('✅ Snapshot Viewer initialized with clean URLs');
    }
});

// Export for modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SnapshotViewer;
}