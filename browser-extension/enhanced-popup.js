// Enhanced popup script with advanced capture capabilities
class SecondBrainCapture {
    constructor() {
        this.serverUrl = 'http://localhost:8082';
        this.isConnected = false;
        this.currentTab = null;
        this.advancedVisible = false;
        
        this.init();
    }

    async init() {
        await this.checkConnection();
        await this.getCurrentTab();
        this.setupEventListeners();
        await this.loadRecentCaptures();
    }

    async checkConnection() {
        try {
            const response = await fetch(`${this.serverUrl}/health`);
            this.isConnected = response.ok;
            this.updateStatus(this.isConnected ? 'Connected to Second Brain' : 'Connection failed');
        } catch (error) {
            this.isConnected = false;
            this.updateStatus('Second Brain not running');
        }
    }

    async getCurrentTab() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            this.currentTab = tab;
        } catch (error) {
            console.error('Failed to get current tab:', error);
        }
    }

    updateStatus(message) {
        const statusEl = document.getElementById('status');
        statusEl.textContent = message;
        statusEl.className = `status ${this.isConnected ? 'status-connected' : 'status-disconnected'}`;
    }

    showMessage(text, type = 'success') {
        const messagesEl = document.getElementById('messages');
        const messageEl = document.createElement('div');
        messageEl.className = type;
        messageEl.textContent = text;
        messagesEl.appendChild(messageEl);

        setTimeout(() => {
            messageEl.remove();
        }, 5000);
    }

    showProgress(show = true, text = 'Processing...', progress = 0) {
        const container = document.getElementById('progressContainer');
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');

        if (show) {
            container.classList.remove('hidden');
            progressText.textContent = text;
            progressBar.style.width = `${progress}%`;
        } else {
            container.classList.add('hidden');
        }
    }

    async sendToSecondBrain(data, endpoint = 'capture') {
        if (!this.isConnected) {
            throw new Error('Not connected to Second Brain');
        }

        const response = await fetch(`${this.serverUrl}/api/${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to capture content');
        }

        return response.json();
    }

    setupEventListeners() {
        // Quick capture buttons
        document.getElementById('captureSelection').addEventListener('click', () => this.captureSelection());
        document.getElementById('capturePage').addEventListener('click', () => this.captureFullPage());
        document.getElementById('captureBookmark').addEventListener('click', () => this.captureBookmark());
        document.getElementById('captureScreenshot').addEventListener('click', () => this.captureScreenshot());

        // Advanced capture buttons
        document.getElementById('capturePDF').addEventListener('click', () => this.capturePDFContent());
        document.getElementById('captureVideo').addEventListener('click', () => this.captureVideoInfo());
        document.getElementById('captureBulkUrls').addEventListener('click', () => this.captureBulkUrls());
        document.getElementById('captureArticle').addEventListener('click', () => this.captureArticle());

        // Toggle advanced options
        document.getElementById('toggleAdvanced').addEventListener('click', () => this.toggleAdvancedOptions());

        // Manual note
        document.getElementById('saveManual').addEventListener('click', () => this.saveManualNote());
        document.getElementById('manualNote').addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                this.saveManualNote();
            }
        });

        // Footer links
        document.getElementById('openDashboard').addEventListener('click', () => {
            chrome.tabs.create({ url: `${this.serverUrl}/dashboard` });
            window.close();
        });

        document.getElementById('openSettings').addEventListener('click', () => {
            chrome.runtime.openOptionsPage();
        });
    }

    toggleAdvancedOptions() {
        const optionsEl = document.getElementById('advancedOptions');
        const toggleBtn = document.getElementById('toggleAdvanced');
        
        this.advancedVisible = !this.advancedVisible;
        
        if (this.advancedVisible) {
            optionsEl.classList.remove('hidden');
            toggleBtn.textContent = '- Less Options';
        } else {
            optionsEl.classList.add('hidden');
            toggleBtn.textContent = '+ More Options';
        }
    }

    getAdvancedOptions() {
        return {
            includeImages: document.getElementById('includeImages')?.checked ?? true,
            takeScreenshot: document.getElementById('takeScreenshot')?.checked ?? true,
            aiProcessing: document.getElementById('aiProcessing')?.checked ?? true,
            customTags: document.getElementById('customTags')?.value || ''
        };
    }

    async captureSelection() {
        try {
            this.showProgress(true, 'Capturing selection...', 25);
            
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            
            // Execute content script to get selection
            const result = await chrome.tabs.sendMessage(tab.id, { 
                action: 'getSelection',
                options: this.getAdvancedOptions()
            });

            if (!result || !result.content) {
                throw new Error('No content selected');
            }

            this.showProgress(true, 'Processing selection...', 75);

            const data = {
                content: result.content,
                url: tab.url,
                title: result.title || tab.title,
                type: 'selection',
                ...this.getAdvancedOptions()
            };

            await this.sendToSecondBrain(data);
            
            this.showProgress(false);
            this.showMessage('Selection captured successfully!');
            await this.loadRecentCaptures();

        } catch (error) {
            this.showProgress(false);
            this.showMessage(`Failed to capture selection: ${error.message}`, 'error');
        }
    }

    async captureFullPage() {
        try {
            this.showProgress(true, 'Capturing full page...', 25);
            
            const options = this.getAdvancedOptions();
            
            const data = {
                url: this.currentTab.url,
                title: this.currentTab.title,
                type: 'full_page',
                ...options
            };

            this.showProgress(true, 'Processing page content...', 75);

            await this.sendToSecondBrain(data, 'web-ingestion');
            
            this.showProgress(false);
            this.showMessage('Full page captured successfully!');
            await this.loadRecentCaptures();

        } catch (error) {
            this.showProgress(false);
            this.showMessage(`Failed to capture page: ${error.message}`, 'error');
        }
    }

    async captureBookmark() {
        try {
            this.showProgress(true, 'Creating bookmark...', 50);

            const data = {
                url: this.currentTab.url,
                title: this.currentTab.title,
                type: 'bookmark',
                tags: 'bookmark, ' + (this.getAdvancedOptions().customTags || ''),
                content: `Bookmarked: ${this.currentTab.title}\nURL: ${this.currentTab.url}`
            };

            await this.sendToSecondBrain(data);
            
            this.showProgress(false);
            this.showMessage('Bookmark saved!');
            await this.loadRecentCaptures();

        } catch (error) {
            this.showProgress(false);
            this.showMessage(`Failed to save bookmark: ${error.message}`, 'error');
        }
    }

    async captureScreenshot() {
        try {
            this.showProgress(true, 'Taking screenshot...', 25);

            // Capture visible tab
            const dataUrl = await chrome.tabs.captureVisibleTab(
                this.currentTab.windowId, 
                { format: 'png', quality: 90 }
            );

            this.showProgress(true, 'Processing screenshot...', 75);

            const data = {
                content: `Screenshot of ${this.currentTab.title}`,
                url: this.currentTab.url,
                title: `Screenshot: ${this.currentTab.title}`,
                type: 'screenshot',
                image_data: dataUrl,
                tags: 'screenshot, ' + (this.getAdvancedOptions().customTags || '')
            };

            await this.sendToSecondBrain(data);
            
            this.showProgress(false);
            this.showMessage('Screenshot captured!');
            await this.loadRecentCaptures();

        } catch (error) {
            this.showProgress(false);
            this.showMessage(`Failed to capture screenshot: ${error.message}`, 'error');
        }
    }

    async capturePDFContent() {
        try {
            if (!this.currentTab.url.endsWith('.pdf')) {
                throw new Error('Current page is not a PDF');
            }

            this.showProgress(true, 'Extracting PDF content...', 50);

            const data = {
                url: this.currentTab.url,
                title: this.currentTab.title,
                type: 'pdf',
                ...this.getAdvancedOptions()
            };

            await this.sendToSecondBrain(data, 'web-ingestion');
            
            this.showProgress(false);
            this.showMessage('PDF content extracted!');
            await this.loadRecentCaptures();

        } catch (error) {
            this.showProgress(false);
            this.showMessage(`PDF capture failed: ${error.message}`, 'error');
        }
    }

    async captureVideoInfo() {
        this.showMessage('Video capture coming soon!', 'error');
    }

    async captureBulkUrls() {
        const urls = prompt('Enter URLs (one per line):');
        if (!urls) return;

        const urlList = urls.split('\n').filter(url => url.trim());
        
        try {
            this.showProgress(true, `Processing ${urlList.length} URLs...`, 0);

            for (let i = 0; i < urlList.length; i++) {
                const url = urlList[i].trim();
                const progress = ((i + 1) / urlList.length) * 100;
                
                this.showProgress(true, `Processing URL ${i + 1}/${urlList.length}...`, progress);

                const data = {
                    url: url,
                    type: 'bulk_url',
                    ...this.getAdvancedOptions()
                };

                await this.sendToSecondBrain(data, 'web-ingestion');
                
                // Small delay to avoid overwhelming the server
                await new Promise(resolve => setTimeout(resolve, 500));
            }

            this.showProgress(false);
            this.showMessage(`Successfully processed ${urlList.length} URLs!`);
            await this.loadRecentCaptures();

        } catch (error) {
            this.showProgress(false);
            this.showMessage(`Bulk URL capture failed: ${error.message}`, 'error');
        }
    }

    async captureArticle() {
        try {
            this.showProgress(true, 'Extracting article content...', 25);

            // Use Readability-style extraction
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            
            const result = await chrome.tabs.sendMessage(tab.id, { 
                action: 'extractArticle',
                options: this.getAdvancedOptions()
            });

            if (!result || !result.content) {
                throw new Error('Failed to extract article content');
            }

            this.showProgress(true, 'Processing article...', 75);

            const data = {
                content: result.content,
                url: tab.url,
                title: result.title || tab.title,
                type: 'article',
                author: result.author,
                publishedDate: result.publishedDate,
                ...this.getAdvancedOptions()
            };

            await this.sendToSecondBrain(data);
            
            this.showProgress(false);
            this.showMessage('Article captured successfully!');
            await this.loadRecentCaptures();

        } catch (error) {
            this.showProgress(false);
            this.showMessage(`Article capture failed: ${error.message}`, 'error');
        }
    }

    async saveManualNote() {
        const noteContent = document.getElementById('manualNote').value.trim();
        const tags = document.getElementById('manualTags').value.trim();

        if (!noteContent) {
            this.showMessage('Please enter some content', 'error');
            return;
        }

        try {
            this.showProgress(true, 'Saving note...', 75);

            const data = {
                content: noteContent,
                tags: tags,
                type: 'manual_note',
                url: this.currentTab?.url || '',
                title: 'Manual Note'
            };

            await this.sendToSecondBrain(data);

            // Clear form
            document.getElementById('manualNote').value = '';
            document.getElementById('manualTags').value = '';
            
            this.showProgress(false);
            this.showMessage('Note saved successfully!');
            await this.loadRecentCaptures();

        } catch (error) {
            this.showProgress(false);
            this.showMessage(`Failed to save note: ${error.message}`, 'error');
        }
    }

    async loadRecentCaptures() {
        try {
            const response = await fetch(`${this.serverUrl}/api/notes?limit=5&recent=true`);
            if (!response.ok) throw new Error('Failed to load recent captures');

            const data = await response.json();
            this.displayRecentCaptures(data.notes || []);

        } catch (error) {
            document.getElementById('recentCaptures').innerHTML = 
                '<div style="color: #dc2626; font-size: 12px;">Failed to load recent captures</div>';
        }
    }

    displayRecentCaptures(captures) {
        const container = document.getElementById('recentCaptures');
        
        if (!captures.length) {
            container.innerHTML = '<div style="color: #6b7280; font-size: 12px;">No recent captures</div>';
            return;
        }

        container.innerHTML = captures.map(capture => {
            const date = new Date(capture.created_at).toLocaleDateString();
            const truncatedTitle = capture.title.length > 40 
                ? capture.title.substring(0, 40) + '...' 
                : capture.title;

            return `
                <div class="recent-item" data-note-id="${capture.id}">
                    <div class="recent-item-title">${truncatedTitle}</div>
                    <div class="recent-item-meta">${date} • ${capture.tags || 'No tags'}</div>
                </div>
            `;
        }).join('');

        // Add click handlers for recent items
        container.querySelectorAll('.recent-item').forEach(item => {
            item.addEventListener('click', () => {
                const noteId = item.dataset.noteId;
                chrome.tabs.create({ url: `${this.serverUrl}/notes/${noteId}` });
                window.close();
            });
        });
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SecondBrainCapture();
});