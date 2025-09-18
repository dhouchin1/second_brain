#!/bin/bash

# build-production.sh - Production build script for Second Brain Extension

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
CHROME_BUILD="$BUILD_DIR/chrome"
FIREFOX_BUILD="$BUILD_DIR/firefox"

echo "🚀 Building Second Brain Extension for Production"
echo "================================================"

# Clean previous builds
rm -rf "$BUILD_DIR"
mkdir -p "$CHROME_BUILD" "$FIREFOX_BUILD"

# Copy base files to both builds
echo "📁 Copying base files..."
cp -r icons/ "$CHROME_BUILD/"
cp -r icons/ "$FIREFOX_BUILD/"
cp *.html "$CHROME_BUILD/"
cp *.html "$FIREFOX_BUILD/"
cp *.css "$CHROME_BUILD/"
cp *.css "$FIREFOX_BUILD/"
cp options.js "$CHROME_BUILD/"
cp options.js "$FIREFOX_BUILD/"
cp validate.js "$CHROME_BUILD/"
cp validate.js "$FIREFOX_BUILD/"

# Use production background script
echo "🔧 Using production background script..."
cp production-background.js "$CHROME_BUILD/background.js"
cp production-background.js "$FIREFOX_BUILD/background.js"

# Copy and optimize popup scripts
echo "📦 Optimizing popup scripts..."
cp popup.js "$CHROME_BUILD/"
cp popup.js "$FIREFOX_BUILD/"
cp content.js "$CHROME_BUILD/"
cp content.js "$FIREFOX_BUILD/"

# Chrome manifest (Manifest V3)
echo "⚙️  Creating Chrome manifest..."
cat > "$CHROME_BUILD/manifest.json" << 'EOF'
{
  "manifest_version": 3,
  "name": "Second Brain",
  "version": "1.0.1",
  "description": "Capture web content to your personal Second Brain with AI-powered processing, summarization, and intelligent organization.",
  "permissions": [
    "activeTab",
    "contextMenus",
    "storage",
    "scripting",
    "notifications",
    "offscreen",
    "alarms"
  ],
  "host_permissions": [
    "*://*/*"
  ],
  "optional_host_permissions": [
    "http://localhost:*/*",
    "https://*.secondbrain.ai/*"
  ],
  "background": {
    "service_worker": "background.js"
  },
  "content_scripts": [
    {
      "matches": ["<all_urls>"],
      "js": ["content.js"],
      "css": ["content.css"]
    }
  ],
  "action": {
    "default_popup": "popup.html",
    "default_title": "Second Brain Capture"
  },
  "commands": {
    "quick-capture": {
      "suggested_key": {
        "default": "Ctrl+Shift+S",
        "mac": "Command+Shift+S"
      },
      "description": "Quick capture selection"
    },
    "capture-page": {
      "suggested_key": {
        "default": "Ctrl+Shift+P",
        "mac": "Command+Shift+P"
      },
      "description": "Capture full page"
    }
  },
  "web_accessible_resources": [
    {
      "resources": ["icons/*", "content.css"],
      "matches": ["<all_urls>"]
    }
  ],
  "options_page": "options.html",
  "icons": {
    "16": "icons/brain-16.png",
    "32": "icons/brain-32.png",
    "48": "icons/brain-48.png",
    "128": "icons/brain-128.png"
  }
}
EOF

# Firefox manifest (Manifest V2 compatibility)
echo "🦊 Creating Firefox manifest..."
cat > "$FIREFOX_BUILD/manifest.json" << 'EOF'
{
  "manifest_version": 2,
  "name": "Second Brain",
  "version": "1.0.1",
  "description": "Capture web content to your personal Second Brain with AI-powered processing, summarization, and intelligent organization.",
  "permissions": [
    "activeTab",
    "contextMenus",
    "storage",
    "notifications",
    "alarms",
    "*://*/*"
  ],
  "optional_permissions": [
    "http://localhost:*/*",
    "https://*.secondbrain.ai/*"
  ],
  "background": {
    "scripts": ["background.js"],
    "persistent": false
  },
  "content_scripts": [
    {
      "matches": ["<all_urls>"],
      "js": ["content.js"],
      "css": ["content.css"]
    }
  ],
  "browser_action": {
    "default_popup": "popup.html",
    "default_title": "Second Brain Capture"
  },
  "commands": {
    "quick-capture": {
      "suggested_key": {
        "default": "Ctrl+Shift+S",
        "mac": "Command+Shift+S"
      },
      "description": "Quick capture selection"
    },
    "capture-page": {
      "suggested_key": {
        "default": "Ctrl+Shift+P",
        "mac": "Command+Shift+P"
      },
      "description": "Capture full page"
    }
  },
  "web_accessible_resources": ["icons/*", "content.css"],
  "options_page": "options.html",
  "icons": {
    "16": "icons/brain-16.png",
    "32": "icons/brain-32.png",
    "48": "icons/brain-48.png",
    "128": "icons/brain-128.png"
  },
  "applications": {
    "gecko": {
      "id": "second-brain@secondbrain.ai",
      "strict_min_version": "91.0"
    }
  }
}
EOF

# Generate icon files if they don't exist
echo "🎨 Generating icon files..."
if [ ! -f "icons/brain-16.png" ]; then
    echo "Warning: Icon files not found. Creating placeholder icons..."
    mkdir -p "$CHROME_BUILD/icons" "$FIREFOX_BUILD/icons"
    
    # Create placeholder SVG if needed
    if [ ! -f "icons/brain.svg" ]; then
        cat > "$CHROME_BUILD/icons/brain.svg" << 'EOF'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
  <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
</svg>
EOF
        cp "$CHROME_BUILD/icons/brain.svg" "$FIREFOX_BUILD/icons/"
    fi
fi

# Create package archives
echo "📦 Creating package archives..."
cd "$CHROME_BUILD"
zip -r ../second-brain-chrome-v1.0.1.zip . -x "*.DS_Store"

cd "$FIREFOX_BUILD"
zip -r ../second-brain-firefox-v1.0.1.zip . -x "*.DS_Store"

cd "$SCRIPT_DIR"

echo ""
echo "✅ Production Build Complete!"
echo "=============================="
echo "Chrome package: build/second-brain-chrome-v1.0.1.zip"
echo "Firefox package: build/second-brain-firefox-v1.0.1.zip"
echo ""
echo "📋 Next Steps:"
echo "1. Test both builds locally"
echo "2. Submit Chrome package to Chrome Web Store"
echo "3. Submit Firefox package to Firefox Add-ons"
echo ""
echo "🔗 Store URLs:"
echo "Chrome Web Store: https://chrome.google.com/webstore/devconsole"
echo "Firefox Add-ons: https://addons.mozilla.org/developers/"