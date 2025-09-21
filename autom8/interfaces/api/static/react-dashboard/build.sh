#!/bin/bash

# Autom8 React Dashboard Build Script
# This script builds the React dashboard and copies it to the FastAPI static directory

set -e

echo "🚀 Building Autom8 React Dashboard..."

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found. Make sure you're in the react-dashboard directory."
    exit 1
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Run type check
echo "🔍 Running type check..."
npm run type-check

# Build the project
echo "🏗️ Building for production..."
npm run build

# Check if build was successful
if [ ! -d "dist" ]; then
    echo "❌ Build failed: dist directory not found"
    exit 1
fi

echo "✅ Build completed successfully!"

# Optional: Copy to parent static directory to serve alongside existing dashboard.html
# Uncomment the following lines if you want to replace the existing static files
# echo "📁 Copying files to static directory..."
# cp -r dist/* ../
# echo "✅ Files copied to static directory"

echo "🎉 Dashboard is ready!"
echo ""
echo "To serve the dashboard:"
echo "1. Start the FastAPI server: python -m autom8.interfaces.api.server"
echo "2. Open http://localhost:8000 in your browser"
echo ""
echo "For development:"
echo "1. Run: npm run dev"
echo "2. Open http://localhost:3000 in your browser"