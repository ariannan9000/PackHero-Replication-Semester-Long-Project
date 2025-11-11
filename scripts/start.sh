#!/bin/bash

echo "🐳 Starting Docker isolated environment..."
echo ""

# Get directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$SCRIPT_DIR/../workspace"

# Create workspace if needed
mkdir -p "$WORKSPACE_DIR"

echo "📁 Workspace: $WORKSPACE_DIR"
echo "🔒 Isolation: Active"
echo ""

# Remove any existing container
docker rm -f packhero-isolated 2>/dev/null || true

# Rebuild the image
echo "🔨 Rebuilding Docker image for ARM64..."
docker build -t packhero-env .

echo ""
echo "🚀 Starting container..."

# Start Docker container
docker run -it --rm \
    --name packhero-isolated \
    -v "$WORKSPACE_DIR:/workspace/data" \
    -v "$SCRIPT_DIR/download_malware.py:/workspace/download_malware.py:ro" \
    -v "$SCRIPT_DIR/organize_samples.py:/workspace/organize_samples.py:ro" \
    -w /workspace \
    packhero-env

echo ""
echo "✅ Exited isolated environment"
