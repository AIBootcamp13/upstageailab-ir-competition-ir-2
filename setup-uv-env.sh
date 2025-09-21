#!/bin/bash
# UV Environment Setup Script
# This script sets up the UV environment variables to suppress warnings

# Set UV link mode to copy to avoid hardlink warnings
export UV_LINK_MODE=copy

# Source additional environment variables if .env.uv exists
if [ -f ".env.uv" ]; then
    source .env.uv
fi

echo "✅ UV environment configured"
echo "   UV_LINK_MODE=$UV_LINK_MODE"