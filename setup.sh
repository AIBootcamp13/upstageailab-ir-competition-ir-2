#!/bin/bash
# One-command setup script for the Information Retrieval RAG project
set -e

echo "🚀 Setting up Information Retrieval RAG Project"
echo "==============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "This setup script is designed for Linux. Please see README.md for other platforms."
    exit 1
fi

# Check for required tools
command -v curl >/dev/null 2>&1 || { print_error "curl is required but not installed. Please install curl."; exit 1; }
command -v git >/dev/null 2>&1 || { print_error "git is required but not installed. Please install git."; exit 1; }

print_status "Checking system requirements..."

# Install uv if not present
if ! command -v uv >/dev/null 2>&1; then
    print_status "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    print_success "uv installed successfully"
else
    print_success "uv is already installed"
fi

# Verify uv works
if ! uv --version >/dev/null 2>&1; then
    print_error "uv installation failed. Please check your PATH."
    exit 1
fi

# Set up environment
print_status "Configuring environment..."
source setup-uv-env.sh

# Install Python dependencies
print_status "Installing Python dependencies..."
uv sync
print_success "Dependencies installed"

# Set up pre-commit hooks (optional)
if command -v pre-commit >/dev/null 2>&1; then
    print_status "Setting up pre-commit hooks..."
    pre-commit install
    print_success "Pre-commit hooks configured"
else
    print_warning "pre-commit not found. Install with: pip install pre-commit"
fi

# Check GPU availability
print_status "Checking GPU availability..."
if python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"; then
    print_success "GPU support detected"
else
    print_warning "GPU not available or PyTorch not properly installed"
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    print_status "Creating .env file from template..."
    cp .env.example .env 2>/dev/null || echo "# Add your API keys here" > .env
    print_warning "Please edit .env file with your API keys"
fi

print_success "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run 'python cli_menu.py' to start the interactive menu"
echo "3. For development, use VS Code with the provided settings"
echo ""
echo "For external services (Elasticsearch, Redis):"
echo "  Run: python cli_menu.py → Setup & Infrastructure"
echo ""
echo "For testing:"
echo "  Run: uv run pytest tests/"
