# Information Retrieval RAG Setup Guide

This guide covers everything you need to set up and run the Information Retrieval RAG project. We've automated as much as possible, but some manual steps are still required.

## 🚀 Quick Start (Recommended)

### Option 1: Docker (Easiest)
```bash
# Clone and run with Docker
git clone https://github.com/AIBootcamp13/upstageailab-ir-competition-ir-2/tree/main/scripts
cd information-retrieval-rag
docker build -t ir-rag .
docker run -it --gpus all -p 8000:8000 ir-rag
```

### Option 2: Local Development
```bash
# Clone repository
git clone https://github.com/AIBootcamp13/upstageailab-ir-competition-ir-2/tree/main/scripts
cd information-retrieval-rag

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Setup environment
uv sync
source ~/.bashrc  # Reload for aliases

# Run tests
python -m pytest tests/test_utils.py -v

# Start development
python cli_menu.py
```

## 📋 Prerequisites

### Required
- **Python 3.10+** (managed by uv)
- **Git**
- **curl** (for uv installation)

### Optional (but recommended)
- **Docker** (for containerized development)
- **NVIDIA GPU** (for accelerated ML workloads)
- **zsh** (better shell experience)

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential git curl

# macOS
brew install git curl

# Windows (WSL recommended)
# Use Ubuntu WSL and follow Ubuntu instructions
```

## 🛠️ Manual Setup Steps

### 1. Install uv Package Manager
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Verify installation
uv --version
```

### 2. Clone and Setup Repository
```bash
# Clone repository
git clone <repository-url>
cd information-retrieval-rag

# Install dependencies
uv sync

# Reload shell for aliases
source ~/.bashrc
```

### 3. Configure Shell Environment
```bash
# The setup includes automatic aliases:
# - python → uv run python
# - python3 → uv run python3
# - pip → uv pip (but not recommended)

# Optional: Load project-specific aliases for convenience
source scripts/project-aliases.sh

# This adds:
# - menu → uv run cli_menu.py (main project interface)
# - cli → uv run scripts/cli.py (command-line interface)
# - sc → uv run switch_config.py (configuration switcher)
# - sd → uv run switch_data_config.py (data config switcher)

# Test the setup
python --version  # Should show Python 3.10.x
which python      # Should show .venv/bin/python
```

### 4. Configure VS Code (Recommended)
```json
// .vscode/settings.json (already configured)
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": false,
  "terminal.integrated.shell.linux": "/usr/bin/zsh",
  "python.analysis.extraPaths": [".", "src", "scripts"]
}
```

### 5. Optional: Switch to Zsh
```bash
# Install zsh (if not available)
sudo apt install zsh

# Change default shell
sudo chsh -s /usr/bin/zsh $USER

# Or configure VS Code to use zsh
# Add to .vscode/settings.json:
"terminal.integrated.shell.linux": "/usr/bin/zsh"
```

## 🐳 Docker Setup

### Current Dockerfile Issues
The existing Dockerfile has several challenges:
- Large image size (many layers, duplicate installations)
- Complex multi-stage build
- GPU support configuration
- Environment variable management

### Recommended Docker Improvements

#### Option 1: Single-Stage with uv
```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --frozen --no-install-project

# Copy source code
COPY . .

# Install project in editable mode
RUN uv sync --frozen

# Expose ports
EXPOSE 8000

# Default command
CMD ["uv", "run", "python", "cli_menu.py"]
```

#### Option 2: Multi-Stage Optimized
```dockerfile
# Base stage
FROM python:3.10-slim as base

# Install uv
RUN pip install uv
ENV UV_CACHE_DIR=/tmp/uv-cache

# Dependencies stage
FROM base as deps
WORKDIR /deps
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --cache-dir /tmp/uv-cache

# Runtime stage
FROM python:3.10-slim as runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from deps stage
COPY --from=deps /deps/.venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set work directory
WORKDIR /app

# Copy source code
COPY . .

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import ir_core; print('OK')" || exit 1

# Default command
CMD ["python", "cli_menu.py"]
```

### Docker Compose for Development
```yaml
# docker-compose.yml
version: '3.8'

services:
  ir-rag:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
      - /app/.venv  # Don't mount venv for better caching
    environment:
      - CUDA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: sleep infinity  # Keep container running
    profiles: ["dev"]

  # For production
  ir-rag-prod:
    build:
      context: .
      target: runtime
    ports:
      - "8000:8000"
    environment:
      - ENV=production
    profiles: ["prod"]
```

## 🔧 Development Workflow

### Daily Development
```bash
# Activate environment (automatic with aliases)
cd /path/to/project

# Install new dependencies
uv add package-name

# Run tests
python -m pytest tests/ -v

# Run specific tests
python -m pytest tests/test_utils.py -v

# Start development server
python cli_menu.py

# Format code
uv run ruff format .

# Lint code
uv run ruff check . --fix
```

### Code Quality Tools
```bash
# Install development dependencies
uv sync --group dev

# Run all quality checks
uv run pre-commit run --all-files

# Format and lint
uv run black .
uv run isort .
uv run flake8 .
```

## 🚨 Troubleshooting

### Common Issues

#### 1. "uv command not found"
```bash
# Reinstall uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

#### 2. Import errors in tests
```bash
# Ensure PYTHONPATH is set
python -c "import sys; print(sys.path)"
# Should include the project root

# Or run with explicit path
PYTHONPATH=/path/to/project/src uv run python -m pytest tests/
```

#### 3. GPU not available in Docker
```bash
# Check GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# For Docker Compose
docker-compose --profile dev up
```

#### 4. Virtual environment conflicts
```bash
# Deactivate any manual activation
deactivate

# Use uv exclusively
uv run python script.py
```

#### 5. Permission issues with uv cache
```bash
# Clear uv cache
uv cache clean

# Or change cache location
export UV_CACHE_DIR=~/.cache/uv
```

### Environment Variables
```bash
# Disable dynamic Python path (if causing issues)
export SKIP_DYNAMIC_PYTHONPATH=1

# Disable venv prompt
export VIRTUAL_ENV_DISABLE_PROMPT=1

# Set log level
export LOG_LEVEL=DEBUG
```

## 📊 Assessment: What Can/Cannot Be Automated

### ✅ Highly Automatable
- **Dependency installation** (uv sync)
- **Environment setup** (VS Code settings, shell aliases)
- **Basic Docker builds** (single-stage with uv)
- **Code formatting/linting** (pre-commit hooks)
- **Basic testing** (unit tests)

### ⚠️ Partially Automatable
- **GPU setup** (detect but requires manual NVIDIA driver install)
- **Complex Docker builds** (multi-stage, but still needs manual optimization)
- **External services** (Elasticsearch, Redis - can be dockerized but need orchestration)
- **Environment-specific configs** (dev vs prod settings)

### ❌ Difficult to Automate
- **Manual VS Code configuration** (user preferences, extensions)
- **System-level dependencies** (NVIDIA drivers, CUDA)
- **Network/firewall settings** (for external services)
- **User-specific credentials** (API keys, database passwords)
- **Development workflow preferences** (editor choice, shell preference)

### 🎯 Recommended Improvements

#### High Priority
1. **Create setup script**: `setup.sh` that runs all installation steps
2. **Improve Docker setup**: Single-stage build with proper caching
3. **Add health checks**: For Docker containers and services
4. **Environment detection**: Auto-detect GPU, OS, available tools

#### Medium Priority
1. **Git hooks**: Pre-commit for code quality
2. **CI/CD pipeline**: GitHub Actions for automated testing
3. **Documentation**: Auto-generated API docs
4. **Development containers**: VS Code dev container configuration

#### Low Priority
1. **Package templates**: For creating new modules
2. **Deployment scripts**: For different environments
3. **Monitoring**: Basic metrics and logging setup

## 📝 Contributing

### For Contributors
1. Follow the setup guide above
2. Run tests before committing: `uv run pytest tests/`
3. Format code: `uv run black . && uv run isort .`
4. Create feature branch from `main`
5. Submit PR with description

### For Maintainers
1. Update dependencies regularly: `uv lock --upgrade`
2. Test Docker builds: `docker build -t ir-rag .`
3. Update documentation when APIs change
4. Monitor CI/CD pipeline

---

**Need help?** Check the troubleshooting section or open an issue with your setup details.
