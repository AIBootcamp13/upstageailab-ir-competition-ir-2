# Poetry Python Usage Protocol

## 🎯 Overview

This project uses **Poetry** for dependency management and virtual environment handling. **ALWAYS** use `poetry run python` instead of bare `python` commands to ensure:

1. ✅ Correct virtual environment activation
2. ✅ Proper dependency isolation
3. ✅ Consistent package versions
4. ✅ Reproducible execution

## 🚀 Quick Start

### Running Python Scripts
```bash
# ✅ CORRECT - Use Poetry
poetry run python script.py

# ❌ WRONG - Don't use bare python
python script.py
```

### Running Python Modules
```bash
# ✅ CORRECT
poetry run python -m module_name

# ❌ WRONG
python -m module_name
```

### Interactive Python
```bash
# ✅ CORRECT
poetry run python

# ❌ WRONG
python
```

## 🛠️ VS Code Integration

The project is configured to automatically use Poetry:

- **Python Interpreter**: Automatically set to Poetry virtual environment
- **Terminal Activation**: Poetry environment activates automatically
- **Run/Debug**: Uses Poetry Python by default

### VS Code Settings Applied

```json
{
  "python.defaultInterpreterPath": "${env:HOME}/.cache/pypoetry/virtualenvs/information-retrieval-rag-Sdp3gFZr-py3.10/bin/python",
  "python.terminal.activateEnvironment": true,
  "python.terminal.activateEnvInCurrentTerminal": true,
  "python.terminal.launchArgs": ["-m", "poetry", "run", "python"],
  "poetry.terminal.activateEnvironment": true
}
```

## 📊 Confidence Score Logging

The system now includes comprehensive confidence score logging with rich formatting:

### Features
- 🎨 **Color-coded confidence levels**: 🟢 High (≥0.8), 🟡 Medium (≥0.5), 🔴 Low (<0.5), ❌ Zero/Error
- 📊 **Visual progress bars**: 20-character confidence bars
- 🕒 **Timestamps**: Every log entry includes execution time
- 📝 **Detailed reasoning**: Explains why confidence scores were assigned
- 🔄 **Fallback tracking**: Logs when techniques fall back to alternatives
- ⚠️ **Error handling**: Special logging for confidence errors

### Example Output
```
════════════════════════════════════════════════════════════════════════════════
🎯 CONFIDENCE SCORE | 14:31:18
────────────────────────────────────────────────────────────────────────────────
Technique: REWRITING
Query: What is machine learning?
Confidence: 🟢 [████████████████░░░░] 0.80
Reasoning: Standard query rewriting applied
Context: enhanced_query: Define machine learning.
════════════════════════════════════════════════════════════════════════════════
```

### Testing the System

Run the confidence logging demonstration:

```bash
poetry run python test_confidence_logging.py
```

## 🔧 Development Workflow

### 1. Install Dependencies
```bash
poetry install
```

### 2. Activate Environment
```bash
poetry shell  # Optional - VS Code does this automatically
```

### 3. Run Scripts
```bash
poetry run python your_script.py
```

### 4. Add Dependencies
```bash
poetry add package_name
```

### 5. Update Dependencies
```bash
poetry update
```

## 📋 Protocol Summary

| Action | Command | Notes |
|--------|---------|-------|
| Run script | `poetry run python script.py` | Always use this |
| Run module | `poetry run python -m module` | Never use bare python |
| Interactive | `poetry run python` | For debugging/testing |
| Install deps | `poetry install` | One-time setup |
| Add package | `poetry add package` | Updates pyproject.toml |
| Update deps | `poetry update` | Updates all packages |

## ⚠️ Important Notes

1. **Never use bare `python`** - Always prefix with `poetry run`
2. **VS Code handles activation** - No need to manually activate environments
3. **Dependencies are isolated** - Poetry ensures clean, reproducible environments
4. **Logging is automatic** - Confidence scores are logged for all enhancement techniques

## 🐛 Troubleshooting

### Poetry Not Found
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add to PATH (usually ~/.local/bin)
export PATH="$HOME/.local/bin:$PATH"
```

### Environment Issues
```bash
# Check Poetry environment
poetry env info

# Recreate environment if needed
poetry env remove --all
poetry install
```

### VS Code Not Using Poetry
- Check `.vscode/settings.json` has correct interpreter path
- Reload VS Code window: `Ctrl+Shift+P` → "Developer: Reload Window"
- Verify Poetry virtual environment exists: `poetry env list`