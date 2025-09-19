# Translation System Integration Guide

This guide explains how to integrate the uv-powered translation system into your RAG validation pipeline.

## Overview

The translation system provides:
- **Dependency Isolation**: Uses uv to avoid conflicts with Poetry-managed dependencies
- **Redis Caching**: Caches translations to avoid redundant API calls
- **Batch Processing**: Efficiently processes large validation datasets
- **Fallback Support**: Gracefully handles translation failures

## Architecture

```
Validation Data (Korean) → Translation Cache → UV Translation → Validation Pipeline
                                      ↓
                                 Redis Cache
```

## Quick Start

### 1. Translate Validation Data

```bash
# Simple translation with caching
./scripts/translation/translate_validation.sh

# Or use the integration script directly
python scripts/translation/integrate_translation.py \
    --input data/validation_balanced.jsonl \
    --output data/validation_balanced_en.jsonl \
    --cache
```

### 2. Run Validation with Translation

```bash
# Run validation with automatic translation
python scripts/translation/validate_with_translation.py --translate

# Run validation with existing translated data
python scripts/translation/validate_with_translation.py
```

## Configuration

### Settings Configuration

Add to `conf/settings.yaml`:

```yaml
# --- Translation Configuration ---
translation:
  enabled: false  # Enable translation for validation queries
  cache_enabled: true  # Use Redis caching for translations
  source_lang: "ko"  # Source language
  target_lang: "en"  # Target language
  cache_ttl_days: 30  # Cache TTL in days
  batch_size: 100  # Batch size for processing
  use_uv: true  # Use uv-powered translation system
```

### Environment Variables

```bash
# Redis configuration (uses existing REDIS_URL)
export REDIS_URL="redis://localhost:6379/0"

# Translation settings
export TRANSLATION_ENABLED=true
export TRANSLATION_CACHE_ENABLED=true
```

## Caching Mechanism

### How It Works

1. **Cache Key Generation**: Uses MD5 hash of original text
   ```
   Key: translation:ko_en:<md5_hash>
   Value: translated_text
   TTL: 30 days
   ```

2. **Cache Lookup**: Checks Redis before making API calls
3. **Cache Storage**: Stores successful translations for reuse

### Cache Benefits

- **Cost Savings**: Reduces Google Translate API calls
- **Performance**: Faster validation runs for repeated queries
- **Reliability**: Handles API rate limits gracefully

### Cache Management

```bash
# Check cache status
redis-cli keys "translation:*" | wc -l

# Clear translation cache
redis-cli del $(redis-cli keys "translation:*")

# Monitor cache hits/misses
redis-cli monitor | grep translation
```

## Integration Examples

### 1. Manual Integration

```python
from scripts.translation.integrate_translation import translate_with_caching

# Translate validation data
success = translate_with_caching(
    input_file="data/validation_balanced.jsonl",
    output_file="data/validation_balanced_en.jsonl",
    use_cache=True
)

if success:
    print("Translation completed!")
```

### 2. Pipeline Integration

```python
# In your validation script
import os
from pathlib import Path

def get_validation_file():
    """Get appropriate validation file (translated or original)"""
    translated_file = "data/validation_balanced_en.jsonl"
    original_file = "data/validation_balanced.jsonl"

    # Check if translation is enabled
    use_translation = os.getenv("TRANSLATION_ENABLED", "false").lower() == "true"

    if use_translation and Path(translated_file).exists():
        return translated_file
    else:
        return original_file

# Use in Hydra config
validation_file = get_validation_file()
# Pass to validation pipeline...
```

### 3. Hydra Configuration

```yaml
# conf/config.yaml
defaults:
  - data: science_qa
  - _self_

# Override validation path based on translation setting
data:
  validation_path: ${env:VALIDATION_FILE, "data/validation_balanced.jsonl"}
```

## Data Format

### Input Format (Korean)
```jsonl
{
  "eval_id": "balanced_mmlu_biology_011",
  "msg": [
    {
      "role": "user",
      "content": "이 생물의 특징은 무엇인가요? ..."
    }
  ],
  "ground_truth_doc_id": "9758eb22-6576-432a-8846-2ba035c2ccb9"
}
```

### Output Format (Translated)
```jsonl
{
  "eval_id": "balanced_mmlu_biology_011",
  "msg": [
    {
      "role": "user",
      "content": "What are the characteristics of this organism? ...",
      "content_original": "이 생물의 특징은 무엇인가요? ...",
      "translation_status": "success"
    }
  ],
  "ground_truth_doc_id": "9758eb22-6576-432a-8846-2ba035c2ccb9",
  "translation_status": "success"
}
```

## Troubleshooting

### Common Issues

1. **UV Not Found**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Redis Connection Failed**
   ```bash
   # Start Redis
   redis-server

   # Check connection
   redis-cli ping
   ```

3. **Translation API Errors**
   - Check Google Translate API limits
   - Verify network connectivity
   - Review error logs in Redis

4. **Import Errors**
   ```bash
   # Ensure src is in PYTHONPATH
   export PYTHONPATH=src:$PYTHONPATH
   ```

### Performance Tuning

```yaml
# conf/settings.yaml
translation:
  batch_size: 50  # Reduce for slower networks
  cache_ttl_days: 7  # Shorter TTL for frequently changing content
```

### Monitoring

```bash
# Check translation cache statistics
redis-cli info | grep keys

# Monitor translation script logs
tail -f logs/translation.log

# Check system resources during translation
htop
```

## Advanced Usage

### Custom Translation Sources

```python
# Extend TranslationCache for other translation services
class CustomTranslationCache(TranslationCache):
    def translate_text(self, text: str) -> str:
        # Implement custom translation logic
        pass
```

### Batch Translation

```bash
# Process large files in chunks
split -l 1000 data/validation_balanced.jsonl chunk_

# Translate each chunk
for chunk in chunk_*; do
    ./scripts/translation/translate_validation.sh "$chunk" "${chunk}_en"
done

# Combine results
cat chunk_*_en > data/validation_balanced_en.jsonl
```

### Integration with CI/CD

```yaml
# .github/workflows/validate.yml
- name: Translate Validation Data
  run: |
    ./scripts/translation/translate_validation.sh

- name: Run Validation
  run: |
    python scripts/evaluation/validate_retrieval.py
```

## FAQ

**Q: Can I use this for other languages?**
A: Yes, modify `source_lang` and `target_lang` in settings.

**Q: What happens if Redis is down?**
A: The system falls back to direct translation without caching.

**Q: How do I clear the cache?**
A: `redis-cli keys "translation:*" | xargs redis-cli del`

**Q: Can I disable caching?**
A: Use `--no-cache` flag or set `cache_enabled: false` in settings.

**Q: What's the performance impact?**
A: Cached translations are ~100x faster than API calls.