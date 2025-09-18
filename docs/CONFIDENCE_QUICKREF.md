# Confidence Scoring Quick Reference

## Scale & Interpretation
- **0.0**: Complete failure, skip enhancement
- **0.5**: Fallback scenarios, moderate reliability
- **0.6**: Partial success (HyDE fallback, translation)
- **0.7**: Good confidence (step-back prompting)
- **0.8**: High confidence (query rewriting)
- **0.9**: Very high confidence (decomposition, successful HyDE)

## Technique Confidence Matrix

| Technique | Default | When Applied | Rationale |
|-----------|---------|--------------|-----------|
| **Rewriting** | 0.8 | Always | Generally reliable for most queries |
| **Step-back** | 0.7 | Ambiguous queries | Effective but context-dependent |
| **Decomposition** | 0.9/0.0 | Complex queries only | Highly effective when needed |
| **HyDE** | 0.9/0.6/0.0 | Questions/short queries | Depends on retrieval success |
| **Translation** | 0.6 | Non-English queries | Language detection dependent |

## Usage Patterns

### High Confidence (0.8-0.9)
```python
# Use directly for retrieval
enhanced_query = result['enhanced_query']
perform_retrieval(enhanced_query)
```

### Moderate Confidence (0.5-0.7)
```python
# Consider enhancement but monitor
if confidence > 0.7:
    use_enhanced_query(result)
else:
    use_original_query_with_fallback(result)
```

### Low Confidence (0.0-0.4)
```python
# Skip enhancement entirely
return original_query
```

## Fallback Logic Examples

### HyDE Fallback
```python
if confidence < 0.7 or not retrieval_results:
    return apply_rewriting(query)
```

### General Fallback
```python
if confidence < 0.5:
    return {
        'enhanced': False,
        'reason': 'Low confidence in enhancement'
    }
```

## Configuration
Modify `DEFAULT_CONFIDENCE_SCORES` in `constants.py` to adjust thresholds based on:
- Domain requirements
- Performance monitoring
- A/B testing results

## Monitoring
Track confidence scores to:
- Identify technique performance trends
- Optimize technique selection
- Debug enhancement failures
- Improve system reliability</content>
<parameter name="filePath">/home/wb2x/workspace/information_retrieval_rag/src/ir_core/query_enhancement/CONFIDENCE_QUICKREF.md