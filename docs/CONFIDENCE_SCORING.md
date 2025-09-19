# Confidence Scoring Documentation

## Overview

Confidence scores in the Query Enhancement Manager represent the system's assessment of the reliability and effectiveness of applied query enhancement techniques. These scores help determine whether enhanced queries should be used for retrieval or if fallback strategies should be employed.

## Confidence Score Scale

Confidence scores range from 0.0 to 1.0, with higher values indicating greater confidence in the enhancement quality:

- **0.0**: Complete failure or no enhancement applied
- **0.5**: Moderate confidence (fallback scenarios)
- **0.6**: Moderate confidence (partial success, translation, HyDE fallback)
- **0.7**: Good confidence (step-back prompting)
- **0.8**: High confidence (query rewriting)
- **0.9**: Very high confidence (decomposition, successful HyDE)

## Default Confidence Scores

The system uses predefined confidence scores for each enhancement technique:

```python
DEFAULT_CONFIDENCE_SCORES = {
    'rewriting': 0.8,      # High confidence - generally reliable
    'step_back': 0.7,      # Good confidence - effective for ambiguous queries
    'decomposition': 0.9,  # Very high confidence - effective for complex queries
    'hyde': 0.9,           # Very high confidence - uses embeddings for retrieval
    'translation': 0.6     # Moderate confidence - depends on language detection
}
```

## Technique-Specific Confidence Determination

### Query Rewriting (`_apply_rewriting`)
**Confidence**: 0.8 (fixed)
**Rationale**: Query rewriting is considered generally reliable for most query types as it expands queries with synonyms and related terms.

### Step-Back Prompting (`_apply_step_back`)
**Confidence**: 0.7 (fixed)
**Rationale**: Step-back prompting is effective for ambiguous queries but depends on the query's inherent ambiguity level.

### Query Decomposition (`_apply_decomposition`)
**Confidence**:
- 0.9 if decomposition is recommended
- 0.0 if decomposition is not needed
**Rationale**: Decomposition is highly effective for complex, multi-part queries but only when actually needed.

### HyDE (Hypothetical Document Embeddings) (`_apply_hyde`)
**Confidence** (contextual):
- **0.9**: Successful retrieval with results
- **0.6**: Generated hypothetical answer but no retrieval results
- **0.0**: Complete failure or error
**Rationale**: HyDE confidence depends on whether the hypothetical answer successfully retrieves relevant documents.

### Query Translation (`_apply_translation`)
**Confidence**: 0.6 (fixed)
**Rationale**: Translation effectiveness depends on language detection accuracy and translation quality.

## Special Cases

### Fallback Classification
**Confidence**: 0.5
**Context**: Used when the strategic classifier is disabled or fails
**Rationale**: Moderate confidence for basic fallback scenarios

### Error Conditions
**Confidence**: 0.0
**Context**: When techniques fail to execute properly
**Rationale**: Zero confidence indicates the enhancement should not be used

## Confidence Score Usage

### 1. Decision Making
Confidence scores help determine whether to:
- Use the enhanced query for retrieval
- Fall back to alternative techniques
- Skip enhancement entirely

### 2. Fallback Logic
The system implements intelligent fallback based on confidence:

```python
# HyDE fallback example
if technique == 'hyde':
    enhanced_query = result.get('enhanced_query', query)
    retrieval_results = result.get('retrieval_results', [])

    # Consider HyDE ineffective if no results or same as original
    ineffective = (enhanced_query.strip() == query.strip()) or (not retrieval_results)
    if ineffective:
        fallback = self._apply_rewriting(query)
        return fallback
```

### 3. Result Metadata
Confidence scores are included in enhancement results:

```python
{
    'enhanced': True,
    'original_query': 'What is machine learning?',
    'enhanced_query': 'What is machine learning and how does it work?',
    'technique_used': 'rewriting',
    'confidence': 0.8
}
```

### 4. Debugging and Monitoring
Confidence scores help:
- Identify which techniques are most reliable
- Monitor system performance
- Debug enhancement failures
- Optimize technique selection

## Confidence Score Interpretation

### High Confidence (0.8-0.9)
- **Use Case**: Primary retrieval queries
- **Action**: Use enhanced query directly
- **Techniques**: Rewriting, Decomposition (when applicable), Successful HyDE

### Moderate Confidence (0.5-0.7)
- **Use Case**: Supplementary enhancement
- **Action**: Consider enhancement but monitor results
- **Techniques**: Step-back, Translation, HyDE fallback

### Low Confidence (0.0-0.4)
- **Use Case**: Fallback scenarios
- **Action**: Skip enhancement or use original query
- **Techniques**: Failed techniques, error conditions

## Configuration

Confidence scores can be customized by modifying the `DEFAULT_CONFIDENCE_SCORES` dictionary in `constants.py`:

```python
# Example customization
DEFAULT_CONFIDENCE_SCORES = {
    'rewriting': 0.9,      # Increased confidence
    'translation': 0.8,    # Increased confidence for better translation models
    'hyde': 0.8,           # Slightly reduced for conservative approach
}
```

## Best Practices

1. **Monitor Confidence Trends**: Track confidence scores over time to identify technique performance
2. **Adjust Thresholds**: Modify confidence thresholds based on domain-specific requirements
3. **Fallback Strategies**: Always implement fallback logic for low-confidence scenarios
4. **A/B Testing**: Test different confidence thresholds to optimize retrieval quality

## Implementation Notes

- Confidence scores are empirically determined based on technique reliability
- Scores are static defaults but can be made dynamic based on historical performance
- The system prioritizes reliability over aggressive enhancement
- Confidence scores directly impact the user experience by determining query quality</content>
<parameter name="filePath">/home/wb2x/workspace/information_retrieval_rag/src/ir_core/query_enhancement/CONFIDENCE_SCORES.md