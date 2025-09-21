# Confidence Score Determination Formulas

## Overview

Confidence scores in the Query Enhancement Manager are **empirically determined** rather than calculated using mathematical formulas. They represent the system's assessment of technique reliability based on historical performance and expected effectiveness.

## Confidence Score "Formulas"

### 1. Static Default Scores
```python
DEFAULT_CONFIDENCE_SCORES = {
    'rewriting': 0.8,      # Formula: EMPIRICAL_CONSTANT
    'step_back': 0.7,      # Formula: EMPIRICAL_CONSTANT
    'decomposition': 0.9,  # Formula: EMPIRICAL_CONSTANT
    'hyde': 0.9,           # Formula: EMPIRICAL_CONSTANT
    'translation': 0.6     # Formula: EMPIRICAL_CONSTANT
}
```

### 2. Conditional Confidence Formulas

#### Decomposition Confidence Formula
```python
confidence = DEFAULT_CONFIDENCE_SCORES['decomposition'] if should_decompose else 0.0
# Formula: IF(decomposition_needed, 0.9, 0.0)
```

#### HyDE Confidence Formula
```python
if retrieval_successful:
    confidence = DEFAULT_CONFIDENCE_SCORES['hyde']  # 0.9
elif answer_generated:
    confidence = 0.6  # Moderate confidence
else:
    confidence = 0.0  # No confidence
# Formula: IF(retrieval_success, 0.9, IF(answer_generated, 0.6, 0.0))
```

#### Fallback Classification Formula
```python
confidence = 0.5  # Fixed moderate confidence
# Formula: CONSTANT(0.5)
```

#### Error Condition Formula
```python
confidence = 0.0  # Zero confidence on failure
# Formula: CONSTANT(0.0)
```

## Detailed Formula Breakdown

### Technique-Specific Formulas

#### Query Rewriting
```
confidence = 0.8
Rationale: Generally reliable for most query types
Formula Type: Static Empirical
```

#### Step-Back Prompting
```
confidence = 0.7
Rationale: Effective for ambiguous queries, context-dependent
Formula Type: Static Empirical
```

#### Query Decomposition
```
confidence = IF(should_decompose(query), 0.9, 0.0)
Rationale: Highly effective when needed, ineffective otherwise
Formula Type: Binary Conditional
```

#### HyDE (Hypothetical Document Embeddings)
```
confidence = CASE
    WHEN retrieval_results_exist THEN 0.9
    WHEN answer_generated THEN 0.6
    ELSE 0.0
END
Rationale: Depends on retrieval success and answer generation
Formula Type: Multi-state Conditional
```

#### Query Translation
```
confidence = 0.6
Rationale: Depends on language detection accuracy
Formula Type: Static Empirical
```

## Mathematical Representation

### Binary Conditional Formula
```
f_confidence(condition, true_value, false_value) = true_value if condition else false_value

Example: f_confidence(decomposition_needed, 0.9, 0.0)
```

### Multi-state Conditional Formula
```
f_confidence_hyde(retrieval_success, answer_generated) =
    0.9 if retrieval_success
    0.6 if answer_generated and not retrieval_success
    0.0 otherwise
```

### Weighted Average (Potential Future Enhancement)
```
confidence = Σ(weight_i × score_i) / Σ(weight_i)

Where:
- weight_i = importance of factor i
- score_i = score for factor i
```

## Implementation Logic

### Current Implementation
```python
def get_confidence_score(technique: str, context: Dict) -> float:
    """Get confidence score for a technique given context."""
    base_score = DEFAULT_CONFIDENCE_SCORES.get(technique, 0.5)

    # Apply conditional logic
    if technique == 'decomposition':
        return base_score if context.get('should_decompose') else 0.0
    elif technique == 'hyde':
        if context.get('retrieval_successful'):
            return base_score
        elif context.get('answer_generated'):
            return 0.6
        else:
            return 0.0
    else:
        return base_score
```

### Fallback Logic Integration
```python
def apply_with_fallback(query: str, technique: str) -> Dict:
    """Apply technique with intelligent fallback based on confidence."""
    result = apply_technique(query, technique)

    if result['confidence'] < 0.7:
        # Low confidence, apply fallback
        fallback_result = apply_technique(query, 'rewriting')
        fallback_result['fallback_from'] = technique
        return fallback_result

    return result
```

## Confidence Score Validation

### Quality Metrics
- **Precision**: How often high confidence leads to good results
- **Recall**: How often good results get high confidence
- **F1-Score**: Balance between precision and recall

### A/B Testing Framework
```python
def test_confidence_threshold(threshold: float) -> Dict:
    """Test different confidence thresholds."""
    results = {
        'threshold': threshold,
        'high_confidence_queries': 0,
        'successful_enhancements': 0,
        'fallback_triggers': 0
    }
    return results
```

## Future Enhancements

### Dynamic Confidence Scoring
```python
def calculate_dynamic_confidence(technique: str, historical_performance: Dict) -> float:
    """Calculate confidence based on historical performance."""
    success_rate = historical_performance.get('success_rate', 0.8)
    recent_performance = historical_performance.get('recent_accuracy', 0.8)

    # Weighted combination
    confidence = (success_rate * 0.7) + (recent_performance * 0.3)
    return min(confidence, 1.0)  # Cap at 1.0
```

### Machine Learning-Based Scoring
```python
def ml_confidence_scorer(query_features: Dict, technique: str) -> float:
    """Use ML model to predict confidence score."""
    features = extract_features(query_features)
    confidence = ml_model.predict(features, technique)
    return confidence
```

## Summary

The confidence scoring system uses a **hybrid approach**:

1. **Empirical Constants**: Pre-determined values based on technique reliability
2. **Conditional Logic**: Context-dependent adjustments
3. **Fallback Mechanisms**: Automatic degradation for low confidence
4. **Extensible Design**: Framework for future dynamic scoring

This approach balances simplicity with effectiveness, providing reliable confidence assessments for query enhancement decisions.</content>
<parameter name="filePath">/home/wb2x/workspace/information_retrieval_rag/docs/CONFIDENCE_FORMULAS.md