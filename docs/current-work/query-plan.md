# Query Enhancement Techniques Implementation Plan

## Overview

This document provides a concrete implementation plan for the five query enhancement techniques outlined in `strategies-query.md`. Each technique includes detailed implementation steps, code examples, integration points with the existing RAG system, and testing strategies.

## Current Implementation Status ✅

### ✅ COMPLETED: Core Infrastructure (Phase 1-5)
- **LLM Client Abstraction**: Implemented unified interface supporting OpenAI and Ollama/Qwen models
- **Query Rewriting**: Fully implemented with multi-provider support (OpenAI + Qwen)
- **Step-Back Prompting**: Implemented with LLM client integration
- **Query Decomposition**: Implemented with sub-query generation and result aggregation
- **HyDE (Hypothetical Document Embeddings)**: Implemented with hypothetical answer generation
- **Query Translation**: Framework implemented (needs local model integration)

### ✅ COMPLETED: Pipeline Integration
- **RAG Pipeline**: Updated to support query enhancement manager
- **Configuration**: Hydra-based configuration with model selection
- **Validation**: Query rewriting confirmed working in logs (despite metric reporting issues)
- **Multi-Provider Support**: Seamless switching between OpenAI GPT and Ollama Qwen models

### 📊 Current Performance Results
- **MAP Score**: 0.063 (slight improvement from 0.059 baseline)
- **Rewrite Rate**: 0.0% (metric issue - logs show active rewriting)
- **Retrieval Success Rate**: 10.4% (improvement from 8.5%)
- **Query Enhancement**: Confirmed working in debug logs

### 🔍 Known Issues & Next Steps
- **Metric Detection**: Validation script reports 0.0% rewrite rate despite active rewriting in logs
- **Domain Performance**: High error rates in biology (89.4%), physics (89.1%), chemistry (87.5%)
- **Complex Queries**: 33 failures with multi-concept questions
- **False Negatives**: 32 instances of incorrect negative results

### 🎯 Immediate Priorities
1. **Fix Rewrite Rate Detection**: Investigate validation metric calculation
2. **Improve Domain-Specific Performance**: Address scientific terminology issues
3. **Handle Complex Queries**: Implement better decomposition for multi-concept questions
4. **Reduce False Negatives**: Improve document matching and retrieval logic

## Prerequisites

### Dependencies
- OpenAI API access (already configured in the project)
- Ollama with Qwen2:7B model (locally running)
- Additional libraries to add to `pyproject.toml`:
  - `googletrans==4.0.0rc1` (for query translation)
  - `nltk` (for text processing, if needed)

### Configuration
Add the following to `conf/settings.yaml`:
```yaml
query_enhancement:
  enabled: true
  default_technique: "rewriting"  # Options: rewriting, step_back, decomposition, hyde, translation
  openai_model: "gpt-3.5-turbo"
  ollama_model: "qwen2:7b"
  max_tokens: 500
  temperature: 0.3
```

## 1. Query Rewriting & Expansion

### Implementation Steps

1. **Create Query Rewriter Module**
   - Location: `src/ir_core/query_enhancement/rewriter.py`
   - Implement LLM-based query expansion using OpenAI

2. **Integration Points**
   - Modify `src/ir_core/orchestration/pipeline.py` to call rewriter before retrieval
   - Add configuration option to enable/disable rewriting

3. **Code Structure**
```python
class QueryRewriter:
    def __init__(self, openai_client, config):
        self.client = openai_client
        self.config = config

    def rewrite_query(self, original_query: str) -> str:
        """Rewrite and expand the query for better retrieval"""
        prompt = f"""
        Transform this query into an effective search query by:
        1. Extracting core concepts
        2. Adding relevant synonyms and related terms
        3. Making it more specific for document retrieval

        Original query: {original_query}

        Provide only the rewritten query, no explanation.
        """

        response = self.client.chat.completions.create(
            model=self.config.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )

        return response.choices[0].message.content.strip()
```

### Testing Strategy
- Unit tests for the rewriter class
- Integration tests comparing retrieval quality with/without rewriting
- A/B testing with sample queries from `data/eval.jsonl`

## 2. Step-Back Prompting

### Implementation Steps

1. **Create Step-Back Module**
   - Location: `src/ir_core/query_enhancement/step_back.py`
   - Implement abstraction layer for vague queries

2. **Integration Points**
   - Add to pipeline as alternative to direct rewriting
   - Use when query ambiguity is detected

3. **Code Structure**
```python
class StepBackPrompting:
    def __init__(self, openai_client, config):
        self.client = openai_client
        self.config = config

    def step_back(self, original_query: str) -> str:
        """Take a step back to find the underlying concept"""
        prompt = f"""
        The user asked: "{original_query}"

        What is the general, underlying concept being asked?
        Provide a clear, abstract description of what information they're really seeking.

        Respond with just the abstracted concept, no explanation.
        """

        response = self.client.chat.completions.create(
            model=self.config.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )

        abstract_concept = response.choices[0].message.content.strip()

        # Use the abstract concept to create a search query
        return self._concept_to_search_query(abstract_concept)

    def _concept_to_search_query(self, concept: str) -> str:
        """Convert abstract concept to searchable keywords"""
        prompt = f"""
        Convert this abstract concept into specific search keywords:
        {concept}

        Provide keywords separated by commas, focused on terms likely to appear in documents.
        """

        response = self.client.chat.completions.create(
            model=self.config.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )

        return response.choices[0].message.content.strip()
```

### Testing Strategy
- Test with ambiguous queries from eval data
- Measure improvement in retrieval relevance scores
- Validate that step-back doesn't over-abstract clear queries

## 3. Query Decomposition

### Implementation Steps

1. **Create Decomposition Module**
   - Location: `src/ir_core/query_enhancement/decomposer.py`
   - Break complex queries into sub-queries

2. **Integration Points**
   - Modify retrieval to handle multiple sub-queries
   - Aggregate results from multiple searches

3. **Code Structure**
```python
class QueryDecomposer:
    def __init__(self, openai_client, config):
        self.client = openai_client
        self.config = config

    def decompose_query(self, original_query: str) -> List[str]:
        """Break complex query into simpler sub-queries"""
        prompt = f"""
        Break this complex query into 2-4 simpler, independent sub-queries:

        Original query: {original_query}

        Provide each sub-query on a new line.
        Focus on questions that can be answered independently.
        """

        response = self.client.chat.completions.create(
            model=self.config.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )

        sub_queries = response.choices[0].message.content.strip().split('\n')
        return [q.strip() for q in sub_queries if q.strip()]

    def aggregate_results(self, sub_query_results: List[List[Dict]]) -> List[Dict]:
        """Aggregate and deduplicate results from multiple sub-queries"""
        all_results = []
        seen_ids = set()

        for results in sub_query_results:
            for result in results:
                doc_id = result.get('document_id') or result.get('id')
                if doc_id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(doc_id)

        # Sort by relevance score
        return sorted(all_results, key=lambda x: x.get('score', 0), reverse=True)
```

### Testing Strategy
- Test with complex multi-part questions
- Validate that decomposition improves coverage without sacrificing precision
- Measure performance impact of multiple retrieval calls

## 4. Hypothetical Document Embeddings (HyDE)

### Implementation Steps

1. **Create HyDE Module**
   - Location: `src/ir_core/query_enhancement/hyde.py`
   - Generate hypothetical answers and use their embeddings

2. **Integration Points**
   - Integrate with existing embedding system
   - Use hypothetical answer embedding for retrieval

3. **Code Structure**
```python
class HyDE:
    def __init__(self, openai_client, embedding_model, config):
        self.client = openai_client
        self.embedding_model = embedding_model
        self.config = config

    def generate_hypothetical_answer(self, query: str) -> str:
        """Generate a detailed hypothetical answer to the query"""
        prompt = f"""
        Provide a detailed, comprehensive answer to this question as if you were writing a reference document:

        Question: {query}

        Write a detailed answer that would appear in an encyclopedia or textbook.
        Include specific facts, examples, and explanations.
        """

        response = self.client.chat.completions.create(
            model=self.config.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens * 2,  # Allow longer responses
            temperature=self.config.temperature
        )

        return response.choices[0].message.content.strip()

    def get_hyde_embedding(self, query: str) -> np.ndarray:
        """Get embedding of hypothetical answer"""
        hypothetical_answer = self.generate_hypothetical_answer(query)
        return self.embedding_model.encode(hypothetical_answer)

    def retrieve_with_hyde(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve documents using HyDE embedding"""
        hyde_embedding = self.get_hyde_embedding(query)

        # Use existing retrieval system with HyDE embedding
        return self.retrieval_system.search_by_embedding(hyde_embedding, top_k=top_k)
```

### Testing Strategy
- Compare retrieval quality using original query vs HyDE embedding
- Test with short, keyword-poor queries
- Measure embedding similarity between hypothetical answers and actual relevant documents

## 5. Query Translation

### Implementation Steps

1. **Create Translation Module**
   - Location: `src/ir_core/query_enhancement/translator.py`
   - Translate queries to English for better retrieval

2. **Integration Points**
   - Add as preprocessing step
   - Optionally search in both languages and merge results

3. **Code Structure**
```python
from googletrans import Translator

class QueryTranslator:
    def __init__(self, config):
        self.translator = Translator()
        self.config = config

    def translate_to_english(self, query: str) -> str:
        """Translate query to English"""
        try:
            translation = self.translator.translate(query, src='auto', dest='en')
            return translation.text
        except Exception as e:
            # Fallback to original query if translation fails
            return query

    def bilingual_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search in both original language and English"""
        original_results = self.retrieval_system.search(query, top_k=top_k//2)
        english_query = self.translate_to_english(query)
        english_results = self.retrieval_system.search(english_query, top_k=top_k//2)

        # Merge and deduplicate results
        combined_results = original_results + english_results
        seen_ids = set()
        unique_results = []

        for result in combined_results:
            doc_id = result.get('document_id') or result.get('id')
            if doc_id not in seen_ids:
                unique_results.append(result)
                seen_ids.add(doc_id)

        return sorted(unique_results, key=lambda x: x.get('score', 0), reverse=True)[:top_k]
```

### Testing Strategy
- Test with non-English queries from eval data
- Compare retrieval quality in original vs translated queries
- Validate translation accuracy for technical/scientific terms

## Integration with Existing Pipeline

### Pipeline Modification
Modify `src/ir_core/orchestration/pipeline.py`:

```python
class EnhancedRetrievalPipeline:
    def __init__(self, config):
        self.config = config
        self.query_enhancer = QueryEnhancementManager(config)

    def process_query(self, query: str) -> Dict:
        """Enhanced query processing with multiple techniques"""
        # Detect query characteristics
        query_type = self._detect_query_type(query)

        # Apply appropriate enhancement technique
        enhanced_query = self.query_enhancer.enhance_query(query, query_type)

        # Proceed with retrieval using enhanced query
        return self.retrieval_system.search(enhanced_query)

    def _detect_query_type(self, query: str) -> str:
        """Detect which enhancement technique to use"""
        # Simple heuristics for technique selection
        if len(query.split()) < 5:
            return "hyde"  # Short queries benefit from HyDE
        elif "비교" in query or "차이" in query:
            return "decomposition"  # Comparison queries need decomposition
        elif "가치" in query or "의미" in query:
            return "step_back"  # Ambiguous terms need step-back
        else:
            return "rewriting"  # Default to rewriting
```

### Configuration Management
Add to `conf/pipeline/enhancement.yaml`:
```yaml
defaults:
  - _self_

query_enhancement:
  enabled: true
  techniques:
    rewriting:
      enabled: true
      priority: 1
    step_back:
      enabled: true
      priority: 2
    decomposition:
      enabled: true
      priority: 3
    hyde:
      enabled: true
      priority: 4
    translation:
      enabled: true
      priority: 5

  openai:
    model: "gpt-3.5-turbo"
    max_tokens: 500
    temperature: 0.3

  translation:
    fallback_on_error: true
    bilingual_search: true
```

## Testing and Validation

### Unit Tests
Create `tests/test_query_enhancement.py`:
```python
import pytest
from ir_core.query_enhancement import QueryRewriter, StepBackPrompting, QueryDecomposer, HyDE, QueryTranslator

class TestQueryEnhancement:
    def test_query_rewriter(self, mock_openai_client):
        rewriter = QueryRewriter(mock_openai_client, config)
        result = rewriter.rewrite_query("나무 분류 방법")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_step_back_prompting(self, mock_openai_client):
        step_back = StepBackPrompting(mock_openai_client, config)
        result = step_back.step_back("통학 버스 가치")
        assert "가치" not in result  # Should abstract away ambiguous terms

    def test_query_decomposition(self, mock_openai_client):
        decomposer = QueryDecomposer(mock_openai_client, config)
        sub_queries = decomposer.decompose_query("한국과 일본 교육비 비교")
        assert len(sub_queries) >= 2
        assert all(isinstance(q, str) for q in sub_queries)
```

### Integration Tests
Create `tests/integration/test_enhanced_pipeline.py`:
```python
def test_enhanced_pipeline_improvement():
    """Test that enhanced pipeline improves retrieval quality"""
    original_pipeline = RetrievalPipeline(config)
    enhanced_pipeline = EnhancedRetrievalPipeline(config)

    test_queries = load_test_queries("data/eval.jsonl")

    improvements = []
    for query in test_queries[:10]:  # Test subset
        original_results = original_pipeline.process_query(query)
        enhanced_results = enhanced_pipeline.process_query(query)

        # Calculate relevance improvement
        original_relevance = calculate_relevance_score(original_results)
        enhanced_relevance = calculate_relevance_score(enhanced_results)

        improvements.append(enhanced_relevance - original_relevance)

    avg_improvement = sum(improvements) / len(improvements)
    assert avg_improvement > 0, f"No improvement detected: {avg_improvement}"
```

### Performance Benchmarks
Create `scripts/evaluation/benchmark_enhancement.py`:
```python
def benchmark_enhancement_techniques():
    """Benchmark performance and quality of each technique"""
    techniques = ["original", "rewriting", "step_back", "decomposition", "hyde", "translation"]

    results = {}
    for technique in techniques:
        start_time = time.time()
        quality_score = evaluate_technique(technique)
        execution_time = time.time() - start_time

        results[technique] = {
            "quality_score": quality_score,
            "execution_time": execution_time,
            "efficiency": quality_score / execution_time
        }

    return results
```

## Deployment and Monitoring

### Feature Flags
Use feature flags to gradually roll out enhancements:
```python
# In pipeline configuration
feature_flags:
  query_enhancement: true
  hyde_enabled: false  # Roll out gradually due to API costs
  translation_enabled: true
```

### Monitoring
Add metrics to track enhancement effectiveness:
- Query enhancement success rate
- Average improvement in retrieval quality
- API usage and costs
- Processing time overhead

### Rollback Strategy
- Keep original pipeline as fallback
- A/B testing framework for comparing techniques
- Automatic rollback if quality metrics degrade

## Next Steps

### ✅ COMPLETED PHASES
1. **Phase 1**: ✅ Query Rewriting (implemented, working with Qwen)
2. **Phase 2**: ✅ Step-Back Prompting (implemented with LLM client)
3. **Phase 3**: ✅ Query Decomposition (implemented with sub-query generation)
4. **Phase 4**: ✅ HyDE (implemented with hypothetical answer generation)
5. **Phase 5**: 🔄 Query Translation (framework ready, needs local model integration)

### 🚀 IMMEDIATE NEXT STEPS
1. **Fix Validation Metrics**: Debug why rewrite rate shows 0.0% despite working logs
2. **Domain-Specific Optimization**: Improve performance for scientific domains (biology, physics, chemistry)
3. **Complex Query Handling**: Enhance decomposition for multi-concept questions
4. **Document Translation**: Implement local translation using open-source models
5. **Language Strategy**: Evaluate English-only pipeline vs Korean-English hybrid

### 📈 Performance Improvement Targets
- **Current MAP**: 0.063 → **Target**: 0.70
- **Current Rewrite Rate**: 0.0% (fix detection) → **Target**: >50%
- **Domain Error Rates**: 87-89% → **Target**: <20%

Each phase should include:
- Implementation of the technique
- Unit and integration tests
- Performance benchmarking
- Gradual rollout with monitoring
- Documentation updates

## Language Strategy & Translation Considerations

### Current Language Dynamics
- **Source Documents**: Korean language with Korean embeddings (snunlp/KR-SBERT-V40K-klueNLI-augSTS)
- **Query Enhancement**: Currently using English prompts with Qwen model
- **Evaluation**: Korean queries against Korean documents
- **Submissions**: Must be in Korean format

### 🔍 Language Mismatch Analysis

#### Query Rewriting in English
**Potential Issues:**
- **Embedding Mismatch**: English-enhanced queries vs Korean document embeddings
- **Semantic Gap**: Loss of Korean-specific scientific terminology nuances
- **Retrieval Degradation**: Cross-language similarity may be suboptimal

**Current Evidence:**
- Query rewriting confirmed working in logs
- Slight MAP improvement (0.059 → 0.063)
- High domain-specific error rates suggest terminology issues

#### Translation Strategy Options

**Option 1: Korean-Only Pipeline**
- Keep all processing in Korean
- Use Korean prompts for enhancement
- Maintain language consistency
- **Pros**: No translation loss, better terminology handling
- **Cons**: Limited by Korean LLM capabilities, potential hallucinations

**Option 2: English Translation Pipeline**
- Translate documents to English using open-source models
- Use English for all processing
- Translate final answers back to Korean
- **Pros**: Better LLM performance, more stable, access to English resources
- **Cons**: Translation quality risks, final submission translation required

**Option 3: Hybrid Approach**
- Bilingual retrieval (search both languages)
- Language-specific enhancement techniques
- **Pros**: Best of both worlds
- **Cons**: Complexity, computational overhead

### 🎯 Recommended Approach: English Translation Pipeline

**Rationale:**
1. **LLM Performance**: English models generally more stable and capable
2. **Scientific Terminology**: Better handling of technical terms in English
3. **Resource Availability**: More tools and models available for English
4. **Hallucination Reduction**: English processing typically more reliable

**Implementation Plan:**
1. **Document Translation**: Use open-source translation models locally
2. **Embedding Recreation**: Generate English embeddings for translated documents
3. **Pipeline Adaptation**: Modify all components for English processing
4. **Final Translation**: Convert answers back to Korean for submission

### Local Translation Implementation

#### Using Open-Source Translation Models
```python
# Using Helsinki-NLP models (works offline)
from transformers import pipeline

class LocalTranslator:
    def __init__(self):
        # Korean to English translation
        self.ko_en_translator = pipeline(
            "translation", 
            model="Helsinki-NLP/opus-mt-ko-en",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # English to Korean translation  
        self.en_ko_translator = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-en-ko", 
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    def translate_documents(self, documents_path: str, target_lang: str = "en"):
        """Translate all documents in JSONL file"""
        translated_docs = []
        
        with open(documents_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                if target_lang == "en":
                    translated_text = self.ko_en_translator(doc['content'])[0]['translation_text']
                else:
                    translated_text = self.en_ko_translator(doc['content'])[0]['translation_text']
                
                translated_doc = doc.copy()
                translated_doc['original_content'] = doc['content']
                translated_doc['content'] = translated_text
                translated_doc['language'] = target_lang
                
                translated_docs.append(translated_doc)
        
        return translated_docs
```

#### Batch Processing for Large Datasets
```python
def batch_translate_documents(documents: List[Dict], batch_size: int = 32):
    """Process documents in batches for efficiency"""
    translated = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_texts = [doc['content'] for doc in batch]
        
        # Translate batch
        translations = translator.ko_en_translator(
            batch_texts, 
            batch_size=batch_size,
            truncation=True,
            max_length=512
        )
        
        # Update documents with translations
        for doc, translation in zip(batch, translations):
            new_doc = doc.copy()
            new_doc['content'] = translation['translation_text']
            translated.append(new_doc)
    
    return translated
```