# Dataset Documentation: Korean Scientific QA with Metadata

## Overview
This document provides comprehensive information about the dataset being used in the RAG system for the Information Retrieval competition.

## Dataset Configuration
- **Current Configuration**: `science_qa_ko_metadata`
- **Documents File**: `data/documents_ko_with_metadata.jsonl`
- **Validation File**: `data/validation_balanced.jsonl`
- **Output File**: `outputs/submission_meta.csv`

## Document Collection Statistics

### Basic Statistics
- **Total Documents**: 4,271
- **Language**: Korean (ko)
- **Average Content Length**: 315 characters per document
- **Content Range**: 123-? characters (Min: 1230 shown, likely longer max)

### Data Sources Distribution
| Source | Count | Percentage |
|--------|-------|------------|
| ko_ai2_arc__ARC_Challenge__test | 943 | 22.1% |
| ko_ai2_arc__ARC_Challenge__train | 865 | 20.2% |
| ko_ai2_arc__ARC_Challenge__validation | 238 | 5.6% |
| ko_mmlu__conceptual_physics__test | 211 | 4.9% |
| ko_mmlu__nutrition__test | 168 | 3.9% |
| ko_mmlu__human_aging__test | 168 | 3.9% |
| ko_mmlu__high_school_biology__test | 131 | 3.1% |
| ko_mmlu__astronomy__test | 122 | 2.9% |
| ko_mmlu__high_school_chemistry__test | 118 | 2.8% |
| ko_mmlu__electrical_engineering__test | 115 | 2.7% |
| ko_mmlu__virology__test | 103 | 2.4% |
| ko_mmlu__college_medicine__test | 98 | 2.3% |
| ko_mmlu__high_school_physics__test | 85 | 2.0% |
| ko_mmlu__global_facts__test | 85 | 2.0% |
| ko_mmlu__human_sexuality__test | 80 | 1.9% |
| ko_mmlu__computer_security__test | 74 | 1.7% |
| ko_mmlu__medical_genetics__test | 73 | 1.7% |
| ko_mmlu__anatomy__test | 72 | 1.7% |
| ko_mmlu__college_biology__test | 71 | 1.7% |
| ko_mmlu__college_physics__test | 66 | 1.5% |
| ko_mmlu__college_chemistry__test | 63 | 1.5% |
| ko_mmlu__college_computer_science__test | 35 | 0.8% |
| ko_mmlu__high_school_computer_science__test | 32 | 0.7% |
| *[Additional validation/train splits]* | *Various* | *~5.0%* |

### Domain Categories
The dataset covers diverse scientific domains:
- **Biology**: nutrition, human aging, virology, anatomy, medical genetics
- **Physics**: conceptual physics, high school physics, college physics
- **Chemistry**: high school chemistry, college chemistry
- **Computer Science**: college CS, high school CS, computer security
- **Medicine**: college medicine, anatomy
- **Engineering**: electrical engineering
- **Astronomy**: space science
- **General Science**: ARC Challenge (science reasoning)

## Document Schema

### Core Fields
```json
{
  "docid": "string (UUID format)",
  "src": "string (source identifier)",
  "content": "string (main document text)"
}
```

### Metadata Fields
```json
{
  "summary": "string (concise summary of the document)",
  "keywords": ["array of strings (7.4 avg per document)"],
  "hypothetical_questions": ["array of strings (5.0 avg per document)"]
}
```

### Example Document
```json
{
  "docid": "42508ee0-c543-4338-878e-d98c6babee66",
  "src": "ko_mmlu__nutrition__test",
  "content": "건강한 사람이 에너지 균형을 평형 상태로 유지하는 것은 중요합니다...",
  "summary": "이 문서는 건강한 사람에게 에너지 균형을 유지하는 중요성을 설명하고 있습니다...",
  "keywords": ["건강", "에너지 균형", "식단", "운동", "영양가 있는 식품"],
  "hypothetical_questions": [
    "1. 에너지 균형을 유지하는 것이 우리의 건강에 어떤 이점이 있나요?",
    "2. 에너지 균형을 유지하려면 어떻게 식단과 운동을 조절해야 하나요?"
  ]
}
```

## Validation Dataset Statistics

### Basic Statistics
- **Total Queries**: 106
- **Average Query Length**: 122 characters
- **Language**: Korean (ko)

### Domain Distribution
| Domain | Count | Percentage |
|--------|-------|------------|
| arc_challenge | 47 | 44.3% |
| mmlu_biology | 22 | 20.8% |
| mmlu_physics | 12 | 11.3% |
| mmlu_other | 5 | 4.7% |
| mmlu_medicine | 5 | 4.7% |
| mmlu_computer_science | 5 | 4.7% |
| mmlu_chemistry | 5 | 4.7% |
| mmlu_astronomy | 5 | 4.7% |

### Validation Query Schema
```json
{
  "eval_id": "string (unique identifier)",
  "msg": [
    {
      "role": "user",
      "content": "string (query text)"
    }
  ],
  "ground_truth_doc_id": "string (target document UUID)",
  "domain": "string (domain category)",
  "source": "string (source dataset)"
}
```

### Example Validation Query
```json
{
  "eval_id": "balanced_mmlu_biology_011",
  "msg": [
    {
      "role": "user",
      "content": "이 생물의 특징은 무엇인가요? 영양 역학에서 교란이 까다로운 도전인 이유는..."
    }
  ],
  "ground_truth_doc_id": "9758eb22-6576-432a-8846-2ba035c2ccb9",
  "domain": "mmlu_biology",
  "source": "ko_mmlu__nutrition__test"
}
```

## Data Characteristics

### Content Analysis
- **Rich Metadata**: Each document includes summary, keywords, and hypothetical questions
- **Educational Focus**: Primarily scientific and educational content
- **Korean Language**: All content is in Korean
- **Structured Format**: JSONL format for efficient processing

### Quality Features
- **Ground Truth Available**: Validation set has known correct answers
- **Domain Balance**: Good distribution across scientific domains
- **Metadata Enrichment**: Additional context through summaries and keywords
- **Question Generation**: Hypothetical questions for each document

### Potential Challenges
- **Scientific Complexity**: Advanced scientific concepts across multiple domains
- **Korean Language Processing**: Requires good Korean NLP capabilities
- **Long-tail Domains**: Some domains have fewer examples
- **Query Diversity**: Mix of different question types and complexities

## Usage in RAG System

### Current Configuration
- **Embedding Model**: EleutherAI/polyglot-ko-1.3b (2048 dimensions)
- **Index**: docs-ko-polyglot-1b-d2048-20250918
- **Text Analyzer**: Nori (custom "korean" analyzer)
- **Retrieval**: Hybrid BM25 + Dense retrieval (alpha=0.8)
- **Reranking**: Top-10 reranking

### Performance Context
- **Current MAP**: 0.333 (after fixing embedding issues)
- **Previous MAP**: 0.000 (before embedding fix)
- **Improvement**: 33x performance increase after proper indexing

## Recommendations for Future Work

1. **Domain-Specific Tuning**: Consider domain-specific retrieval parameters
2. **Query Enhancement**: Leverage the hypothetical questions for better retrieval
3. **Metadata Utilization**: Use keywords and summaries in retrieval process
4. **Cross-Domain Evaluation**: Test performance across different scientific domains
5. **Query Type Analysis**: Analyze performance by question complexity and type

## File Locations
- **Documents**: `/home/wb2x/workspace/information_retrieval_rag/data/documents_ko_with_metadata.jsonl`
- **Validation**: `/home/wb2x/workspace/information_retrieval_rag/data/validation_balanced.jsonl`
- **Configuration**: `/home/wb2x/workspace/information_retrieval_rag/conf/data/science_qa_ko_metadata.yaml`
- **Index**: `docs-ko-polyglot-1b-d2048-20250918` (Elasticsearch)

## Validator
Run the pre-flight validator before evaluation or reindexing:
```
PYTHONPATH=src uv run python scripts/indexing/validate_index_dimensions.py --index docs-ko-polyglot-1b-d2048-20250918 --check-analyzer
```

This dataset provides a rich foundation for scientific question answering in Korean, with comprehensive metadata and balanced domain coverage.</content>
<parameter name="filePath">/home/wb2x/workspace/information_retrieval_rag/DATASET_DOCUMENTATION.md</parameter>