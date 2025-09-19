# Ollama Integration Guide for RAG Pipeline

## Overview
This guide explains how to use Ollama models in your RAG pipeline, including current capabilities and future enhancement possibilities.

## Current Ollama Support ✅

### What's Working Now:
- **Answer Generation**: Full Ollama support for final answer generation
- **Parallel Processing**: Concurrent query processing with multiple workers
- **Model Selection**: Support for qwen2:7b and llama3.1:8b

### Configuration Files:

#### 1. `conf/pipeline/ollama.yaml` (Primary)
```yaml
generator_type: "ollama"
generator_model_name: "qwen2:7b"  # Your main Ollama model
```
- ✅ Uses Ollama for answer generation
- ⚠️ Uses OpenAI for tool calling and query rewriting
- 💡 Best balance of cost and performance

#### 2. `conf/pipeline/ollama-llama.yaml`
```yaml
generator_type: "ollama"
generator_model_name: "llama3.1:8b"  # Llama for potentially better reasoning
```
- ✅ Uses Llama 3.1 8B for answer generation
- ⚠️ Uses OpenAI for tool calling and query rewriting
- 💡 Better reasoning but slower inference

#### 3. `conf/pipeline/hybrid-ollama.yaml`
```yaml
generator_type: "ollama"
generator_model_name: "qwen2:7b"
```
- ✅ Same as primary Ollama config
- 📝 Includes detailed notes about limitations

## Why OpenAI is Still Used ⚠️

### Tool Calling (Function Calling)
- **Why OpenAI**: Ollama doesn't have native function calling support
- **What it does**: Enables `scientific_search` tool execution
- **Impact**: Critical for retrieval pipeline functionality

### Query Rewriting
- **Why OpenAI**: Requires complex reasoning and consistent behavior
- **What it does**: Converts conversational queries to search-optimized queries
- **Impact**: Improves retrieval quality significantly

## Future Enhancements 🚀

### 1. Ollama Function Calling
**Status**: Not implemented
**Requirements**:
- Custom JSON parsing for tool calls
- Structured output prompting
- Manual parameter extraction

### 2. Ollama Query Rewriting
**Status**: Experimental implementation available
**Location**: `src/ir_core/orchestration/rewriter_ollama.py`
**Features**:
- Custom prompt engineering
- Response parsing and cleaning
- Fallback to original query

### 3. Full OSS Pipeline
**Configuration**: `conf/pipeline/ollama-full.yaml`
**Requirements**:
- Implement Ollama function calling
- Integrate OllamaQueryRewriter
- Extensive testing and validation

## Usage Instructions

### CLI Menu Options:
```bash
python scripts/cli_menu.py
# Select: Experiments & Validation
# Choose: Validate Retrieval (Ollama) - for Qwen2
# Or: Validate Retrieval (Ollama Llama) - for Llama
```

### Direct Commands:
```bash
# Use Qwen2
PYTHONPATH=src poetry run python scripts/evaluation/validate_retrieval.py \
  --config-dir conf pipeline=ollama limit=10

# Use Llama
PYTHONPATH=src poetry run python scripts/evaluation/validate_retrieval.py \
  --config-dir conf pipeline=ollama-llama limit=10
```

## Model Recommendations

### Qwen2 7B
- ✅ **Pros**: Fast inference, good general performance
- ⚠️ **Cons**: May have slightly lower reasoning quality
- 💡 **Best for**: Speed-focused workflows, cost optimization

### Llama 3.1 8B
- ✅ **Pros**: Better reasoning, more accurate responses
- ⚠️ **Cons**: Slower inference, higher resource usage
- 💡 **Best for**: Quality-focused workflows, complex reasoning tasks

## Performance Comparison

| Configuration | Generation | Tool Calling | Rewriting | Speed | Cost |
|---------------|------------|--------------|-----------|-------|------|
| OpenAI Only   | OpenAI     | OpenAI       | OpenAI    | Fast  | High |
| Hybrid Ollama | Ollama     | OpenAI       | OpenAI    | Medium| Low  |
| Full Ollama   | Ollama     | Ollama*      | Ollama*   | Slow  | Free |

*Requires implementation

## Cost Savings

### Current Hybrid Approach:
- **API Costs**: Only for tool calling + rewriting (~20-30% of total)
- **Local Costs**: Answer generation (~70-80% of total)
- **Estimated Savings**: 70-80% compared to full OpenAI

### Full Ollama (Future):
- **API Costs**: $0
- **Local Costs**: Electricity only
- **Estimated Savings**: 100% API costs

## Next Steps

1. **Immediate**: Use hybrid configurations for cost savings
2. **Short-term**: Implement Ollama query rewriting
3. **Long-term**: Implement Ollama function calling for full OSS stack

## Troubleshooting

### Common Issues:
1. **Ollama server not running**: Start with `ollama serve`
2. **Model not available**: Pull with `ollama pull qwen2:7b`
3. **Slow performance**: Reduce `analysis.max_workers` in config
4. **API errors**: Check OpenAI API key and rate limits

### Performance Tuning:
- Adjust `analysis.max_workers` based on your hardware
- Use smaller models for faster inference
- Enable parallel processing for batch operations
