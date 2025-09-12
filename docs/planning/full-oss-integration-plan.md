# Full Open Source Integration Plan for RAG Pipeline

## Executive Summary

This plan outlines the implementation of a complete open-source software (OSS) stack for the RAG pipeline, replacing OpenAI dependencies with Ollama-based alternatives. The goal is to achieve 100% API cost reduction while maintaining or improving performance.

**Current Status**: Hybrid implementation (70-80% cost reduction)
**Target Status**: Full OSS implementation (100% cost reduction)
**Timeline**: 4-6 weeks
**Priority**: High (Cost Optimization + Privacy + Vendor Independence)

## Current Architecture Analysis

### ✅ Working Components (Ollama)
- Answer Generation: Ollama models (qwen2:7b, llama3.1:8b)
- Parallel Processing: ThreadPoolExecutor with 4 workers
- Model Selection: Dynamic configuration support

### ⚠️ OpenAI Dependencies
- Tool Calling: Function calling for `scientific_search`
- Query Rewriting: Conversational → search optimization
- Estimated API Usage: 20-30% of total costs

## Phase 1: Foundation & Assessment (Week 1)

### 1.1 Codebase Analysis
**Objective**: Understand current OpenAI usage patterns and integration points

**Tasks**:
- [ ] Audit all OpenAI API calls in codebase
- [ ] Document function calling usage patterns
- [ ] Analyze query rewriting requirements and complexity
- [ ] Identify fallback mechanisms and error handling

**Deliverables**:
- OpenAI usage report (`docs/openai-audit.md`)
- Function calling specification document
- Query rewriting requirements analysis

**Time Estimate**: 2-3 days
**Owner**: Development Team
**Success Criteria**: Complete inventory of OpenAI dependencies

### 1.2 Ollama Capabilities Assessment
**Objective**: Evaluate Ollama's capabilities for replacing OpenAI features

**Tasks**:
- [ ] Test Ollama structured output capabilities
- [ ] Evaluate JSON parsing for function calling
- [ ] Assess prompt engineering for query rewriting
- [ ] Benchmark Ollama vs OpenAI performance

**Deliverables**:
- Ollama capabilities assessment report
- Performance benchmark results
- Feasibility analysis for each component

**Time Estimate**: 2-3 days
**Owner**: AI/ML Engineer
**Success Criteria**: Clear understanding of Ollama limitations and workarounds

## Phase 2: Ollama Query Rewriting (Week 2-3)

### 2.1 Enhanced OllamaQueryRewriter Implementation
**Objective**: Implement production-ready Ollama-based query rewriting

**Tasks**:
- [ ] Enhance `src/ir_core/orchestration/rewriter_ollama.py`
- [ ] Implement robust response parsing and cleaning
- [ ] Add fallback mechanisms for parsing failures
- [ ] Create comprehensive prompt templates for different query types

**Technical Details**:
```python
class OllamaQueryRewriter:
    def __init__(self, model_name: str, prompt_template_path: str):
        # Initialize with model selection
        # Load and validate prompt templates

    def rewrite_query(self, query: str) -> str:
        # Generate structured prompt
        # Call Ollama API with proper parameters
        # Parse and validate response
        # Return cleaned rewritten query or original
```

**Deliverables**:
- Production-ready `OllamaQueryRewriter` class
- Comprehensive test suite
- Performance benchmarks vs OpenAI

**Time Estimate**: 5-7 days
**Owner**: AI/ML Engineer
**Dependencies**: Phase 1 completion
**Success Criteria**: 90%+ success rate in query rewriting

### 2.2 Configuration Integration
**Objective**: Integrate Ollama query rewriter into pipeline configuration

**Tasks**:
- [ ] Update `src/ir_core/orchestration/__init__.py` to support Ollama rewriter
- [ ] Modify pipeline factory to instantiate correct rewriter type
- [ ] Update configuration files to support `query_rewriter_type: "ollama"`
- [ ] Add model selection for rewriting (`rewriter_model`)

**Configuration Example**:
```yaml
# conf/pipeline/ollama-rewriter.yaml
tool_calling_model: "gpt-3.5-turbo-1106"  # Still OpenAI
rewriter_model: "qwen2:7b"                # Now Ollama
query_rewriter_type: "ollama"             # New option
generator_type: "ollama"
generator_model_name: "qwen2:7b"
```

**Deliverables**:
- Updated pipeline factory
- New configuration files
- Integration tests

**Time Estimate**: 2-3 days
**Owner**: Backend Engineer
**Dependencies**: Enhanced OllamaQueryRewriter
**Success Criteria**: Pipeline correctly instantiates Ollama rewriter

### 2.3 Testing & Validation
**Objective**: Ensure Ollama query rewriting maintains quality

**Tasks**:
- [ ] Create comprehensive test dataset for query rewriting
- [ ] Compare Ollama vs OpenAI rewriting quality
- [ ] Measure impact on downstream retrieval performance
- [ ] Implement A/B testing framework

**Deliverables**:
- Quality assessment report
- Performance impact analysis
- A/B testing results

**Time Estimate**: 3-4 days
**Owner**: QA Engineer
**Dependencies**: Configuration integration
**Success Criteria**: <5% performance degradation vs OpenAI

## Phase 3: Ollama Function Calling (Week 4-5)

### 3.1 Function Calling Architecture Design
**Objective**: Design Ollama-based function calling system

**Tasks**:
- [ ] Research Ollama function calling approaches
- [ ] Design JSON parsing and validation system
- [ ] Create structured prompting framework
- [ ] Design fallback mechanisms for parsing failures

**Technical Approaches**:
1. **Structured Output**: Use Ollama's JSON mode with schema validation
2. **Regex Parsing**: Extract function calls from natural language responses
3. **Hybrid Approach**: Combine structured output with regex fallbacks

**Deliverables**:
- Function calling architecture document
- Technical specification for implementation
- Proof of concept prototypes

**Time Estimate**: 3-4 days
**Owner**: AI/ML Engineer
**Dependencies**: Phase 2 completion
**Success Criteria**: Clear technical approach with success probability >80%

### 3.2 OllamaToolCaller Implementation
**Objective**: Implement Ollama-based tool calling system

**Tasks**:
- [ ] Create `src/ir_core/orchestration/tool_caller_ollama.py`
- [ ] Implement function schema to prompt conversion
- [ ] Add response parsing and parameter extraction
- [ ] Create comprehensive error handling and fallbacks

**Core Implementation**:
```python
class OllamaToolCaller:
    def __init__(self, model_name: str, tools: List[Dict]):
        # Convert tool schemas to prompts
        # Initialize Ollama client

    def call_tool(self, query: str, tools: List[Dict]) -> Dict:
        # Generate tool calling prompt
        # Call Ollama API
        # Parse tool call from response
        # Execute tool and return results
```

**Deliverables**:
- Complete `OllamaToolCaller` implementation
- Tool schema conversion utilities
- Comprehensive test suite

**Time Estimate**: 5-7 days
**Owner**: AI/ML Engineer
**Dependencies**: Architecture design
**Success Criteria**: 85%+ success rate in tool calling

### 3.3 Pipeline Integration
**Objective**: Integrate Ollama tool calling into RAG pipeline

**Tasks**:
- [ ] Update `src/ir_core/orchestration/pipeline.py` to support Ollama tool calling
- [ ] Modify `run_retrieval_only` method for Ollama compatibility
- [ ] Add configuration support for tool calling model selection
- [ ] Implement graceful fallback to OpenAI when Ollama fails

**Integration Points**:
- Replace OpenAI client with OllamaToolCaller
- Update tool definition handling
- Add error recovery mechanisms

**Deliverables**:
- Updated pipeline implementation
- New configuration files
- Integration tests

**Time Estimate**: 4-5 days
**Owner**: Backend Engineer
**Dependencies**: OllamaToolCaller implementation
**Success Criteria**: Pipeline works with both OpenAI and Ollama tool calling

## Phase 4: Full OSS Integration & Optimization (Week 6)

### 4.1 Complete Configuration Suite
**Objective**: Create comprehensive OSS configuration options

**Tasks**:
- [ ] Create `conf/pipeline/full-oss.yaml` for complete Ollama setup
- [ ] Update CLI menu with full OSS options
- [ ] Add configuration validation for OSS requirements
- [ ] Create migration guide from hybrid to full OSS

**Configuration Options**:
```yaml
# Full OSS Configuration
tool_calling_model: "qwen2:7b"        # Ollama
rewriter_model: "qwen2:7b"            # Ollama
query_rewriter_type: "ollama"         # Ollama
generator_type: "ollama"              # Ollama
generator_model_name: "qwen2:7b"      # Ollama
```

**Deliverables**:
- Complete configuration suite
- Updated CLI interface
- Migration documentation

**Time Estimate**: 2-3 days
**Owner**: DevOps Engineer
**Dependencies**: All previous phases
**Success Criteria**: Seamless configuration switching

### 4.2 Performance Optimization
**Objective**: Optimize full OSS pipeline for production use

**Tasks**:
- [ ] Profile and optimize Ollama API calls
- [ ] Implement intelligent caching for repeated queries
- [ ] Optimize parallel processing for Ollama models
- [ ] Add model warm-up and connection pooling

**Optimization Areas**:
- API call batching
- Response caching
- Connection management
- Memory optimization

**Deliverables**:
- Performance optimization report
- Optimized configuration files
- Monitoring and alerting setup

**Time Estimate**: 3-4 days
**Owner**: Performance Engineer
**Dependencies**: Full integration
**Success Criteria**: <10% performance degradation vs OpenAI

### 4.3 Production Readiness
**Objective**: Prepare full OSS solution for production deployment

**Tasks**:
- [ ] Implement comprehensive monitoring and logging
- [ ] Create automated health checks for Ollama services
- [ ] Add circuit breakers for API failures
- [ ] Develop rollback procedures to OpenAI

**Production Requirements**:
- Service health monitoring
- Automated failover mechanisms
- Performance alerting
- Cost tracking and reporting

**Deliverables**:
- Production deployment guide
- Monitoring dashboard configuration
- Incident response procedures

**Time Estimate**: 2-3 days
**Owner**: DevOps Engineer
**Dependencies**: Performance optimization
**Success Criteria**: Production-ready OSS pipeline

## Risk Assessment & Mitigation

### Technical Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Ollama function calling reliability | Medium | High | Implement robust fallbacks and error handling |
| Performance degradation | Low | Medium | Comprehensive benchmarking and optimization |
| Model availability issues | Low | Medium | Implement model download and health checks |
| Complex integration points | Medium | High | Incremental implementation with thorough testing |

### Business Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| API cost reduction goals not met | Low | Medium | Phased approach with measurable milestones |
| Performance requirements not satisfied | Low | High | Extensive testing and A/B comparisons |
| Vendor lock-in concerns | Low | Low | Already addressed by hybrid approach |

## Success Metrics

### Technical Metrics
- **Function Calling Success Rate**: >85%
- **Query Rewriting Quality**: <5% degradation vs OpenAI
- **End-to-End Performance**: <10% slower than OpenAI
- **Error Rate**: <1% for normal operations

### Business Metrics
- **API Cost Reduction**: 100% (target)
- **Infrastructure Cost**: <20% increase
- **Time to Complete**: <6 weeks
- **Quality Maintenance**: >95% of OpenAI performance

## Resource Requirements

### Team Composition
- **AI/ML Engineer** (Primary): 4-6 weeks
- **Backend Engineer**: 2-3 weeks
- **QA Engineer**: 2-3 weeks
- **DevOps Engineer**: 1-2 weeks
- **Performance Engineer**: 1-2 weeks

### Infrastructure Requirements
- Ollama server with GPU acceleration
- Sufficient RAM for model loading (16GB+ recommended)
- Fast storage for model caching
- Monitoring and logging infrastructure

### Dependencies
- Ollama 0.1.0+ with JSON mode support
- Python 3.10+ with async support
- Sufficient GPU memory for concurrent model usage
- Network connectivity for model downloads

## Timeline & Milestones

```
Week 1: Foundation & Assessment
├── Days 1-3: Codebase analysis & Ollama assessment
└── Days 4-5: Planning & resource allocation

Week 2-3: Ollama Query Rewriting
├── Days 6-10: Enhanced OllamaQueryRewriter implementation
├── Days 11-12: Configuration integration
└── Days 13-15: Testing & validation

Week 4-5: Ollama Function Calling
├── Days 16-19: Architecture design & POC
├── Days 20-24: OllamaToolCaller implementation
└── Days 25-28: Pipeline integration

Week 6: Full Integration & Optimization
├── Days 29-32: Complete configuration suite
├── Days 33-36: Performance optimization
└── Days 37-40: Production readiness
```

## Budget Considerations

### Development Costs
- Team effort: 8-12 person-weeks
- Infrastructure: Ollama server setup
- Testing: Comprehensive validation suite

### Operational Savings
- OpenAI API costs: 100% elimination
- Infrastructure costs: Minimal increase
- ROI: Positive within 1-2 months

## Conclusion & Next Steps

This plan provides a structured approach to achieving full OSS integration while minimizing risks and ensuring quality maintenance. The phased approach allows for:

1. **Early Wins**: Cost savings from hybrid approach
2. **Incremental Progress**: Measurable milestones each week
3. **Risk Mitigation**: Fallback mechanisms and thorough testing
4. **Quality Assurance**: Performance monitoring and validation

**Immediate Next Step**: Begin Phase 1 codebase analysis and Ollama capabilities assessment.

**Success Probability**: High (80%+) with proper execution of the phased approach.

---

*Document Version: 1.0*
*Last Updated: September 12, 2025*
*Owner: wchoi189@gmail.com*
