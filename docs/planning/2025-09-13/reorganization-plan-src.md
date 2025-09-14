
**Current Structure:**
- **core.py** (~500 lines): Main orchestrator with `RetrievalAnalyzer` class handling batch analysis, parallel processing, and result aggregation. Contains extensive hardcoded domain keywords and scientific terms.
- **metrics.py** (~250 lines): Focused on metric calculations (`RetrievalMetrics` class). Well-structured but has hardcoded K-values (e.g., [1,3,5,10]) and some duplication in metric computations.
- **query_analyzer.py** (~600 lines): Comprehensive query analysis (`QueryAnalyzer` class) with feature extraction, domain classification, and complexity scoring. Heavily reliant on hardcoded lists (domain keywords, scientific terms, regex patterns).

**Key Issues Identified:**
1. **Long Files & Classes**: core.py and query_analyzer.py are monolithic, making them hard to maintain and test.
2. **Hardcoded Values**: Extensive duplication of domain keywords, scientific terms, and thresholds across files (e.g., physics/biology keywords appear in both core.py and query_analyzer.py).
3. **Mixed Responsibilities**: Classes handle too many concerns (e.g., `RetrievalAnalyzer` does metrics, analysis, and recommendations).
4. **Configuration Gaps**: Many parameters (e.g., thresholds, worker counts) use defaults instead of proper config integration.
5. **Limited Reusability**: Tight coupling makes it hard to reuse individual components.
6. **Testing Gaps**: No visible unit tests for individual functions/methods.
7. **Documentation**: Inconsistent docstrings and type hints.

**Strengths:**
- Functional parallel processing support.
- Comprehensive feature extraction in `QueryAnalyzer`.
- Clear separation of metrics logic in metrics.py.

### Suggested Improvements for Maintainability & Understandability

1. **Modularize Code**: Break large classes into smaller, focused components.
2. **Centralize Constants**: Extract all hardcoded lists to a shared constants file.
3. **Configuration-Driven**: Make thresholds and parameters configurable via Hydra config.
4. **Improve Abstractions**: Use dataclasses/interfaces for better type safety.
5. **Add Testing**: Unit tests for each component.
6. **Documentation**: Consistent docstrings and type hints.

### Refactoring Plan

I've created a structured plan above with 10 tasks. Here's a high-level overview:

**Phase 1: Foundation (Tasks 1-3)**
- Extract constants and make parameters configurable. This addresses the core issue of hardcoded values and improves flexibility.

**Phase 2: Structural Refactoring (Tasks 4-6)**
- Break down large classes into smaller, single-responsibility components. This will make the code more modular and easier to understand/maintain.

**Phase 3: Quality Assurance (Tasks 7-10)**
- Add documentation, tests, and validation to ensure the refactored code is robust.

**Estimated Effort:** This is a substantial refactoring that could take several sessions. Start with Phase 1 (extracting constants) as it's low-risk and provides immediate benefits. Each phase builds on the previous one.

**Benefits After Refactoring:**
- Easier to modify domain keywords or add new scientific terms (centralized in one file).
- Smaller files/classes are easier to read, test, and debug.
- Better reusability (e.g., use `DomainClassifier` independently).
- Improved maintainability for future enhancements.

Would you like me to start with the first task (extracting constants) or focus on a specific aspect of the plan?