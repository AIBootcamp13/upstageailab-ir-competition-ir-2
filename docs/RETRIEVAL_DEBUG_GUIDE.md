# Retrieval Debug Tool

This Streamlit page provides comprehensive debugging capabilities for testing and analyzing retrieval performance in the RAG system.

## Features

### 🔍 Retrieval Testing
- **Sparse Retrieval (BM25)**: Traditional text matching using TF-IDF and BM25 scoring
- **Dense Retrieval (Embeddings)**: Semantic search using vector embeddings and cosine similarity
- **Hybrid Retrieval**: Combines both sparse and dense retrieval for optimal results

### 📊 Detailed Output
- Document IDs and ranking
- Retrieval scores (BM25, Cosine, Final scores)
- Content previews with expandable full content
- Confidence logging with detailed analysis

### 🔧 Debug Options
- Enable/disable debug logging
- Show full content or previews
- Configurable parameters (alpha, k values, etc.)

## How to Use

1. **Select Page**: Choose "🔍 Retrieval Debug" from the sidebar
2. **Enter Query**: Input your search query in Korean
3. **Choose Method**: Select Sparse, Dense, or Hybrid retrieval
4. **Adjust Parameters**:
   - Number of results to retrieve
   - For Hybrid: Alpha weight, BM25 K, Rerank K
5. **Enable Debug**: Check "Enable Debug Logging" for detailed analysis
6. **Run Test**: Click "🚀 Run Retrieval Test"

## Example Queries

- `통학 버스의 가치` (The value of school buses)
- `인공지능의 발전` (Development of AI)
- `환경 보호 방법` (Environmental protection methods)
- `학교 교육의 중요성` (Importance of school education)

## Output Analysis

### Sparse Retrieval Results
- **BM25 Score**: Relevance score from text matching
- **Document ID**: Unique identifier for each document
- **Content Preview**: First 300 characters of the document

### Dense Retrieval Results
- **Cosine Score**: Semantic similarity score
- **Embedding Shape**: Vector dimensions used
- **Content Preview**: Relevant document excerpts

### Hybrid Retrieval Results
- **Final Score**: Combined score from BM25 + Dense
- **BM25 Score**: Text matching component
- **Cosine Score**: Semantic similarity component
- **Alpha**: Weight given to dense retrieval

## Debug Information

When debug logging is enabled, you'll see:
- Detailed confidence scores
- Retrieval performance metrics
- Document metadata analysis
- Query processing information

## Tips

- Start with simple queries to understand baseline performance
- Use Hybrid retrieval for most comprehensive results
- Enable debug logging to see detailed scoring breakdowns
- Adjust Alpha values to balance between sparse and dense retrieval
- Try different K values to see how result count affects quality</content>
<parameter name="filePath">/home/wb2x/workspace/information_retrieval_rag/docs/RETRIEVAL_DEBUG_GUIDE.md