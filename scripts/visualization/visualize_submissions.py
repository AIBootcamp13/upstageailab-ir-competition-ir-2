import streamlit as st
import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

# Import retrieval functions for debug page
try:
    from ir_core.retrieval.core import sparse_retrieve, dense_retrieve, hybrid_retrieve
    from ir_core.embeddings.core import encode_texts
    from ir_core.query_enhancement.confidence_logger import ConfidenceLogger
    RETRIEVAL_AVAILABLE = True
except ImportError as e:
    st.error(f"Retrieval modules not available: {e}")
    RETRIEVAL_AVAILABLE = False
    # Define as None to avoid "possibly unbound" errors
    sparse_retrieve = None
    dense_retrieve = None
    hybrid_retrieve = None
    encode_texts = None
    ConfidenceLogger = None

# Function to load JSONL data
def load_jsonl(file_path):
    try:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        st.warning(f"Skipping malformed JSON at line {line_num} in {Path(file_path).name}: {e}")
                        continue
        return pd.DataFrame(data)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading {Path(file_path).name}: {e}")
        return pd.DataFrame()

# Function to get submission files
def get_submission_files():
    outputs_dir = Path("/home/wb2x/workspace/information_retrieval_rag/outputs")
    files = list(outputs_dir.glob("submission*.jsonl")) + list(outputs_dir.glob("sample_submission*.jsonl"))
    return [str(f) for f in files]

# Function to get document files
def get_document_files():
    data_dir = Path("/home/wb2x/workspace/information_retrieval_rag/data")
    files = list(data_dir.glob("document*.jsonl"))
    return [str(f) for f in files]

def get_filename(path):
    return Path(path).name

# Main app
def main():
    st.set_page_config(layout="wide", page_title="RAG Analysis & Debug Tool")

    # Page selector
    st.sidebar.title("🔧 RAG Analysis Tool")
    page = st.sidebar.radio("Select Page", ["📊 Submission Visualizer", "🔍 Retrieval Debug"])

    if page == "📊 Submission Visualizer":
        show_submission_visualizer()
    elif page == "🔍 Retrieval Debug":
        show_retrieval_debug()

def show_submission_visualizer():
    st.markdown("<h1 style='font-size: 32px;'>📊 RAG Submission Visualizer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 18px;'>Visualize and compare generation results from submission files.</p>", unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("Settings")
    submission_files = get_submission_files()
    selected_file = st.sidebar.selectbox("Select Submission File", submission_files, format_func=get_filename)

    # Documents
    document_files = get_document_files()
    selected_doc_file = st.sidebar.selectbox("Select Document File", document_files, format_func=get_filename)
    load_docs = st.sidebar.checkbox("Load Documents")
    df_docs = None
    if load_docs and selected_doc_file:
        df_docs = load_jsonl(selected_doc_file)
        st.sidebar.success(f"Loaded {len(df_docs)} documents from {Path(selected_doc_file).name}")

    # Comparison file
    compare_file = st.sidebar.selectbox(
        "Select Comparison Submission File",
        [None] + submission_files,
        format_func=lambda x: get_filename(x) if x else "None"
    )

    if selected_file:
        try:
            df = load_jsonl(selected_file)
            if df.empty:
                st.error(f"No data loaded from {Path(selected_file).name}. The file might be empty or contain invalid JSON.")
                st.stop()
            st.sidebar.success(f"Loaded {len(df)} entries from {Path(selected_file).name}")
        except Exception as e:
            st.error(f"Failed to load {Path(selected_file).name}: {e}")
            st.stop()

        df2 = None
        if compare_file and compare_file != selected_file:
            try:
                df2 = load_jsonl(compare_file)
                if df2.empty:
                    st.warning(f"No data loaded from comparison file {Path(compare_file).name}")
                else:
                    st.sidebar.success(f"Loaded comparison: {Path(compare_file).name}")
            except Exception as e:
                st.warning(f"Failed to load comparison file {Path(compare_file).name}: {e}")

        # Check if required columns exist
        required_cols = ['eval_id', 'standalone_query', 'answer']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns in {Path(selected_file).name}: {missing_cols}")
            st.write("Available columns:", df.columns.tolist())
            st.stop()

        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Detailed View", "Statistics", "Comparison", "Documents"])

        with tab1:
            st.header("Overview")
            # Display basic info
            st.dataframe(df[['eval_id', 'standalone_query', 'answer']].head(50))
            if len(df) > 50:
                st.info(f"Showing first 50 of {len(df)} entries. Use Detailed View for more.")

        with tab2:
            st.header("Detailed View")
            # Expandable details
            for idx, row in df.iterrows():
                with st.expander(f"Entry {row['eval_id']}: {row['standalone_query'][:50]}..."):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Query")
                        st.write(row['standalone_query'])
                        st.subheader("Answer")
                        st.write(row['answer'])
                    with col2:
                        st.subheader("TopK Sources")
                        st.write(row['topk'])
                        st.subheader("References")
                        for i, ref in enumerate(row['references']):
                            st.write(f"**Reference {i+1}** - Score: {ref['score']:.4f}")
                            with st.expander("Content Preview"):
                                st.write(ref['content'][:300] + "..." if len(ref['content']) > 300 else ref['content'])
                            with st.expander("Full Content"):
                                st.text_area("Full Content", ref['content'], height=200, key=f"content_{idx}_{i}", label_visibility="hidden")

        with tab3:
            st.header("Statistics")
            view_mode = st.radio("View Mode", ["Main File", "Comparison File", "Parallel"], horizontal=True)

            current_df = df
            file_name = Path(selected_file).name

            if view_mode == "Comparison File" and df2 is not None and compare_file:
                current_df = df2
                file_name = Path(compare_file).name
            elif view_mode == "Parallel" and df2 is not None and compare_file:
                # For parallel, we'll show two columns
                pass
            elif view_mode == "Comparison File" or view_mode == "Parallel":
                st.info("Select a comparison file in the sidebar for Comparison or Parallel view.")
                view_mode = "Main File"  # fallback

            if view_mode != "Parallel":
                # Extract scores for current_df
                all_scores = [ref['score'] for refs in current_df['references'] for ref in refs]
                scores_per_entry = [np.mean([ref['score'] for ref in refs]) if refs else 0.0 for refs in current_df['references']]

                # Normalization option
                normalize_scores = st.checkbox("Normalize Scores (0-1 range)", key=f"norm_{view_mode}")
                if normalize_scores and all_scores:
                    min_score = min(all_scores)
                    max_score = max(all_scores)
                    if max_score > min_score:
                        all_scores = [(s - min_score) / (max_score - min_score) for s in all_scores]
                        scores_per_entry = [(np.mean([ref['score'] for ref in refs]) if refs else 0.0) for refs in current_df['references']]
                        scores_per_entry = [(s - min_score) / (max_score - min_score) if s > 0 else 0.0 for s in scores_per_entry]

                # 100x scaling for extremely small scores
                original_max = max(all_scores) if all_scores else 0.0
                scale_100x = st.checkbox("Scale small scores by 100x", key=f"scale_{view_mode}",
                                       help="Multiply scores by 100 when max score < 0.1")
                if scale_100x and all_scores and original_max < 0.1:
                    all_scores = [s * 100 for s in all_scores]
                    scores_per_entry = [s * 100 for s in scores_per_entry]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Entries", len(current_df))
                with col2:
                    avg_score_val = np.mean(all_scores) if all_scores else 0.0
                    st.metric("Average Score", f"{avg_score_val:.4f}")
                with col3:
                    max_score_val = np.max(all_scores) if all_scores else 0.0
                    st.metric("Max Score", f"{max_score_val:.4f}")

                st.subheader(f"Score Distribution for {file_name}")
                fig = px.histogram(all_scores, nbins=20, title="Distribution of Similarity Scores")
                st.plotly_chart(fig)

                st.subheader("Average Score per Entry")
                fig2 = px.histogram(scores_per_entry, nbins=20, title="Average Similarity Score per Query")
                st.plotly_chart(fig2)

                # Number of references
                num_refs = [len(refs) for refs in current_df['references']]
                st.subheader("Number of References per Entry")
                fig3 = px.histogram(num_refs, title="Distribution of Reference Counts")
                st.plotly_chart(fig3)

                # Lowest scores
                st.subheader("Entries with Lowest Average Scores")
                low_score_df = pd.DataFrame({
                    'eval_id': current_df['eval_id'],
                    'query': current_df['standalone_query'],
                    'avg_score': scores_per_entry
                }).sort_values('avg_score').head(10)
                st.dataframe(low_score_df)

                # Zero or very low scores
                if all_scores:
                    min_score = min(all_scores)
                    max_score = max(all_scores)
                    # Set dynamic range based on actual scores
                    slider_min = 0.0
                    slider_max = min(1.0, max_score * 1.1)  # Cap at 1.0 for normalized scores
                    default_threshold = min(0.01, max_score * 0.1)  # 10% of max score or 0.01

                    threshold = st.slider(
                        "Threshold for low scores",
                        slider_min,
                        slider_max,
                        default_threshold,
                        step=slider_max/1000  # Fine-grained steps
                    )
                else:
                    threshold = st.slider("Threshold for low scores", 0.0, 0.01, 0.001)

                low_entries = [i for i, score in enumerate(scores_per_entry) if score < threshold]
                if low_entries:
                    st.subheader(f"Entries with Average Score < {threshold:.4f}")
                    low_df = pd.DataFrame({
                        'eval_id': current_df.iloc[low_entries]['eval_id'],
                        'query': current_df.iloc[low_entries]['standalone_query'],
                        'avg_score': [scores_per_entry[i] for i in low_entries]
                    })
                    st.dataframe(low_df)
                    st.info(f"Found {len(low_entries)} entries below threshold")
                else:
                    st.info(f"No entries with average score below {threshold:.4f}")
            else:
                # Parallel view
                if df2 is not None and compare_file:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Main File:")
                        st.write(f"**{Path(selected_file).name}**")
                        all_scores1 = [ref['score'] for refs in df['references'] for ref in refs]
                        scores_per_entry1 = [np.mean([ref['score'] for ref in refs]) if refs else 0.0 for refs in df['references']]

                        # Apply 100x scaling if needed
                        max_score1 = max(all_scores1) if all_scores1 else 0.0
                        if max_score1 < 0.1:
                            all_scores1 = [s * 100 for s in all_scores1]

                        st.metric("Total Entries", len(df))
                        st.metric("Average Score", f"{np.mean(all_scores1) if all_scores1 else 0.0:.4f}")
                        fig1 = px.histogram(all_scores1, nbins=20, title="Score Distribution")
                        st.plotly_chart(fig1)
                    with col2:
                        st.subheader("Comparison File:")
                        st.write(f"**{Path(compare_file).name}**")
                        all_scores2 = [ref['score'] for refs in df2['references'] for ref in refs]
                        scores_per_entry2 = [np.mean([ref['score'] for ref in refs]) if refs else 0.0 for refs in df2['references']]

                        # Apply 100x scaling if needed
                        max_score2 = max(all_scores2) if all_scores2 else 0.0
                        if max_score2 < 0.1:
                            all_scores2 = [s * 100 for s in all_scores2]

                        st.metric("Total Entries", len(df2))
                        st.metric("Average Score", f"{np.mean(all_scores2) if all_scores2 else 0.0:.4f}")
                        fig2 = px.histogram(all_scores2, nbins=20, title="Score Distribution")
                        st.plotly_chart(fig2)
                else:
                    st.info("Select a comparison file for Parallel view.")

        with tab4:
            st.header("Comparison")
            if compare_file and compare_file != selected_file:
                df2 = load_jsonl(compare_file)
                st.success(f"Loaded comparison file: {Path(compare_file).name}")

                # Merge on eval_id
                merged = pd.merge(df, df2, on='eval_id', suffixes=('_file1', '_file2'))

                # Compare scores
                scores1 = [np.mean([ref['score'] for ref in refs]) if refs else 0.0 for refs in merged['references_file1']]
                scores2 = [np.mean([ref['score'] for ref in refs]) if refs else 0.0 for refs in merged['references_file2']]

                # 100x scaling for extremely small scores in comparison
                max_score1 = max(scores1) if scores1 else 0.0
                max_score2 = max(scores2) if scores2 else 0.0
                scale_comparison = st.checkbox("Scale small scores by 100x in comparison",
                                            help="Multiply scores by 100 when max score < 0.1")
                if scale_comparison:
                    if max_score1 < 0.1:
                        scores1 = [s * 100 for s in scores1]
                    if max_score2 < 0.1:
                        scores2 = [s * 100 for s in scores2]

                comp_df = pd.DataFrame({
                    'eval_id': merged['eval_id'],
                    'query': merged['standalone_query_file1'],
                    'score_file1': scores1,
                    'score_file2': scores2,
                    'score_diff': np.array(scores2) - np.array(scores1)
                })

                st.dataframe(comp_df)

                # Difference distribution
                st.subheader("Score Difference Distribution")
                fig = px.histogram(comp_df['score_diff'], title="Differences in Average Scores")
                st.plotly_chart(fig)

                # Highlight significant differences
                st.subheader("Entries with Largest Score Differences")
                sorted_comp = comp_df.reindex(comp_df['score_diff'].abs().sort_values(ascending=False).index)
                st.dataframe(sorted_comp.head(10))

        with tab5:
            st.header("Documents")
            if df_docs is not None:
                search = st.text_input("Search documents (by content or ID)")
                if search:
                    # Simple search in all columns
                    filtered = df_docs[df_docs.apply(lambda row: search.lower() in str(row).lower(), axis=1)]
                    st.write(f"Found {len(filtered)} matching documents")
                    st.dataframe(filtered)
                else:
                    st.dataframe(df_docs)
                st.info("Check 'Load Documents' in the sidebar to view the documents.")

def show_retrieval_debug():
    st.markdown("<h1 style='font-size: 32px;'>🔍 Retrieval Debug Tool</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 18px;'>Test and debug sparse/dense retrieval with detailed output and confidence logging.</p>", unsafe_allow_html=True)

    if not RETRIEVAL_AVAILABLE:
        st.error("❌ Retrieval modules are not available. Please check the installation.")
        return

    # Sidebar controls
    st.sidebar.header("🔧 Debug Settings")

    # Query input
    default_query = "통학 버스의 가치"
    query = st.sidebar.text_input("Query", value=default_query, help="Enter your search query in Korean")

    # Retrieval parameters
    st.sidebar.subheader("Retrieval Parameters")
    retrieval_type = st.sidebar.selectbox(
        "Retrieval Type",
        ["Sparse (BM25)", "Dense (Embedding)", "Hybrid (Combined)"],
        help="Choose the retrieval method to test"
    )

    size = st.sidebar.slider("Number of Results", min_value=1, max_value=20, value=5,
                           help="Number of documents to retrieve")

    # Hybrid-specific parameters
    alpha = 0.4
    bm25_k = 200
    rerank_k = 10
    if retrieval_type == "Hybrid (Combined)":
        alpha = st.sidebar.slider("Alpha (Dense Weight)", min_value=0.0, max_value=1.0, value=0.4, step=0.1,
                                help="Weight for dense retrieval (0.0 = pure sparse, 1.0 = pure dense)")
        bm25_k = st.sidebar.slider("BM25 K", min_value=10, max_value=500, value=200, step=10,
                                 help="Number of documents to retrieve with BM25 for hybrid")
        rerank_k = st.sidebar.slider("Rerank K", min_value=5, max_value=50, value=10,
                                   help="Number of documents to rerank with dense retrieval")

    # Debug options
    st.sidebar.subheader("Debug Options")
    enable_debug = st.sidebar.checkbox("Enable Debug Logging", value=True,
                                     help="Show detailed retrieval scores and confidence information")
    show_full_content = st.sidebar.checkbox("Show Full Content", value=False,
                                          help="Display complete document content instead of preview")

    # Test button
    test_button = st.sidebar.button("🚀 Run Retrieval Test", type="primary", use_container_width=True)

    # Main content area
    if test_button and query.strip():
        st.header("🔍 Retrieval Results")

        # Show current settings
        st.info(f"**Query:** {query} | **Type:** {retrieval_type} | **Size:** {size}")
        if retrieval_type == "Hybrid (Combined)":
            st.info(f"**Hybrid Settings:** Alpha={alpha}, BM25 K={bm25_k}, Rerank K={rerank_k}")

        with st.spinner("Running retrieval test..."):
            try:
                if retrieval_type == "Sparse (BM25)":
                    run_sparse_retrieval(query, size, enable_debug, show_full_content)
                elif retrieval_type == "Dense (Embedding)":
                    run_dense_retrieval(query, size, enable_debug, show_full_content)
                elif retrieval_type == "Hybrid (Combined)":
                    run_hybrid_retrieval(query, bm25_k, rerank_k, alpha, enable_debug, show_full_content)

            except Exception as e:
                st.error(f"❌ Error during retrieval: {str(e)}")
                st.code(str(e))
                st.info("💡 If you're seeing import errors, try restarting the Streamlit app.")

    elif test_button and not query.strip():
        st.warning("⚠️ Please enter a query to test retrieval.")

    # Instructions
    if not test_button:
        st.info("💡 **How to use this debug tool:**")
        st.markdown("""
        1. **Enter a query** in Korean in the sidebar
        2. **Choose retrieval type** (Sparse, Dense, or Hybrid)
        3. **Adjust parameters** as needed
        4. **Enable debug logging** for detailed output
        5. **Click "Run Retrieval Test"** to see results

        **Retrieval Types:**
        - **Sparse (BM25)**: Traditional text matching using TF-IDF and BM25 scoring
        - **Dense (Embedding)**: Semantic search using vector embeddings and cosine similarity
        - **Hybrid**: Combines both sparse and dense retrieval for better results
        """)

        # Show status
        if RETRIEVAL_AVAILABLE:
            st.success("✅ Retrieval modules are loaded and ready.")
        else:
            st.error("❌ Retrieval modules failed to load. Check the terminal for import errors.")

        # Example queries
        st.subheader("📝 Example Queries")
        example_queries = [
            "통학 버스의 가치",
            "인공지능의 발전",
            "환경 보호 방법",
            "학교 교육의 중요성"
        ]

        cols = st.columns(2)
        for i, ex_query in enumerate(example_queries):
            if cols[i % 2].button(f"Try: {ex_query}", key=f"example_{i}"):
                st.session_state.query = ex_query
                st.rerun()

def run_sparse_retrieval(query, size, enable_debug, show_full_content):
    """Run sparse retrieval test with detailed output."""
    if not RETRIEVAL_AVAILABLE or sparse_retrieve is None:
        st.error("Retrieval modules not available")
        return

    st.subheader("🔍 Sparse Retrieval Results (BM25)")

    # Show query info
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Query:** {query}")
    with col2:
        st.markdown(f"**Results:** {size}")

    # Perform retrieval
    results = sparse_retrieve(query, size=size)

    if not results:
        st.warning("No results found.")
        return

    st.success(f"✅ Retrieved {len(results)} documents")

    # Display results
    for i, hit in enumerate(results):
        with st.expander(f"📄 Document {i+1}", expanded=(i < 3)):
            source = hit.get('_source', {})
            doc_id = hit.get('_id', 'N/A')
            bm25_score = hit.get('_score', 0.0)

            # Header with key metrics
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"**Document ID:** {doc_id}")
            with col2:
                st.markdown(f"**BM25 Score:** {bm25_score:.4f}")
            with col3:
                st.markdown(f"**Rank:** {i+1}")

            # Content
            content = source.get('content', '')
            if content:
                if show_full_content:
                    st.text_area("Full Content", content, height=200, key=f"sparse_content_{i}")
                else:
                    preview_length = 300
                    preview = content[:preview_length] + "..." if len(content) > preview_length else content
                    st.markdown("**Content Preview:**")
                    st.write(preview)

                    if len(content) > preview_length:
                        if st.button(f"Show Full Content ({len(content)} chars)", key=f"sparse_full_{i}"):
                            st.text_area("Full Content", content, height=300, key=f"sparse_full_content_{i}")
            else:
                st.warning("No content available")

    # Debug logging if enabled
    if enable_debug:
        st.subheader("🔧 Debug Information")
        with st.expander("Detailed Scores & Analysis", expanded=False):
            logger = ConfidenceLogger(debug_mode=True) if ConfidenceLogger is not None else None

            # Create a list to capture debug information
            debug_info = []

            for i, hit in enumerate(results):
                source = hit.get('_source', {})
                doc_id = hit.get('_id', 'N/A')
                bm25_score = hit.get('_score', 0.0)

                retrieval_scores = {
                    'bm25_score': bm25_score,
                    'rank': i + 1,
                    'total_results': len(results)
                }

                # Capture debug info instead of just logging
                debug_entry = {
                    'rank': i + 1,
                    'doc_id': doc_id,
                    'bm25_score': bm25_score,
                    'has_content': bool(source.get('content')),
                    'content_length': len(source.get('content', ''))
                }
                debug_info.append(debug_entry)

                if logger is not None:
                    logger.log_confidence_score(
                        technique='sparse_retrieval',
                        confidence=min(bm25_score / 100.0, 1.0),
                        query=query,
                        reasoning=f"BM25 retrieval result #{i+1}",
                        retrieval_scores=retrieval_scores,
                        context={
                            'doc_id': doc_id,
                            'has_content': bool(source.get('content')),
                            'content_length': len(source.get('content', ''))
                        }
                    )

            # Display debug information in the UI
            if debug_info:
                debug_df = pd.DataFrame(debug_info)
                st.dataframe(debug_df)
                st.info("📝 Debug logs are also printed to the terminal where Streamlit is running.")
            else:
                st.warning("No debug information available.")

def run_dense_retrieval(query, size, enable_debug, show_full_content):
    """Run dense retrieval test with detailed output."""
    if not RETRIEVAL_AVAILABLE or dense_retrieve is None or encode_texts is None:
        st.error("Retrieval modules not available")
        return

    st.subheader("🔍 Dense Retrieval Results (Embeddings)")

    # Show query info
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Query:** {query}")
    with col2:
        st.markdown(f"**Results:** {size}")

    # Get query embedding
    with st.spinner("Encoding query..."):
        q_emb = encode_texts([query])[0]
        st.info(f"✅ Query encoded - Embedding shape: {q_emb.shape}")

    # Perform retrieval
    results = dense_retrieve(q_emb, size=size)

    if not results:
        st.warning("No results found.")
        return

    st.success(f"✅ Retrieved {len(results)} documents")

    # Display results
    for i, hit in enumerate(results):
        with st.expander(f"📄 Document {i+1}", expanded=(i < 3)):
            source = hit.get('_source', {})
            doc_id = hit.get('_id', 'N/A')
            cosine_score = hit.get('_score', 0.0)

            # Header with key metrics
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"**Document ID:** {doc_id}")
            with col2:
                st.markdown(f"**Cosine Score:** {cosine_score:.4f}")
            with col3:
                st.markdown(f"**Rank:** {i+1}")

            # Content
            content = source.get('content', '')
            if content:
                if show_full_content:
                    st.text_area("Full Content", content, height=200, key=f"dense_content_{i}")
                else:
                    preview_length = 300
                    preview = content[:preview_length] + "..." if len(content) > preview_length else content
                    st.markdown("**Content Preview:**")
                    st.write(preview)

                    if len(content) > preview_length:
                        if st.button(f"Show Full Content ({len(content)} chars)", key=f"dense_full_{i}"):
                            st.text_area("Full Content", content, height=300, key=f"dense_full_content_{i}")
            else:
                st.warning("No content available")

    # Debug logging if enabled
    if enable_debug:
        st.subheader("🔧 Debug Information")
        with st.expander("Detailed Scores & Analysis", expanded=False):
            logger = ConfidenceLogger(debug_mode=True) if ConfidenceLogger is not None else None

            # Create a list to capture debug information
            debug_info = []

            for i, hit in enumerate(results):
                source = hit.get('_source', {})
                doc_id = hit.get('_id', 'N/A')
                cosine_score = hit.get('_score', 0.0)

                retrieval_scores = {
                    'cosine_score': cosine_score,
                    'rank': i + 1,
                    'total_results': len(results),
                    'embedding_norm': np.linalg.norm(q_emb)
                }

                # Capture debug info instead of just logging
                debug_entry = {
                    'rank': i + 1,
                    'doc_id': doc_id,
                    'cosine_score': cosine_score,
                    'embedding_norm': np.linalg.norm(q_emb),
                    'has_content': bool(source.get('content')),
                    'content_length': len(source.get('content', ''))
                }
                debug_info.append(debug_entry)

                if logger is not None:
                    logger.log_confidence_score(
                        technique='dense_retrieval',
                        confidence=min(abs(cosine_score), 1.0),
                        query=query,
                        reasoning=f"Dense retrieval result #{i+1}",
                        retrieval_scores=retrieval_scores,
                        context={
                            'doc_id': doc_id,
                            'has_content': bool(source.get('content')),
                            'content_length': len(source.get('content', ''))
                        }
                    )

            # Display debug information in the UI
            if debug_info:
                debug_df = pd.DataFrame(debug_info)
                st.dataframe(debug_df)
                st.info("📝 Debug logs are also printed to the terminal where Streamlit is running.")
            else:
                st.warning("No debug information available.")

def run_hybrid_retrieval(query, bm25_k, rerank_k, alpha, enable_debug, show_full_content):
    """Run hybrid retrieval test with detailed output."""
    if not RETRIEVAL_AVAILABLE or hybrid_retrieve is None:
        st.error("Retrieval modules not available")
        return

    st.subheader("🔍 Hybrid Retrieval Results")

    # Show query info
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.markdown(f"**Query:** {query}")
    with col2:
        st.markdown(f"**BM25 K:** {bm25_k}")
    with col3:
        st.markdown(f"**Rerank K:** {rerank_k}")
    with col4:
        st.markdown(f"**Alpha:** {alpha}")

    # Perform retrieval
    results = hybrid_retrieve(query, bm25_k=bm25_k, rerank_k=rerank_k, alpha=alpha)

    if not results:
        st.warning("No results found.")
        return

    st.success(f"✅ Retrieved {len(results)} documents")

    # Display results
    for i, result in enumerate(results):
        with st.expander(f"📄 Document {i+1}", expanded=(i < 3)):
            hit = result.get('hit', {})
            source = hit.get('_source', {})
            doc_id = hit.get('_id', 'N/A')
            final_score = result.get('score', 0.0)
            bm25_score = hit.get('_score', 0.0)
            cosine_score = result.get('cosine', 0.0)

            # Header with key metrics
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
            with col1:
                st.markdown(f"**Document ID:** {doc_id}")
            with col2:
                st.markdown(f"**Final Score:** {final_score:.4f}")
            with col3:
                st.markdown(f"**BM25:** {bm25_score:.4f}")
            with col4:
                st.markdown(f"**Cosine:** {cosine_score:.4f}")
            with col5:
                st.markdown(f"**Rank:** {i+1}")

            # Content
            content = source.get('content', '')
            if content:
                if show_full_content:
                    st.text_area("Full Content", content, height=200, key=f"hybrid_content_{i}")
                else:
                    preview_length = 300
                    preview = content[:preview_length] + "..." if len(content) > preview_length else content
                    st.markdown("**Content Preview:**")
                    st.write(preview)

                    if len(content) > preview_length:
                        if st.button(f"Show Full Content ({len(content)} chars)", key=f"hybrid_full_{i}"):
                            st.text_area("Full Content", content, height=300, key=f"hybrid_full_content_{i}")
            else:
                st.warning("No content available")

    # Debug logging if enabled
    if enable_debug:
        st.subheader("🔧 Debug Information")
        with st.expander("Detailed Scores & Analysis", expanded=False):
            logger = ConfidenceLogger(debug_mode=True) if ConfidenceLogger is not None else None

            # Create a list to capture debug information
            debug_info = []

            for i, result in enumerate(results):
                hit = result.get('hit', {})
                source = hit.get('_source', {})
                doc_id = hit.get('_id', 'N/A')
                final_score = result.get('score', 0.0)
                bm25_score = hit.get('_score', 0.0)
                cosine_score = result.get('cosine', 0.0)

                retrieval_scores = {
                    'final_score': final_score,
                    'bm25_score': bm25_score,
                    'cosine_score': cosine_score,
                    'rank': i + 1,
                    'total_results': len(results),
                    'alpha': alpha
                }

                # Capture debug info instead of just logging
                debug_entry = {
                    'rank': i + 1,
                    'doc_id': doc_id,
                    'final_score': final_score,
                    'bm25_score': bm25_score,
                    'cosine_score': cosine_score,
                    'alpha': alpha,
                    'has_content': bool(source.get('content')),
                    'content_length': len(source.get('content', ''))
                }
                debug_info.append(debug_entry)

                if logger is not None:
                    logger.log_confidence_score(
                        technique='hybrid_retrieval',
                        confidence=min(final_score, 1.0),
                        query=query,
                        reasoning=f"Hybrid retrieval result #{i+1} (BM25 + Dense)",
                        retrieval_scores=retrieval_scores,
                        context={
                            'doc_id': doc_id,
                            'has_content': bool(source.get('content')),
                            'content_length': len(source.get('content', ''))
                        }
                    )

            # Display debug information in the UI
            if debug_info:
                debug_df = pd.DataFrame(debug_info)
                st.dataframe(debug_df)
                st.info("📝 Debug logs are also printed to the terminal where Streamlit is running.")
            else:
                st.warning("No debug information available.")

if __name__ == "__main__":
    main()
