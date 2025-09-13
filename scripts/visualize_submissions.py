import streamlit as st
import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path

# Function to load JSONL data
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# Function to get submission files
def get_submission_files():
    outputs_dir = Path("/home/wb2x/workspace/information_retrieval_rag/outputs")
    files = list(outputs_dir.glob("submission*.jsonl")) + list(outputs_dir.glob("submission*.csv")) + list(outputs_dir.glob("sample_submission*.jsonl")) + list(outputs_dir.glob("sample_submission*.csv"))
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
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='font-size: 32px;'>RAG Submission Visualizer</h1>", unsafe_allow_html=True)
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
        df = load_jsonl(selected_file)
        st.sidebar.success(f"Loaded {len(df)} entries from {Path(selected_file).name}")

        df2 = None
        if compare_file and compare_file != selected_file:
            df2 = load_jsonl(compare_file)
            st.sidebar.success(f"Loaded comparison: {Path(compare_file).name}")

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
            else:
                st.info("Check 'Load Documents' in the sidebar to view the documents.")

if __name__ == "__main__":
    main()
