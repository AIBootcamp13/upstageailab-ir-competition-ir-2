#!/usr/bin/env python3
"""
Quick Integration Demo: Using Phase 1 Profiling Insights
Demonstrates how to load and use profiling results for retrieval optimization
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any

def load_profiling_insights():
    """Load Phase 1 profiling results for integration"""
    base_path = Path("outputs/reports/data_profile/latest")

    insights = {}

    # Load long document analysis
    with open(base_path / "long_doc_analysis.json", 'r', encoding='utf-8') as f:
        insights['long_doc'] = json.load(f)

    # Load source clusters
    with open(base_path / "src_clusters_by_vocab.json", 'r', encoding='utf-8') as f:
        insights['clusters'] = json.load(f)

    # Load per-source stats
    with open(base_path / "per_src_length_stats.json", 'r', encoding='utf-8') as f:
        insights['per_src_stats'] = json.load(f)

    return insights

def get_chunking_recommendation(src: str, insights: Dict) -> Dict[str, Any]:
    """Get chunking recommendations based on profiling"""
    if src not in insights['long_doc']['per_source']:
        return {"chunk_size": 512, "overlap": 50}  # Default

    long_doc_info = insights['long_doc']['per_source'][src]
    long_fraction = long_doc_info['long_doc_fraction']

    # Adjust chunk size based on long document fraction
    if long_fraction > 0.12:  # High long-doc fraction
        chunk_size = 768
        overlap = 100
    elif long_fraction > 0.08:  # Medium long-doc fraction
        chunk_size = 640
        overlap = 75
    else:  # Low long-doc fraction
        chunk_size = 512
        overlap = 50

    return {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "long_doc_fraction": long_fraction,
        "recommendation": f"Optimized for {long_fraction:.1%} long documents"
    }

def get_domain_cluster(src: str, insights: Dict) -> Dict[str, Any]:
    """Find which cluster a source belongs to"""
    for cluster in insights['clusters']['clusters']:
        if src in cluster['sources']:
            return {
                "cluster_id": cluster['id'],
                "cluster_size": cluster['size'],
                "common_terms": cluster['common_terms'][:5],
                "related_sources": cluster['sources'][:10]  # Top 10 related
            }
    return {"cluster_id": None, "error": "Source not found in clusters"}

def demo_integration():
    """Demonstrate profiling insights integration"""
    print("🔍 Phase 1 Profiling Insights Integration Demo")
    print("=" * 50)

    try:
        insights = load_profiling_insights()
        print("✅ Successfully loaded profiling insights")

        # Demo 1: Chunking optimization
        print("\n📏 Chunking Optimization Demo:")
        test_sources = ["ko_ai2_arc__ARC_Challenge__test", "ko_mmlu__anatomy__test"]
        for src in test_sources:
            rec = get_chunking_recommendation(src, insights)
            print(f"  {src}:")
            print(f"    Chunk Size: {rec['chunk_size']} tokens")
            print(f"    Overlap: {rec['overlap']} tokens")
            print(f"    Long Doc Fraction: {rec['long_doc_fraction']:.1%}")

        # Demo 2: Domain clustering
        print("\n🏷️  Domain Clustering Demo:")
        for src in test_sources:
            cluster_info = get_domain_cluster(src, insights)
            print(f"  {src}:")
            print(f"    Cluster ID: {cluster_info['cluster_id']}")
            print(f"    Cluster Size: {cluster_info['cluster_size']} sources")
            print(f"    Common Terms: {', '.join(cluster_info['common_terms'][:3])}")

        print("\n🎯 Integration Ready!")
        print("Use these insights in your retrieval pipeline:")
        print("1. get_chunking_recommendation() for dynamic chunk sizing")
        print("2. get_domain_cluster() for query routing")
        print("3. Load insights at startup for real-time optimization")

    except Exception as e:
        print(f"❌ Error loading insights: {e}")
        print("Make sure Phase 1 profiling has been run successfully")

if __name__ == "__main__":
    demo_integration()