# src/ir_core/analysis/core.py

"""
Core analysis classes and data structures for the Scientific QA retrieval system.
"""

from typing import Dict, List, Any, Optional, Union, cast
from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .metrics import RetrievalMetrics
from .query_analyzer import QueryAnalyzer


@dataclass
class QueryAnalysis:
    """Analysis results for a single query."""
    original_query: str
    rewritten_query: str
    query_length: int
    domain: List[str]  # Support multiple domains
    complexity_score: float
    processing_time: float
    rewrite_effective: bool


@dataclass
class RetrievalResult:
    """Results from a single retrieval operation."""
    query: str
    ground_truth_id: str
    predicted_ids: List[str]
    predicted_scores: List[float]
    ap_score: float
    rank_of_ground_truth: Optional[int]
    top_k_precision: Dict[int, float]
    retrieval_time: float


@dataclass
class AnalysisResult:
    """Comprehensive analysis results for a batch of queries."""
    # Basic metrics
    map_score: float
    mean_ap: float
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]

    # Query analysis
    total_queries: int
    avg_query_length: float
    rewrite_rate: float
    domain_distribution: Dict[str, int]

    # Retrieval analysis
    retrieval_success_rate: float
    avg_retrieval_time: float
    error_categories: Dict[str, int]

    # Detailed results
    query_analyses: List[QueryAnalysis] = field(default_factory=list)
    retrieval_results: List[RetrievalResult] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    timestamp: float = field(default_factory=time.time)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)


class RetrievalAnalyzer:
    """
    Main analysis orchestrator for Scientific QA retrieval evaluation.

    This class coordinates various analysis components to provide comprehensive
    insights into retrieval performance.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the retrieval analyzer.

        Args:
            config: Hydra configuration object containing analysis settings
        """
        self.config = config
        self.metrics_calculator = RetrievalMetrics()
        self.query_analyzer = QueryAnalyzer(config)

        # Parallel processing configuration
        self.max_workers = config.get('analysis', {}).get('max_workers', None)
        self.enable_parallel = config.get('analysis', {}).get('enable_parallel', True)

    def analyze_batch(
        self,
        queries: List[Dict[str, Any]],
        retrieval_results: List[Dict[str, Any]],
        rewritten_queries: Optional[List[str]] = None,
        max_workers: Optional[int] = None
    ) -> AnalysisResult:
        """
        Perform comprehensive analysis on a batch of retrieval results with optional parallel processing.

        Args:
            queries: List of query dictionaries with original queries
            retrieval_results: List of retrieval result dictionaries
            rewritten_queries: Optional list of rewritten queries
            max_workers: Maximum number of worker threads for parallel processing

        Returns:
            AnalysisResult: Comprehensive analysis results
        """
        start_time = time.time()

        # Extract query information
        original_queries = [q.get("msg", [{}])[0].get("content", "") for q in queries]
        ground_truth_ids = [q.get("ground_truth_doc_id", "") for q in queries]

        # Extract retrieval results
        predicted_docs_list = []
        for result in retrieval_results:
            if result and "docs" in result:
                predicted_docs_list.append(result["docs"])
            else:
                predicted_docs_list.append([])

        # Prepare rewritten queries
        if rewritten_queries is None:
            rewritten_queries = original_queries

        # Calculate basic metrics with optional parallel processing
        all_results_for_map = []
        ap_scores = []

        if len(predicted_docs_list) > 10 and self.enable_parallel and max_workers != 0:
            # Use parallel processing for metric calculation
            if max_workers is None:
                max_workers = self.max_workers or min(16, len(predicted_docs_list))

            print(f"🔄 Calculating metrics for {len(predicted_docs_list)} queries using {max_workers} parallel workers...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit metric calculation tasks
                future_to_index = {}
                for i, (pred_docs, gt_id) in enumerate(zip(predicted_docs_list, ground_truth_ids)):
                    future = executor.submit(self._calculate_query_metrics, pred_docs, gt_id, i)
                    future_to_index[future] = i

                # Collect results in order
                results_by_index = {}
                for future in as_completed(future_to_index):
                    try:
                        index, pred_ids, ap_score = future.result()
                        results_by_index[index] = (pred_ids, ap_score)
                    except Exception as e:
                        index = future_to_index[future]
                        print(f"Error calculating metrics for query {index}: {e}")
                        results_by_index[index] = ([], 0.0)

                # Sort results back to original order
                for i in range(len(predicted_docs_list)):
                    if i in results_by_index:
                        pred_ids, ap_score = results_by_index[i]
                        all_results_for_map.append((pred_ids, [ground_truth_ids[i]]))
                        ap_scores.append(ap_score)
        else:
            # Sequential processing
            for i, (pred_docs, gt_id) in enumerate(zip(predicted_docs_list, ground_truth_ids)):
                pred_ids = [doc.get("id", "") for doc in pred_docs]
                relevant_ids = [gt_id]

                all_results_for_map.append((pred_ids, relevant_ids))

                # Calculate AP for this query
                ap_score = self.metrics_calculator.average_precision(pred_ids, relevant_ids)
                ap_scores.append(ap_score if ap_score is not None else 0.0)

        # Calculate overall metrics
        map_score = self.metrics_calculator.mean_average_precision(all_results_for_map)
        mean_ap = sum(ap_scores) / len(ap_scores) if ap_scores else 0.0

        # Calculate precision@K
        precision_at_k = {}
        for k in [1, 3, 5, 10]:
            precision_at_k[k] = self.metrics_calculator.precision_at_k(all_results_for_map, k)

        # Calculate recall@K (simplified version)
        recall_at_k = {}
        for k in [1, 3, 5, 10]:
            recall_at_k[k] = self.metrics_calculator.recall_at_k(all_results_for_map, k)

        # Enhanced query analysis using QueryAnalyzer with parallel processing
        query_features_list = self.query_analyzer.analyze_batch(original_queries, max_workers)
        total_queries = len(original_queries)

        # Calculate aggregate statistics from features
        avg_query_length = sum(f.length for f in query_features_list) / total_queries if total_queries > 0 else 0
        avg_query_complexity = sum(f.complexity_score for f in query_features_list) / total_queries if total_queries > 0 else 0

        # Rewrite analysis
        rewrite_changes = sum(1 for orig, rew in zip(original_queries, rewritten_queries) if orig != rew)
        rewrite_rate = rewrite_changes / total_queries if total_queries > 0 else 0

        # Domain distribution from QueryAnalyzer features
        domain_distribution = {}
        for features in query_features_list:
            for domain in features.domain:
                domain_distribution[domain] = domain_distribution.get(domain, 0) + 1

        # Retrieval analysis
        retrieval_success_count = sum(1 for pred_docs, gt_id in zip(predicted_docs_list, ground_truth_ids)
                                    if any(doc.get("id") == gt_id for doc in pred_docs[:10]))
        retrieval_success_rate = retrieval_success_count / total_queries if total_queries > 0 else 0

        # Error categorization based on retrieval failures
        error_categories = self._analyze_error_categories(
            predicted_docs_list, ground_truth_ids, original_queries
        )

        # Create detailed results
        query_analyses = []
        retrieval_results_detailed = []

        for i, (orig_q, rew_q, gt_id, pred_docs, features) in enumerate(zip(
            original_queries, rewritten_queries, ground_truth_ids, predicted_docs_list, query_features_list
        )):
            # Query analysis using QueryAnalyzer features
            query_analysis = QueryAnalysis(
                original_query=orig_q,
                rewritten_query=rew_q,
                query_length=features.length,
                domain=features.domain,
                complexity_score=features.complexity_score,
                processing_time=0.0,  # Placeholder
                rewrite_effective=orig_q != rew_q
            )
            query_analyses.append(query_analysis)

            # Retrieval result details
            pred_ids = [doc.get("id", "") for doc in pred_docs]
            pred_scores = [doc.get("score", 0.0) for doc in pred_docs]

            retrieval_result = RetrievalResult(
                query=orig_q,
                ground_truth_id=gt_id,
                predicted_ids=pred_ids,
                predicted_scores=pred_scores,
                ap_score=ap_scores[i],
                rank_of_ground_truth=self._find_rank_of_ground_truth(pred_ids, gt_id),
                top_k_precision=self._calculate_top_k_precision(pred_ids, [gt_id]),
                retrieval_time=0.0  # Placeholder
            )
            retrieval_results_detailed.append(retrieval_result)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            map_score, retrieval_success_rate, rewrite_rate
        )

        analysis_time = time.time() - start_time
        print(".2f")

        return AnalysisResult(
            map_score=map_score,
            mean_ap=mean_ap,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            total_queries=total_queries,
            avg_query_length=avg_query_length,
            rewrite_rate=rewrite_rate,
            domain_distribution=domain_distribution,
            retrieval_success_rate=retrieval_success_rate,
            avg_retrieval_time=0.0,  # Placeholder
            error_categories=error_categories,
            query_analyses=query_analyses,
            retrieval_results=retrieval_results_detailed,
            recommendations=recommendations,
            timestamp=time.time(),
            config_snapshot=cast(Dict[str, Any], OmegaConf.to_container(self.config)) if self.config else {}
        )

    def _calculate_query_metrics(self, pred_docs: List[Dict[str, Any]], gt_id: str, index: int):
        """
        Calculate metrics for a single query (for parallel processing).

        Args:
            pred_docs: Predicted documents
            gt_id: Ground truth document ID
            index: Query index

        Returns:
            Tuple[int, List[str], float]: (index, predicted_ids, ap_score)
        """
        pred_ids = [doc.get("id", "") for doc in pred_docs]
        relevant_ids = [gt_id]

        # Calculate AP for this query
        ap_score = self.metrics_calculator.average_precision(pred_ids, relevant_ids)
        ap_score = ap_score if ap_score is not None else 0.0

        return index, pred_ids, ap_score

    def _find_rank_of_ground_truth(self, pred_ids: List[str], gt_id: str) -> Optional[int]:
        """
        Find the rank of ground truth document in predicted results.

        Args:
            pred_ids: List of predicted document IDs
            gt_id: Ground truth document ID

        Returns:
            Optional[int]: Rank of ground truth (1-based), None if not found
        """
        try:
            return pred_ids.index(gt_id) + 1
        except ValueError:
            return None

    def _calculate_top_k_precision(self, pred_ids: List[str], relevant_ids: List[str]) -> Dict[int, float]:
        """
        Calculate precision@K for different K values.

        Args:
            pred_ids: List of predicted document IDs
            relevant_ids: List of relevant document IDs

        Returns:
            Dict[int, float]: Precision values for K=1,3,5,10
        """
        precision_at_k = {}
        relevant_set = set(relevant_ids)

        for k in [1, 3, 5, 10]:
            if len(pred_ids) >= k:
                top_k_preds = pred_ids[:k]
                correct = len([pid for pid in top_k_preds if pid in relevant_set])
                precision_at_k[k] = correct / k
            else:
                precision_at_k[k] = 0.0

        return precision_at_k

        # Generate recommendations
        recommendations = self._generate_recommendations(
            map_score, retrieval_success_rate, rewrite_rate
        )

        # Create final result
        analysis_result = AnalysisResult(
            map_score=map_score,
            mean_ap=mean_ap,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            total_queries=total_queries,
            avg_query_length=avg_query_length,
            rewrite_rate=rewrite_rate,
            domain_distribution=domain_distribution,
            retrieval_success_rate=retrieval_success_rate,
            avg_retrieval_time=0.0,  # Placeholder
            error_categories=error_categories,
            query_analyses=query_analyses,
            retrieval_results=retrieval_results_detailed,
            recommendations=recommendations,
            config_snapshot={}
        )

        analysis_time = time.time() - start_time
        print(f"Analysis completed in {analysis_time:.2f} seconds")
        return analysis_result

    def _calculate_query_complexity(self, query: str) -> float:
        """
        Calculate a simple query complexity score based on length and scientific terms.

        Args:
            query: The query string

        Returns:
            float: Complexity score between 0 and 1
        """
        # Simple heuristic: longer queries with scientific terms are more complex
        length_score = min(len(query) / 100, 1.0)  # Normalize to 0-1

        scientific_terms = [
            '원자', '분자', '세포', '유전자', '단백질', 'RNA', 'DNA',
            '화합물', '반응', '에너지', '힘', '운동', '속도', '질량'
        ]

        term_count = sum(1 for term in scientific_terms if term in query)
        term_score = min(term_count / 5, 1.0)  # Normalize to 0-1

        return (length_score + term_score) / 2

    def _generate_recommendations(
        self,
        map_score: float,
        retrieval_success_rate: float,
        rewrite_rate: float
    ) -> List[str]:
        """
        Generate actionable recommendations based on analysis results.

        Args:
            map_score: Mean Average Precision
            retrieval_success_rate: Percentage of successful retrievals
            rewrite_rate: Percentage of queries that were rewritten

        Returns:
            List[str]: List of recommendations
        """
        recommendations = []

        if map_score < 0.5:
            recommendations.append("MAP score is below 0.5. Consider improving retrieval algorithm or expanding document collection.")

        if retrieval_success_rate < 0.7:
            recommendations.append("Retrieval success rate is below 70%. Focus on improving top-10 retrieval accuracy.")

        if rewrite_rate > 0.8:
            recommendations.append("High rewrite rate detected. Verify that query rewriting is improving rather than degrading performance.")

        if rewrite_rate < 0.1:
            recommendations.append("Low rewrite rate. Consider enabling query rewriting to improve retrieval for conversational queries.")

        if not recommendations:
            recommendations.append("Overall performance looks good. Consider fine-tuning hyperparameters for marginal improvements.")

        return recommendations

    def _analyze_domain_distribution(self, queries: List[str]) -> Dict[str, int]:
        """
        Analyze and categorize queries by scientific domain.

        Args:
            queries: List of query strings

        Returns:
            Dict[str, int]: Distribution of queries by domain
        """
        domain_keywords = {
            "physics": ["물리", "힘", "에너지", "운동", "속도", "질량", "전자", "원자", "분자", "반응", "화합물"],
            "biology": ["생물", "세포", "유전자", "단백질", "RNA", "DNA", "미생물", "생태", "진화"],
            "chemistry": ["화학", "원소", "화합물", "반응", "결합", "용액", "산", "염기", "pH"],
            "astronomy": ["천문", "별", "행성", "은하", "우주", "태양", "달", "지구"],
            "geology": ["지질", "암석", "광물", "지층", "화산", "지진", "대륙"],
            "general": ["과학", "연구", "실험", "관찰", "측정", "계산"]
        }

        distribution = {domain: 0 for domain in domain_keywords.keys()}
        distribution["unknown"] = 0

        for query in queries:
            query_lower = query.lower()
            matched_domains = []

            for domain, keywords in domain_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    matched_domains.append(domain)

            if matched_domains:
                # Use the first matched domain, or "general" if multiple matches
                assigned_domain = matched_domains[0] if len(matched_domains) == 1 else "general"
                distribution[assigned_domain] += 1
            else:
                distribution["unknown"] += 1

        # Remove domains with zero counts
        return {k: v for k, v in distribution.items() if v > 0}

    def _analyze_error_categories(self, predicted_docs_list: List[List[Dict]], ground_truth_ids: List[str],
                                original_queries: List[str]) -> Dict[str, int]:
        """
        Analyze retrieval errors and categorize them.

        Args:
            predicted_docs_list: List of predicted documents for each query
            ground_truth_ids: List of ground truth document IDs
            original_queries: List of original query strings

        Returns:
            Dict[str, int]: Distribution of error categories
        """
        error_categories = {
            "no_retrieval": 0,  # No documents retrieved
            "wrong_domain": 0,  # Retrieved docs from wrong scientific domain
            "low_relevance": 0,  # Retrieved docs have low relevance scores
            "query_mismatch": 0,  # Query and retrieved docs don't match semantically
            "successful": 0      # Successful retrievals
        }

        for pred_docs, gt_id, query in zip(predicted_docs_list, ground_truth_ids, original_queries):
            if not pred_docs:
                error_categories["no_retrieval"] += 1
                continue

            # Check if ground truth is in top 10 results
            top_10_ids = [doc.get("id", "") for doc in pred_docs[:10]]
            if gt_id in top_10_ids:
                error_categories["successful"] += 1
                continue

            # Analyze failure reasons
            query_lower = query.lower()

            # Check for domain mismatch (simplified heuristic)
            retrieved_domains = set()
            for doc in pred_docs[:5]:  # Check top 5 results
                doc_content = doc.get("content", "").lower()[:200]  # First 200 chars
                if any(term in doc_content for term in ["물리", "힘", "에너지"]):
                    retrieved_domains.add("physics")
                elif any(term in doc_content for term in ["생물", "세포", "유전자"]):
                    retrieved_domains.add("biology")
                elif any(term in doc_content for term in ["화학", "원소", "화합물"]):
                    retrieved_domains.add("chemistry")

            query_domain = "unknown"
            if any(term in query_lower for term in ["물리", "힘", "에너지"]):
                query_domain = "physics"
            elif any(term in query_lower for term in ["생물", "세포", "유전자"]):
                query_domain = "biology"
            elif any(term in query_lower for term in ["화학", "원소", "화합물"]):
                query_domain = "chemistry"

            if query_domain != "unknown" and retrieved_domains and query_domain not in retrieved_domains:
                error_categories["wrong_domain"] += 1
            else:
                # Check relevance scores
                top_scores = [doc.get("score", 0.0) for doc in pred_docs[:3]]
                avg_top_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
                if avg_top_score < 0.3:  # Low relevance threshold
                    error_categories["low_relevance"] += 1
                else:
                    error_categories["query_mismatch"] += 1

        # Remove categories with zero counts
        return {k: v for k, v in error_categories.items() if v > 0}

    def _classify_query_domain(self, query: str) -> str:
        """
        Classify a single query into a scientific domain.

        Args:
            query: The query string

        Returns:
            str: The classified domain
        """
        query_lower = query.lower()

        domain_keywords = {
            "physics": ["물리", "힘", "에너지", "운동", "속도", "질량", "전자", "원자", "분자", "반응", "화합물"],
            "biology": ["생물", "세포", "유전자", "단백질", "RNA", "DNA", "미생물", "생태", "진화"],
            "chemistry": ["화학", "원소", "화합물", "반응", "결합", "용액", "산", "염기", "pH"],
            "astronomy": ["천문", "별", "행성", "은하", "우주", "태양", "달", "지구"],
            "geology": ["지질", "암석", "광물", "지층", "화산", "지진", "대륙"],
            "general": ["과학", "연구", "실험", "관찰", "측정", "계산"]
        }

        matched_domains = []
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                matched_domains.append(domain)

        if matched_domains:
            # Return the most specific domain, or "general" if multiple matches
            return matched_domains[0] if len(matched_domains) == 1 else "general"

        return "unknown"
