"""Evaluation metrics core implementations."""

def precision_at_k(retrieved, relevant, k):
    """Calculates precision at k."""
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    retrieved_k = retrieved[:k]
    return len(set(retrieved_k) & relevant_set) / k

def average_precision(retrieved, relevant):
    """
    Calculates Average Precision (AP) for a single query.
    AP is the average of P@k values for each relevant document found.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    hits = 0
    sum_precisions = 0
    for k, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant_set:
            hits += 1
            sum_precisions += precision_at_k(retrieved, relevant, k)

    if hits == 0:
        return 0.0

    return sum_precisions / hits

def mean_average_precision(results):
    """
    Calculates Mean Average Precision (MAP) for a set of queries.

    Args:
        results: A list of tuples, where each tuple is (retrieved_ids, relevant_ids).
    """
    if not results:
        return 0.0

    return sum(average_precision(retrieved, relevant) for retrieved, relevant in results) / len(results)

def mrr(retrieved, relevant):
    """Calculates Mean Reciprocal Rank (MRR)."""
    relevant_set = set(relevant)
    for i, d in enumerate(retrieved, start=1):
        if d in relevant_set:
            return 1.0 / i
    return 0.0

def mean_reciprocal_rank(results):
    """
    Calculates Mean Reciprocal Rank (MRR) for a set of queries.

    Args:
        results: A list of tuples, where each tuple is (retrieved_ids, relevant_ids).
    """
    if not results:
        return 0.0

    return sum(mrr(retrieved, relevant) for retrieved, relevant in results) / len(results)