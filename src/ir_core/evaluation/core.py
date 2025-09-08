"""Evaluation metrics core implementations."""


def precision_at_k(retrieved, relevant, k):
    if k <= 0:
        return 0.0
    rel_set = set(relevant)
    return sum(1 for d in retrieved[:k] if d in rel_set) / float(k)


def mrr(retrieved, relevant):
    rel_set = set(relevant)
    for i, d in enumerate(retrieved, start=1):
        if d in rel_set:
            return 1.0 / i
    return 0.0
