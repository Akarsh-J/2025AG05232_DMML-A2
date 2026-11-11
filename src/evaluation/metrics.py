"""Ranking and rating metrics (pure functions, unit-tested)."""
import numpy as np


def precision_at_k(recs: list, relevant: set, k: int) -> float:
    return len(set(recs[:k]) & relevant) / k


def recall_at_k(recs: list, relevant: set, k: int) -> float:
    return len(set(recs[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recs: list, relevant: set, k: int) -> float:
    dcg = sum(1.0 / np.log2(pos + 2) for pos, it in enumerate(recs[:k]) if it in relevant)
    ideal = sum(1.0 / np.log2(pos + 2) for pos in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def coverage(all_recs: list[list], n_items: int) -> float:
    """Share of the catalogue that appears in at least one recommendation list."""
    return len({i for r in all_recs for i in r}) / n_items


def diversity(all_recs: list[list], item_category: dict) -> float:
    """Intra-list diversity: average share of item pairs in a list with DIFFERENT categories."""
    vals = []
    for recs in all_recs:
        cats = [item_category[i] for i in recs]
        n = len(cats)
        pairs = n * (n - 1) / 2
        if pairs:
            same = sum(cats[a] == cats[b] for a in range(n) for b in range(a + 1, n))
            vals.append(1 - same / pairs)
    return float(np.mean(vals))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))
