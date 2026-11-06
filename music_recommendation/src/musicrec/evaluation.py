"""Full-catalogue macro ranking metrics with explicit cold-target accounting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import topk_unseen


def ranking_metrics(ranked, relevant, k: int) -> dict[str, float]:
    if k < 1:
        raise ValueError("K must be positive")
    relevant = set(relevant)
    if not relevant:
        raise ValueError("Metric undefined for a user without held-out positives")
    ranked = list(ranked)[:k]
    if len(ranked) != len(set(ranked)):
        raise ValueError("Duplicate recommended items")
    hits = np.array([item in relevant for item in ranked], dtype=float)
    dcg = (hits / np.log2(np.arange(2, len(hits) + 2))).sum()
    idcg = (1 / np.log2(np.arange(2, min(k, len(relevant)) + 2))).sum()
    return {"precision": float(hits.sum() / k), "recall": float(hits.sum() / len(relevant)),
            "ndcg": float(dcg / idcg), "hit_rate": float(hits.any())}


def evaluate(train, heldout, user_index, item_index, popularity, scores=None, ks=(10, 20)):
    """scores=None is popularity; scores otherwise is user-item sparse/dense.

    Held-out data is used only for metric labels, never candidate filtering.
    Cold positives stay in every denominator, making their failures visible.
    """
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")
    rows, exposure = [], set()
    for uid, group in heldout.groupby("userID", sort=True):
        user = user_index[int(uid)]
        relevant = [item_index[int(v)] for v in group.artistID]
        seen = train.indices[train.indptr[user]:train.indptr[user + 1]]
        if set(relevant) & set(seen):
            raise ValueError("Evaluation labels overlap training history")
        if scores is None:
            vector = popularity
        elif hasattr(scores, "getrow"):
            vector = scores.getrow(user).toarray().ravel()
        else:
            vector = np.asarray(scores[user]).ravel()
        ranked = topk_unseen(vector, seen, popularity, max(10, max(ks)))
        row = {"userID": int(uid), "n_targets": len(relevant),
               "n_cold_targets": sum(popularity[v] == 0 for v in relevant),
               "train_degree": len(seen)}
        for k in ks:
            row.update({f"{name}@{k}": v for name, v in ranking_metrics(ranked, relevant, k).items()})
        exposure.update(ranked[:10].tolist())
        rows.append(row)
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("No evaluable users")
    summary = {c: float(detail[c].mean()) for c in detail if "@" in c}
    summary.update({"evaluated_users": len(detail), "targets": int(detail.n_targets.sum()),
                    "cold_targets": int(detail.n_cold_targets.sum()),
                    "catalogue_coverage@10": len(exposure) / train.shape[1]})
    return summary, detail


def bootstrap_mean_ci(values, seed=2026, samples=1000):
    """User bootstrap; this describes this split, not split/model uncertainty."""
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for i in range(samples):
        means[i] = rng.choice(values, len(values), replace=True).mean()
    lo, hi = np.quantile(means, [.025, .975])
    return {"mean": float(values.mean()), "lower_95": float(lo), "upper_95": float(hi),
            "samples": samples, "seed": seed}
