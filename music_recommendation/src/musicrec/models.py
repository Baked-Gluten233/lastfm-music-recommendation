"""Popularity and directed top-neighbour item cosine CF; no held-out inputs."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags


def popularity_scores(train: csr_matrix) -> np.ndarray:
    """Number of distinct TRAIN users per artist, not play-count sum."""
    return np.asarray(train.sum(axis=0)).ravel()


def item_cosine(train: csr_matrix, shrinkage: float = 0.0) -> csr_matrix:
    """cos(i,j) * common(i,j)/(common(i,j)+shrinkage), diagonal zero."""
    if shrinkage < 0:
        raise ValueError("shrinkage must be nonnegative")
    common = (train.T @ train).astype(np.float64).tocsr()
    common.setdiag(0)
    common.eliminate_zeros()
    norms = np.sqrt(popularity_scores(train))
    inverse = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    if shrinkage:
        common.data *= common.data / (common.data + shrinkage)
    similarity = (diags(inverse) @ common @ diags(inverse)).tocsr()
    similarity.sort_indices()
    return similarity


def retain_neighbors(similarity: csr_matrix, neighbors: int) -> csr_matrix:
    """Each source artist keeps its strongest N outgoing neighbours.

    Ties use catalogue column index (sorted original artist ID).
    Directed truncation means X @ S propagates a user's TRAIN artist signals.
    """
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    indices, values, indptr = [], [], [0]
    for row in range(similarity.shape[0]):
        a, b = similarity.indptr[row:row + 2]
        cols, scores = similarity.indices[a:b], similarity.data[a:b]
        select = np.lexsort((cols, -scores))[:neighbors]
        indices.extend(cols[select])
        values.extend(scores[select])
        indptr.append(len(indices))
    result = csr_matrix((values, indices, indptr), shape=similarity.shape)
    result.sort_indices()
    return result


def topk_unseen(scores: np.ndarray, seen: np.ndarray, popularity: np.ndarray, k: int) -> np.ndarray:
    """Rank ALL catalogue items; tie break by train popularity then artist ID."""
    if k < 1 or len(scores) != len(popularity):
        raise ValueError("invalid ranking dimensions or K")
    available = np.ones(len(scores), dtype=bool)
    available[seen] = False
    candidates = np.flatnonzero(available)
    if not np.isfinite(scores[candidates]).all():
        raise ValueError("nonfinite candidate score")
    order = np.lexsort((candidates, -popularity[candidates], -scores[candidates]))
    return candidates[order[:k]]
