"""Phase 1: audit, split, train-only EDA and validation baselines. Never test."""
from __future__ import annotations

import json
import os
import platform
import sys
import time
import hashlib
from zipfile import ZipFile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ARCHIVE_SHA256, indexed_matrix, load_data, save_json, sha256, split_interactions
from .evaluation import bootstrap_mean_ci, evaluate
from .models import item_cosine, popularity_scores, retain_neighbors


def make_figures(train, popularity, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output / ".cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(train.groupby("userID").size(), bins=25, color="#31688e", edgecolor="white")
    ax.set(title="Training profile length", xlabel="Distinct artists per user", ylabel="Users")
    fig.tight_layout(); fig.savefig(output / "01_user_activity.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    values = np.sort(popularity[popularity > 0])[::-1]
    ax.loglog(np.arange(1, len(values) + 1), values, color="#35a78d")
    ax.set(title="Training artist popularity: long tail", xlabel="Artist rank (log scale)",
           ylabel="Distinct training listeners (log scale)")
    fig.tight_layout(); fig.savefig(output / "02_artist_long_tail.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log10(train.weight), bins=50, color="#7755a3", edgecolor="white")
    ax.set(title="Training listening-count distribution", xlabel="log10(cumulative listening count)",
           ylabel="User-artist pairs")
    fig.tight_layout(); fig.savefig(output / "03_listening_counts.png"); plt.close(fig)


def run_phase1(root: Path, seed=42, verbose=True):
    root = Path(root).resolve()
    raw = root / "data/raw/hetrec2011-lastfm-2k"
    output = root / "outputs"
    processed = root / "data/processed"
    for folder in (output, output / "figures", processed):
        folder.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    archive = root / "data/raw/hetrec2011-lastfm-2k.zip"
    if not archive.exists():
        raise FileNotFoundError("Run scripts/download_data.py before this experiment")
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Downloaded archive checksum mismatch; inspect the source")
    with ZipFile(archive) as zipped:
        for name in zipped.namelist():
            if sha256(raw / name) != hashlib.sha256(zipped.read(name)).hexdigest():
                raise ValueError(f"Extracted data differs from verified archive: {name}")
    frame, artists, audit = load_data(raw)
    save_json(output / "data_audit.json", audit)
    splits = split_interactions(frame, seed)
    for name, split in splits.items():
        split.to_csv(processed / f"{name}.csv", index=False, lineterminator="\n")
    # Test labels are written and checked for split integrity, then excluded from evaluation.
    test_count = len(splits["test"])
    del splits["test"]
    train, validation = splits["train"], splits["validation"]
    matrix, user_ids, artist_ids, user_index, item_index = indexed_matrix(train, artists)
    pd.DataFrame({"userID": user_ids, "user_index": np.arange(len(user_ids))}).to_csv(processed / "user_map.csv", index=False, lineterminator="\n")
    pd.DataFrame({"artistID": artist_ids, "item_index": np.arange(len(artist_ids))}).to_csv(processed / "artist_map.csv", index=False, lineterminator="\n")
    popularity = popularity_scores(matrix)
    manifest = {
        "seed": seed, "policy": "user degree >=5: floor(n/10), at least 1 for validation and test; remainder train; lower degree train only",
        "temporal": False, "train_rows": len(train), "validation_rows": len(validation), "test_rows": test_count,
        "train_users": len(user_ids), "validation_users": int(validation.userID.nunique()),
        "training_only_low_degree_users": int((frame.groupby('userID').size() < 5).sum()),
        "catalogue_size": len(artist_ids), "train_observed_artists": int((popularity > 0).sum()),
        "candidate_set": "all IDs in artists.dat minus the user's train history; no negative sampling; no validation/test-informed filtering",
        "primary_metric": "macro NDCG@10 on validation; cold targets retained",
        "tie_break": "score descending, TRAIN user-frequency descending, original artist ID ascending",
        "test_metrics_evaluated": False,
        "split_sha256": {s: sha256(processed / f"{s}.csv") for s in ("train", "validation", "test")},
    }
    save_json(output / "split_manifest.json", manifest)
    make_figures(train, popularity, output / "figures")
    experiments, details = [], {}
    fit_start = time.perf_counter()
    summary, detail = evaluate(matrix, validation, user_index, item_index, popularity)
    summary.update({"model": "popularity", "neighbors": 0, "shrinkage": 0.0,
                    "fit_seconds": 0.0, "evaluation_seconds": time.perf_counter() - fit_start})
    experiments.append(summary); details["popularity"] = detail
    detail.to_csv(output / "validation_users_popularity.csv", index=False)
    for shrinkage in (0.0, 10.0):
        similarity_start = time.perf_counter()
        similarity = item_cosine(matrix, shrinkage)
        similarity_seconds = time.perf_counter() - similarity_start
        for neighbors in (20, 50, 100):
            name = f"itemcf_n{neighbors}_s{int(shrinkage)}"
            fit_start = time.perf_counter()
            scores = matrix @ retain_neighbors(similarity, neighbors)
            fit_seconds = time.perf_counter() - fit_start + similarity_seconds
            eval_start = time.perf_counter()
            summary, detail = evaluate(matrix, validation, user_index, item_index, popularity, scores)
            summary.update({"model": name, "neighbors": neighbors, "shrinkage": shrinkage,
                            "fit_seconds": fit_seconds, "evaluation_seconds": time.perf_counter() - eval_start})
            experiments.append(summary); details[name] = detail
            detail.to_csv(output / f"validation_users_{name}.csv", index=False)
            if verbose:
                print(f"{name}: validation NDCG@10={summary['ndcg@10']:.6f}", flush=True)
    results = pd.DataFrame(experiments)
    results.to_csv(output / "validation_results.csv", index=False)
    # Stable selection: primary metric, then lower neighbourhood count/shrinkage.
    selected = results[results.model != "popularity"].sort_values(
        ["ndcg@10", "neighbors", "shrinkage"], ascending=[False, True, True]).iloc[0]
    chosen = str(selected.model)
    aligned = details[chosen].merge(details["popularity"], on="userID", suffixes=("_cf", "_pop"), validate="one_to_one")
    uncertainty = {
        "popularity_ndcg10": bootstrap_mean_ci(details["popularity"]["ndcg@10"]),
        "selected_itemcf_ndcg10": bootstrap_mean_ci(details[chosen]["ndcg@10"]),
        "paired_itemcf_minus_popularity_ndcg10": bootstrap_mean_ci(aligned["ndcg@10_cf"] - aligned["ndcg@10_pop"]),
        "caution": "Validation split only; tuned-winner selection and random-split variability are not covered. No final superiority claim.",
    }
    save_json(output / "validation_uncertainty.json", uncertainty)
    groups = []
    for name in ("popularity", chosen):
        detail = details[name]
        for label, mask in (("all", np.ones(len(detail), dtype=bool)),
                            ("has_cold_target", detail.n_cold_targets > 0),
                            ("all_targets_warm", detail.n_cold_targets == 0),
                            ("train_degree_below_30", detail.train_degree < 30)):
            group = detail.loc[mask]
            groups.append({"model": name, "group": label, "users": len(group),
                           "ndcg@10": float(group["ndcg@10"].mean()) if len(group) else None,
                           "recall@10": float(group["recall@10"].mean()) if len(group) else None})
    pd.DataFrame(groups).to_csv(output / "validation_user_groups.csv", index=False)
    save_json(output / "selection.json", {"phase": 1, "selected_itemcf_baseline": chosen,
        "selection_metric": "validation macro NDCG@10", "final_model_selected": False,
        "remaining_models": ["BPR-MF", "NeuMF", "content two-tower (leakage-safe feature gate required)"],
        "test_metrics_evaluated": False})
    environment = {"python": sys.version, "platform": platform.platform(),
        "dependencies": {name: version(name) for name in ("numpy", "scipy", "pandas", "matplotlib")},
        "windows_execution_verified": platform.system() == "Windows", "elapsed_seconds": time.perf_counter() - start}
    save_json(output / "runtime.json", environment)
    if verbose:
        print(results[["model", "ndcg@10", "recall@10", "precision@10", "catalogue_coverage@10"]].to_string(index=False))
        print("Test metrics: NOT EVALUATED. Selected ItemCF baseline:", chosen)
    return {"audit": audit, "manifest": manifest, "results": results,
            "selection": chosen, "uncertainty": uncertainty, "runtime": environment}
