"""Dataset checks and a deterministic per-user random holdout (not temporal)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DATA_URL = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
ARCHIVE_SHA256 = "6738f48195667ff03caaab4d32ca9a3133d8cc026b7c3cdaf6ce1010e913c59c"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")


def load_data(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    interactions = pd.read_csv(raw / "user_artists.dat", sep="\t")
    artists = pd.read_csv(raw / "artists.dat", sep="\t")
    tags = pd.read_csv(raw / "tags.dat", sep="\t", encoding="latin-1")
    events = pd.read_csv(raw / "user_taggedartists-timestamps.dat", sep="\t")
    required = ["userID", "artistID", "weight"]
    if list(interactions.columns) != required:
        raise ValueError("Unexpected listening schema")
    if interactions[required].isna().any().any():
        raise ValueError("Missing listening values: stop rather than silently drop")
    if interactions.duplicated(["userID", "artistID"]).any():
        raise ValueError("Duplicate user-artist pairs: inspect instead of silently sum")
    if (interactions.weight <= 0).any():
        raise ValueError("Nonpositive listening counts")
    if artists.id.duplicated().any() or not set(interactions.artistID).issubset(artists.id):
        raise ValueError("Invalid artist catalogue")
    n_users = interactions.userID.nunique()
    n_items = len(artists)
    counts = interactions.groupby("userID").size()
    popularity = interactions.groupby("artistID").size()
    audit = {
        "source": DATA_URL,
        "license": "Dataset README: non-commercial use; cite Last.fm and HetRec 2011",
        "users": int(n_users), "catalogue_artists": n_items,
        "listened_artists": int(interactions.artistID.nunique()),
        "interactions": len(interactions),
        "density": len(interactions) / (n_users * n_items),
        "sparsity": 1 - len(interactions) / (n_users * n_items),
        "duplicate_pairs": int(interactions.duplicated(["userID", "artistID"]).sum()),
        "missing_interaction_values": interactions.isna().sum().astype(int).to_dict(),
        "missing_artist_values": artists.isna().sum().astype(int).to_dict(),
        "nonpositive_weights": int((interactions.weight <= 0).sum()),
        "weight_quantiles": {str(k): float(v) for k, v in interactions.weight.quantile([0,.5,.9,.99,1]).items()},
        "user_degree_quantiles": {str(k): float(v) for k, v in counts.quantile([0,.5,.9,1]).items()},
        "users_with_50_artists": int((counts == 50).sum()),
        "artists_with_one_listener": int((popularity == 1).sum()),
        "tag_vocabulary": len(tags), "tag_events": len(events),
        "tag_event_unknown_artists": int((~events.artistID.isin(artists.id)).sum()),
        "tag_event_unknown_tags": int((~events.tagID.isin(tags.tagID)).sum()),
        "tag_event_duplicate_rows": int(events.duplicated().sum()),
        "listening_timestamp_available": False,
        "tag_timestamp_is_listening_timestamp": False,
        "raw_files_sha256": {p.name: sha256(p) for p in sorted(raw.iterdir()) if p.is_file()},
    }
    return interactions, artists, audit


def split_interactions(frame: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """80/10/10 approximate; floor holdouts with >=1 each for degree >=5.

    Each user's RNG is seeded independently, so input row order has no effect.
    Users with fewer than five interactions are retained in train only.
    No item coverage repair, k-core filtering, or held-out-informed resampling.
    """
    parts: dict[str, list[pd.DataFrame]] = {s: [] for s in ("train", "validation", "test")}
    for uid, group in frame.groupby("userID", sort=True):
        group = group.sort_values("artistID").reset_index(drop=True)
        if len(group) < 5:
            parts["train"].append(group)
            continue
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(uid)]))
        order = rng.permutation(len(group))
        holdout = max(1, len(group) // 10)
        parts["validation"].append(group.iloc[order[:holdout]])
        parts["test"].append(group.iloc[order[holdout:2 * holdout]])
        parts["train"].append(group.iloc[order[2 * holdout:]])
    result = {s: pd.concat(v, ignore_index=True).sort_values(["userID", "artistID"]).reset_index(drop=True)
              if v else frame.iloc[:0].copy() for s, v in parts.items()}
    pair_sets = {s: set(zip(f.userID, f.artistID)) for s, f in result.items()}
    assert not pair_sets["train"] & pair_sets["validation"]
    assert not pair_sets["train"] & pair_sets["test"]
    assert not pair_sets["validation"] & pair_sets["test"]
    assert sum(map(len, result.values())) == len(frame)
    assert set(result["validation"].userID).issubset(result["train"].userID)
    return result


def indexed_matrix(train: pd.DataFrame, artists: pd.DataFrame):
    user_ids = np.sort(train.userID.unique())
    artist_ids = np.sort(artists.id.unique())
    user_index = {int(v): i for i, v in enumerate(user_ids)}
    item_index = {int(v): i for i, v in enumerate(artist_ids)}
    rows = train.userID.map(user_index).to_numpy()
    cols = train.artistID.map(item_index).to_numpy()
    matrix = csr_matrix((np.ones(len(train), dtype=np.float64), (rows, cols)),
                        shape=(len(user_ids), len(artist_ids)))
    assert matrix.nnz == len(train)
    return matrix, user_ids, artist_ids, user_index, item_index
