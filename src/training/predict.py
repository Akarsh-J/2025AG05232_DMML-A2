"""Prediction/serving demo: top-K for a user using the stored model + the SAME feature store.

* Known user  -> personalised model (item-kNN CF by default), already-seen items removed.
* Cold-start user (not in train) -> popularity fallback (optionally restricted to a preferred category).

Usage:  python -m src.training.predict --user U0001 --k 10
"""
import argparse
import pickle

import pandas as pd
import scipy.sparse as sp

from src.common import load_config, load_json, path
from src.features.feature_store import FeatureStore


def recommend(user_id: str, k: int = 10, model_name: str = "item_knn_cf", category: str | None = None, cfg=None):
    cfg = cfg or load_config()
    proc, mdir = path(cfg, "processed"), path(cfg, "models")
    maps = load_json(proc / "id_mappings.json")
    idx2item = {v: k_ for k_, v in maps["item_to_idx"].items()}
    train_mat = sp.load_npz(proc / "user_item_matrix.npz").tocsr()
    fs = FeatureStore(cfg)
    known = user_id in maps["user_to_idx"] and len(fs.get_user_features([user_id])) > 0
    name = model_name if known else "popularity_baseline"
    with open(mdir / f"{name}.pkl", "rb") as f:
        model = pickle.load(f)
    u = maps["user_to_idx"].get(user_id, 0)
    recs = [idx2item[i] for i in model.recommend([u], 50 if category else k, train_mat if known else sp.csr_matrix(train_mat.shape))[0]]
    items = fs.get_item_features(recs).set_index("item_id").loc[recs].reset_index()
    if category is None and known:
        category = None
    if category:
        items = items[items.item_category == category]
    return name, items.head(k)[["item_id", "item_category", "item_popularity", "item_weighted_rating"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--model", default="item_knn_cf")
    a = ap.parse_args()
    m, df = recommend(a.user, a.k, a.model)
    print(f"Model used: {m}")
    print(df.to_string(index=False))
