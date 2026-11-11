"""Stage 4 - Data transformation.

Builds ID <-> index mappings and derived datasets from the TRAIN split only
(no test leakage):
  * user_activity_summary.parquet  - per-user activity statistics
  * item_popularity.parquet        - per-item interaction counts / ratings
  * user_item_matrix.npz           - sparse explicit-rating matrix (users x items)
  * user_item_binary.npz           - sparse binary matrix of positive interactions (rating >= threshold)
"""
import pandas as pd
import scipy.sparse as sp

from src.common import get_logger, load_config, path, save_json

log = get_logger("transformation")


def build_mappings(users: pd.DataFrame, items: pd.DataFrame) -> dict:
    return {"user_to_idx": {u: i for i, u in enumerate(sorted(users.user_id))},
            "item_to_idx": {it: i for i, it in enumerate(sorted(items.item_id))}}


def to_matrix(df: pd.DataFrame, maps: dict, value_col: str | None = "rating") -> sp.csr_matrix:
    r = df.user_id.map(maps["user_to_idx"]).to_numpy()
    c = df.item_id.map(maps["item_to_idx"]).to_numpy()
    vals = df[value_col].to_numpy(dtype=float) if value_col else [1.0] * len(df)
    return sp.csr_matrix((vals, (r, c)), shape=(len(maps["user_to_idx"]), len(maps["item_to_idx"])))


def user_activity_summary(train: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("user_id")
    return g.agg(n_interactions=("item_id", "count"), avg_rating=("rating", "mean"),
                 first_interaction=("timestamp", "min"), last_interaction=("timestamp", "max")).reset_index()


def item_popularity(train: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("item_id")
    out = g.agg(n_interactions=("user_id", "count"), n_users=("user_id", "nunique"),
                avg_rating=("rating", "mean")).reset_index()
    return out.sort_values("n_interactions", ascending=False).reset_index(drop=True)


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    proc = path(cfg, "processed")
    thr = cfg["preparation"]["positive_rating_threshold"]
    users, items = pd.read_parquet(proc / "users.parquet"), pd.read_parquet(proc / "items.parquet")
    train = pd.read_parquet(proc / "train.parquet")

    maps = build_mappings(users, items)
    save_json(maps, proc / "id_mappings.json")
    ratings = to_matrix(train, maps, "rating")
    binary = to_matrix(train[train.rating >= thr], maps, None)
    sp.save_npz(proc / "user_item_matrix.npz", ratings)
    sp.save_npz(proc / "user_item_binary.npz", binary)
    user_activity_summary(train).to_parquet(proc / "user_activity_summary.parquet", index=False)
    item_popularity(train).to_parquet(proc / "item_popularity.parquet", index=False)
    density = ratings.nnz / (ratings.shape[0] * ratings.shape[1])
    log.info("User-item matrix %s, density=%.4f, positives=%d", ratings.shape, density, binary.nnz)
    return {"matrix_shape": list(ratings.shape), "density": density}


if __name__ == "__main__":
    run()
