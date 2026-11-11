"""Stage 5 - Feature management (lightweight custom feature layer).

* FEATURE_REGISTRY is the single source of truth: it documents every feature and is used
  to generate the feature catalog (reports/feature_catalog.md).
* The SAME functions (build_user_features / build_item_features / build_pair_features) are
  used for training and for prediction -> no training/serving skew.
* Features are computed "as of" a timestamp (end of train data) from train interactions only.
* FeatureStore materialises features to parquet (offline store) and serves lookups by id.
"""
import numpy as np
import pandas as pd

from src.common import get_logger, load_config, path

log = get_logger("features")

FEATURE_REGISTRY = [
    # entity, name, dtype, description, logic
    ("user", "user_avg_rating", "float", "Average rating the user has given", "mean(rating) over user's train interactions"),
    ("user", "user_rating_std", "float", "Rating variability (0 if single rating)", "std(rating)"),
    ("user", "user_activity_count", "int", "Number of interactions by the user", "count(interactions)"),
    ("user", "user_positive_rate", "float", "Share of ratings considered positive", "mean(rating >= positive_threshold)"),
    ("user", "user_preferred_category", "string", "Category the user interacted with most (tie -> highest avg rating)", "mode(item.category)"),
    ("user", "user_n_categories", "int", "Number of distinct categories explored", "nunique(item.category)"),
    ("user", "user_days_since_last_interaction", "float", "Days between user's last interaction and as_of date", "as_of - max(timestamp)"),
    ("user", "user_tenure_days", "int", "Days since signup at as_of date", "as_of - signup_date"),
    ("item", "item_popularity", "int", "Number of interactions the item received", "count(interactions)"),
    ("item", "item_avg_rating", "float", "Average rating received (raw)", "mean(rating)"),
    ("item", "item_weighted_rating", "float", "Bayesian-smoothed average rating (shrunk toward global mean)", "(n*avg + m*global_avg)/(n+m)"),
    ("item", "item_positive_rate", "float", "Share of positive ratings", "mean(rating >= positive_threshold)"),
    ("item", "item_days_since_last_interaction", "float", "Days since the item was last interacted with", "as_of - max(timestamp)"),
    ("item", "item_category", "string", "Item category (standardised)", "items.category"),
    ("item", "item_price_bucket", "string", "Price quartile bucket (low/mid/high/premium)", "qcut(price, 4)"),
    ("pair", "category_match", "int", "1 if item category == user's preferred category", "item_category == user_preferred_category"),
    ("pair", "rating_gap_vs_user_avg", "float", "Item avg rating minus user avg rating", "item_avg_rating - user_avg_rating"),
    ("pair", "days_since_interaction", "float", "Days between this interaction and as_of (training pairs)", "as_of - timestamp"),
    ("pair", "recency_weight", "float", "Exponential time-decay weight of an interaction", "0.5 ** (days / half_life)"),
]


def build_user_features(train, users, items, as_of, cfg) -> pd.DataFrame:
    thr = cfg["preparation"]["positive_rating_threshold"]
    df = train.merge(items[["item_id", "category"]], on="item_id")
    g = df.groupby("user_id")
    f = g.agg(user_avg_rating=("rating", "mean"), user_rating_std=("rating", "std"),
              user_activity_count=("item_id", "count"), user_n_categories=("category", "nunique"),
              last=("timestamp", "max")).reset_index()
    f["user_rating_std"] = f["user_rating_std"].fillna(0.0)
    f["user_positive_rate"] = g["rating"].apply(lambda s: (s >= thr).mean()).values
    f["user_days_since_last_interaction"] = (as_of - f.pop("last")).dt.total_seconds() / 86400
    # preferred category: most interactions, tie-break by higher average rating
    cat = df.groupby(["user_id", "category"]).agg(n=("rating", "size"), r=("rating", "mean")).reset_index()
    cat = cat.sort_values(["user_id", "n", "r"], ascending=[True, False, False]).drop_duplicates("user_id")
    f = f.merge(cat[["user_id", "category"]].rename(columns={"category": "user_preferred_category"}), on="user_id")
    f = f.merge(users[["user_id", "signup_date"]], on="user_id", how="left")
    f["user_tenure_days"] = (as_of - f.pop("signup_date")).dt.days
    return f


def build_item_features(train, items, as_of, cfg) -> pd.DataFrame:
    thr, m = cfg["preparation"]["positive_rating_threshold"], cfg["features"]["popularity_prior_strength"]
    gavg = train["rating"].mean()
    g = train.groupby("item_id")
    f = g.agg(item_popularity=("user_id", "count"), item_avg_rating=("rating", "mean"),
              last=("timestamp", "max")).reset_index()
    f["item_positive_rate"] = g["rating"].apply(lambda s: (s >= thr).mean()).values
    f["item_weighted_rating"] = (f.item_popularity * f.item_avg_rating + m * gavg) / (f.item_popularity + m)
    f["item_days_since_last_interaction"] = (as_of - f.pop("last")).dt.total_seconds() / 86400
    f = items[["item_id", "category", "price"]].merge(f, on="item_id", how="left")
    f["item_popularity"] = f["item_popularity"].fillna(0).astype(int)
    f["item_avg_rating"] = f["item_avg_rating"].fillna(gavg)
    f["item_weighted_rating"] = f["item_weighted_rating"].fillna(gavg)
    f["item_positive_rate"] = f["item_positive_rate"].fillna(0.0)
    f["item_days_since_last_interaction"] = f["item_days_since_last_interaction"].fillna(9999.0)
    f["item_price_bucket"] = pd.qcut(f.pop("price"), 4, labels=["low", "mid", "high", "premium"]).astype(str)
    return f.rename(columns={"category": "item_category"})


def build_pair_features(pairs: pd.DataFrame, user_f: pd.DataFrame, item_f: pd.DataFrame,
                        as_of, cfg) -> pd.DataFrame:
    """Interaction-level features; works for training pairs (with timestamp) and for
    candidate (user,item) pairs at prediction time (timestamp absent)."""
    out = (pairs.merge(user_f[["user_id", "user_avg_rating", "user_preferred_category"]], on="user_id", how="left")
                .merge(item_f[["item_id", "item_avg_rating", "item_category"]], on="item_id", how="left"))
    out["category_match"] = (out.item_category == out.user_preferred_category).astype(int)
    out["rating_gap_vs_user_avg"] = out.item_avg_rating - out.user_avg_rating
    if "timestamp" in out:
        out["days_since_interaction"] = (as_of - out["timestamp"]).dt.total_seconds() / 86400
        out["recency_weight"] = 0.5 ** (out["days_since_interaction"] / cfg["features"]["recency_half_life_days"])
    return out.drop(columns=["user_avg_rating", "user_preferred_category", "item_avg_rating", "item_category"])


class FeatureStore:
    """Offline store (parquet) with simple online-style lookups."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.dir = path(self.cfg, "features")
        self._user = self._item = None

    @property
    def user_features(self) -> pd.DataFrame:
        if self._user is None:
            self._user = pd.read_parquet(self.dir / "user_features.parquet")
        return self._user

    @property
    def item_features(self) -> pd.DataFrame:
        if self._item is None:
            self._item = pd.read_parquet(self.dir / "item_features.parquet")
        return self._item

    def get_user_features(self, user_ids) -> pd.DataFrame:
        return self.user_features[self.user_features.user_id.isin(user_ids)]

    def get_item_features(self, item_ids) -> pd.DataFrame:
        return self.item_features[self.item_features.item_id.isin(item_ids)]


def write_catalog(cfg: dict) -> None:
    rep = path(cfg, "reports")
    lines = ["# Feature Catalog", "",
             "Generated from `FEATURE_REGISTRY` in `src/features/feature_store.py`. Features are computed "
             "from the **train split only**, \"as of\" the last train timestamp, with the same code path used "
             "for training and prediction.", "",
             "| Entity | Feature | Type | Description | Logic |", "|---|---|---|---|---|"]
    lines += [f"| {e} | `{n}` | {t} | {d} | `{l}` |" for e, n, t, d, l in FEATURE_REGISTRY]
    (rep / "feature_catalog.md").write_text("\n".join(lines) + "\n")
    pd.DataFrame(FEATURE_REGISTRY, columns=["entity", "feature", "type", "description", "logic"]) \
        .to_csv(rep / "feature_catalog.csv", index=False)


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    proc, feat = path(cfg, "processed"), path(cfg, "features")
    users, items = pd.read_parquet(proc / "users.parquet"), pd.read_parquet(proc / "items.parquet")
    train = pd.read_parquet(proc / "train.parquet")
    as_of = train["timestamp"].max()

    uf = build_user_features(train, users, items, as_of, cfg)
    itf = build_item_features(train, items, as_of, cfg)
    pf = build_pair_features(train[["user_id", "item_id", "timestamp"]], uf, itf, as_of, cfg)
    uf.to_parquet(feat / "user_features.parquet", index=False)
    itf.to_parquet(feat / "item_features.parquet", index=False)
    pf.to_parquet(feat / "interaction_features.parquet", index=False)
    write_catalog(cfg)
    log.info("Features written: users=%d (%d cols) items=%d (%d cols) pairs=%d",
             len(uf), uf.shape[1] - 1, len(itf), itf.shape[1] - 1, len(pf))
    return {"features_dir": str(feat), "as_of": str(as_of)}


if __name__ == "__main__":
    run()
