"""Synthetic e-commerce data generator (documented, reproducible via config seed).

Ground truth: users and items have latent vectors tied to a product category, so
interactions are personalised (users prefer some categories) but still popularity
skewed (Zipf-like). A small share of records is deliberately corrupted so the
validation stage has real problems to catch.
"""
import numpy as np
import pandas as pd

from src.common import get_logger

log = get_logger("ingestion.generate")

CATEGORIES = ["Electronics", "Books", "Fashion", "Home", "Sports",
              "Beauty", "Toys", "Grocery", "Garden", "Music"]
COUNTRIES = ["India", "USA", "UK", "Germany", "Canada", "Australia"]


def generate(cfg: dict) -> dict[str, pd.DataFrame]:
    s = cfg["synthetic_data"]
    rng = np.random.default_rng(cfg["seed"])
    U, I, C, k = s["n_users"], s["n_items"], s["n_categories"], s["latent_dim"]
    start, end = pd.Timestamp(s["start_date"]), pd.Timestamp(s["end_date"])

    # ---- items -------------------------------------------------------------
    cat_vec = rng.normal(size=(C, k))
    item_cat = rng.integers(0, C, I)
    item_f = 0.5 * rng.normal(size=(I, k)) + 1.0 * cat_vec[item_cat]
    items = pd.DataFrame({
        "item_id": [f"I{i:04d}" for i in range(1, I + 1)],
        "title": [f"Product {i}" for i in range(1, I + 1)],
        "category": [CATEGORIES[c % len(CATEGORIES)] for c in item_cat],
        "price": np.round(rng.lognormal(3.5, 0.8, I), 2),
        "release_year": rng.integers(2015, 2024, I),
    })

    # ---- users -------------------------------------------------------------
    pref_cat = rng.integers(0, C, U)
    user_f = 0.5 * rng.normal(size=(U, k)) + 1.0 * cat_vec[pref_cat]
    signup_span = (start - pd.Timestamp("2022-01-01")).days
    signup = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, signup_span + 300, U), unit="D")
    users = pd.DataFrame({
        "user_id": [f"U{u:04d}" for u in range(1, U + 1)],
        "age": rng.integers(16, 70, U),
        "gender": rng.choice(["M", "F", "Other"], U, p=[0.48, 0.48, 0.04]),
        "country": rng.choice(COUNTRIES, U, p=[0.35, 0.25, 0.12, 0.1, 0.1, 0.08]),
        "signup_date": signup.strftime("%Y-%m-%d"),
    })

    # ---- interactions ------------------------------------------------------
    pop = (np.arange(1, I + 1) ** -0.7)
    rng.shuffle(pop)
    affinity = user_f @ item_f.T / np.sqrt(k)               # U x I
    item_bias = rng.normal(0, 0.3, I)
    n_per_user = np.clip(rng.lognormal(np.log(s["mean_interactions_per_user"]), 0.6, U).astype(int), 6, I // 2)
    rows = []
    for u in range(U):
        logits = np.log(pop) + 1.6 * affinity[u]
        p = np.exp(logits - logits.max()); p /= p.sum()
        chosen = rng.choice(I, size=n_per_user[u], replace=False, p=p)
        lo = max(start, signup[u])
        span = max((end - lo).days, 1)
        for j in chosen:
            r = 2.5 + 0.8 * affinity[u, j] + item_bias[j] + rng.normal(0, 0.6)
            ts = lo + pd.to_timedelta(int(rng.integers(0, span)), unit="D") + pd.to_timedelta(int(rng.integers(0, 86400)), unit="s")
            rows.append((users.user_id[u], items.item_id[j], float(np.clip(np.round(r), 1, 5)), ts))
    inter = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    inter = inter.sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)
    inter["timestamp"] = inter["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    log.info("Clean synthetic data: %d users, %d items, %d interactions", len(users), len(items), len(inter))

    return _inject_dirt(users, items, inter, rng, s["dirty_rate"])


def _inject_dirt(users, items, inter, rng, rate) -> dict[str, pd.DataFrame]:
    """Corrupt a small share of records (all as strings, like a raw CSV export)."""
    users, items, inter = users.astype(object).copy(), items.astype(object).copy(), inter.astype(object).copy()
    n = len(inter)

    def pick(frame, frac):
        return rng.choice(len(frame), size=max(1, int(len(frame) * frac)), replace=False)

    # interactions
    inter.loc[pick(inter, rate * 0.25), "rating"] = rng.choice(["0", "6", "-1", "11"], 1)[0]
    inter.loc[pick(inter, rate * 0.20), "rating"] = np.nan                      # missing rating
    inter.loc[pick(inter, rate * 0.10), "rating"] = "five"                      # wrong type
    inter.loc[pick(inter, rate * 0.15), "user_id"] = np.nan                     # missing key
    inter.loc[pick(inter, rate * 0.20), "timestamp"] = "2031-06-15 10:00:00"    # future
    inter.loc[pick(inter, rate * 0.10), "timestamp"] = "not_a_date"             # unparseable
    inter.loc[pick(inter, rate * 0.05), "timestamp"] = "1999-01-01 00:00:00"    # too old
    inter.loc[pick(inter, rate * 0.20), "user_id"] = "U9999"                    # orphan user
    inter.loc[pick(inter, rate * 0.20), "item_id"] = "I9999"                    # orphan item
    dups = inter.iloc[pick(inter, rate * 0.35)]
    inter = pd.concat([inter, dups], ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)

    # users
    users.loc[pick(users, 0.04), "age"] = np.nan                                # missing (imputable)
    users.loc[pick(users, 0.01), "age"] = "twenty"                              # wrong type
    users.loc[pick(users, 0.005), "age"] = "250"                                # out of range
    users.loc[pick(users, 0.03), "country"] = np.nan
    users.loc[pick(users, 0.2), "gender"] = users["gender"].map({"M": "male", "F": "Female"}).fillna("other")  # inconsistent format
    users = pd.concat([users, users.iloc[pick(users, 0.01)]], ignore_index=True)   # duplicate users

    # items
    items.loc[pick(items, 0.04), "category"] = np.nan
    items.loc[pick(items, 0.15), "category"] = items["category"].str.upper()        # inconsistent case
    items.loc[pick(items, 0.01), "price"] = "abc"
    items.loc[pick(items, 0.01), "price"] = "-5.0"
    items = pd.concat([items, items.iloc[pick(items, 0.01)]], ignore_index=True)

    log.info("Injected dirty records: interactions=%d (incl. duplicates), users=%d, items=%d",
             len(inter), len(users), len(items))
    return {"users": users, "items": items, "interactions": inter}
