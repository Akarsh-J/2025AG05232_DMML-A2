"""Stage 3 - Data preparation.

Cleans validated data: casts types, standardises formats, imputes warn-level missing
values, removes duplicate user-item pairs (keeps the latest), and creates a
TIME-BASED train/test split (latest X% of interactions -> test).
Cold-start users/items in test (unseen in train) are dropped from the test set for
offline evaluation (documented limitation; cold-start is handled by a fallback at serving time).
"""
import pandas as pd

from src.common import get_logger, load_config, path, save_json

log = get_logger("preparation")

GENDER_MAP = {"m": "Male", "male": "Male", "f": "Female", "female": "Female", "other": "Other"}


def clean_users(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age"] = df["age"].fillna(df["age"].median()).round().astype(int)           # impute median
    df["gender"] = df["gender"].str.strip().str.lower().map(GENDER_MAP).fillna("Other")
    df["country"] = df["country"].fillna("Unknown").str.strip().str.title()
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    return df.reset_index(drop=True)


def clean_items(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["category"] = df["category"].fillna("Unknown").str.strip().str.title()      # 'BOOKS' -> 'Books'
    df["price"] = pd.to_numeric(df["price"])
    df["release_year"] = pd.to_numeric(df["release_year"]).astype(int)
    return df.reset_index(drop=True)


def clean_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rating"] = pd.to_numeric(df["rating"]).astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    before = len(df)
    # one row per user-item pair: keep the most recent interaction
    df = df.sort_values("timestamp").drop_duplicates(["user_id", "item_id"], keep="last")
    log.info("Removed %d repeated user-item pairs (kept latest)", before - len(df))
    return df.sort_values("timestamp").reset_index(drop=True)


def time_split(df: pd.DataFrame, test_fraction: float):
    cutoff = df["timestamp"].quantile(1 - test_fraction)
    train, test = df[df["timestamp"] <= cutoff], df[df["timestamp"] > cutoff]
    n_test = len(test)
    test = test[test.user_id.isin(train.user_id) & test.item_id.isin(train.item_id)]
    log.info("Time split at %s -> train=%d test=%d (dropped %d cold-start test rows)",
             cutoff.date(), len(train), len(test), n_test - len(test))
    return train, test, cutoff, n_test - len(test)


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    proc = path(cfg, "processed")
    v = proc / "validated"
    users = clean_users(pd.read_parquet(v / "users.parquet"))
    items = clean_items(pd.read_parquet(v / "items.parquet"))
    inter = clean_interactions(pd.read_parquet(v / "interactions.parquet"))

    train, test, cutoff, dropped = time_split(inter, cfg["preparation"]["test_fraction"])
    assert train["timestamp"].max() < test["timestamp"].min(), "time-split leakage!"

    users.to_parquet(proc / "users.parquet", index=False)
    items.to_parquet(proc / "items.parquet", index=False)
    inter.to_parquet(proc / "interactions.parquet", index=False)
    train.to_parquet(proc / "train.parquet", index=False)
    test.to_parquet(proc / "test.parquet", index=False)
    save_json({"split_cutoff": cutoff, "train_rows": len(train), "test_rows": len(test),
               "cold_start_test_rows_dropped": dropped, "n_users": len(users), "n_items": len(items)},
              proc / "split_summary.json")
    return {"processed_dir": str(proc)}


if __name__ == "__main__":
    run()
