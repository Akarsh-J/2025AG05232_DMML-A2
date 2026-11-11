"""Stage 2 - Automated data validation.

Each rule is a function returning a boolean mask of *violating* rows. Rules have a
severity:
  reject -> row is moved to the error dataset (data/errors/<table>_errors.csv)
  warn   -> row is kept and logged; it is repaired later in preparation (e.g. imputation)
Outputs: clean validated parquet files + error datasets + validation report (md + json).
"""
import pandas as pd

from src.common import get_logger, load_config, path, save_json

log = get_logger("validation")


# --------------------------------------------------------------------------- helpers
def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _missing(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype("string").str.strip() == "")


def _dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", format="mixed")


# --------------------------------------------------------------------------- rules
def user_rules(df, cfg):
    v = cfg["validation"]
    age = _num(df["age"])
    return [
        ("missing_user_id", "reject", _missing(df["user_id"])),
        ("duplicate_user_id", "reject", df.duplicated("user_id", keep="first") & ~_missing(df["user_id"])),
        ("age_wrong_type", "reject", ~_missing(df["age"]) & age.isna()),
        ("age_out_of_range", "reject", age.notna() & ((age < v["age_min"]) | (age > v["age_max"]))),
        ("invalid_signup_date", "reject", _dt(df["signup_date"]).isna()),
        ("missing_age", "warn", _missing(df["age"])),
        ("missing_country", "warn", _missing(df["country"])),
    ]


def item_rules(df, cfg):
    price = _num(df["price"])
    return [
        ("missing_item_id", "reject", _missing(df["item_id"])),
        ("duplicate_item_id", "reject", df.duplicated("item_id", keep="first") & ~_missing(df["item_id"])),
        ("price_wrong_type", "reject", ~_missing(df["price"]) & price.isna()),
        ("negative_price", "reject", price < 0),
        ("missing_category", "warn", _missing(df["category"])),
    ]


def interaction_rules(df, cfg, valid_users: set, valid_items: set):
    v = cfg["validation"]
    rating = _num(df["rating"])
    ts = _dt(df["timestamp"])
    ref = pd.Timestamp(cfg["reference_date"])
    exact_dup = df.duplicated(["user_id", "item_id", "rating", "timestamp"], keep="first")
    return [
        ("missing_user_id", "reject", _missing(df["user_id"])),
        ("missing_item_id", "reject", _missing(df["item_id"])),
        ("missing_rating", "reject", _missing(df["rating"])),
        ("rating_wrong_type", "reject", ~_missing(df["rating"]) & rating.isna()),
        ("rating_out_of_range", "reject", rating.notna() & ((rating < v["rating_min"]) | (rating > v["rating_max"]))),
        ("invalid_timestamp_format", "reject", ts.isna()),
        ("timestamp_in_future", "reject", ts.notna() & (ts > ref)),
        ("timestamp_too_old", "reject", ts.notna() & (ts < pd.Timestamp(v["min_valid_timestamp"]))),
        ("duplicate_interaction", "reject", exact_dup),
        ("orphan_user_id (referential integrity)", "reject",
         ~_missing(df["user_id"]) & ~df["user_id"].isin(valid_users)),
        ("orphan_item_id (referential integrity)", "reject",
         ~_missing(df["item_id"]) & ~df["item_id"].isin(valid_items)),
    ]


# --------------------------------------------------------------------------- engine
def _apply(table: str, df: pd.DataFrame, rules: list) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    reject_mask = pd.Series(False, index=df.index)
    reasons = pd.Series("", index=df.index)
    summary, errors = [], []
    for name, severity, mask in rules:
        mask = mask.fillna(False)
        summary.append({"table": table, "check": name, "severity": severity,
                        "violations": int(mask.sum()), "pct": round(100 * mask.mean(), 3)})
        if severity == "reject":
            reasons[mask] = reasons[mask] + (name + "; ")
            reject_mask |= mask
    bad = df[reject_mask].copy()
    bad.insert(0, "error_reasons", reasons[reject_mask].str.rstrip("; "))
    bad.insert(0, "source_row", bad.index + 2)      # +2 = line number in the CSV (header = line 1)
    return df[~reject_mask].copy(), bad, summary


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    raw, proc, err, rep = path(cfg, "raw"), path(cfg, "processed"), path(cfg, "errors"), path(cfg, "reports")
    users = pd.read_csv(raw / "users.csv", dtype=str)
    items = pd.read_csv(raw / "items.csv", dtype=str)
    inter = pd.read_csv(raw / "interactions.csv", dtype=str)

    clean_u, bad_u, s_u = _apply("users", users, user_rules(users, cfg))
    clean_i, bad_i, s_i = _apply("items", items, item_rules(items, cfg))
    clean_x, bad_x, s_x = _apply("interactions", inter,
                                 interaction_rules(inter, cfg, set(clean_u.user_id), set(clean_i.item_id)))

    for name, bad in [("users", bad_u), ("items", bad_i), ("interactions", bad_x)]:
        bad.to_csv(err / f"{name}_errors.csv", index=False)
    vdir = proc / "validated"
    vdir.mkdir(exist_ok=True)
    clean_u.to_parquet(vdir / "users.parquet", index=False)
    clean_i.to_parquet(vdir / "items.parquet", index=False)
    clean_x.to_parquet(vdir / "interactions.parquet", index=False)

    summary = pd.DataFrame(s_u + s_i + s_x)
    totals = {"users": (len(users), len(clean_u)), "items": (len(items), len(clean_i)),
              "interactions": (len(inter), len(clean_x))}
    _write_report(summary, totals, rep)
    save_json({"totals": {k: {"input": a, "valid": b, "rejected": a - b} for k, (a, b) in totals.items()},
               "checks": summary.to_dict("records")}, rep / "data_validation_report.json")
    for t, (a, b) in totals.items():
        log.info("Validated %-12s input=%d valid=%d rejected=%d", t, a, b, a - b)
    return {"validated_dir": str(vdir), "errors_dir": str(err)}


def _write_report(summary: pd.DataFrame, totals: dict, rep) -> None:
    lines = ["# Data Validation Report", "",
             "Generated automatically by `src/validation/validate.py`. "
             "`reject` = row moved to `data/errors/`; `warn` = row kept, repaired in preparation.", "",
             "## Row counts", "", "| Table | Input rows | Valid rows | Rejected rows | Rejected % |", "|---|---:|---:|---:|---:|"]
    for t, (a, b) in totals.items():
        lines.append(f"| {t} | {a} | {b} | {a - b} | {100 * (a - b) / a:.2f} |")
    lines += ["", "## Check results", "", "| Table | Check | Severity | Violations | % of rows |", "|---|---|---|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.table} | {r.check} | {r.severity} | {r.violations} | {r.pct} |")
    lines += ["", "Note: one row can violate several checks, so violations can exceed rejected rows.", ""]
    (rep / "data_validation_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    run()
