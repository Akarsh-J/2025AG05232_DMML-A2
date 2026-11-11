"""Stage 1 - Data ingestion.

Reads raw CSVs (user, item, interaction). If they do not exist, a documented synthetic
dataset is generated. Every file is read as *strings* so type problems are caught by
the validation stage instead of being silently coerced here. A manifest with row counts
and checksums is written for reproducibility/lineage.
"""
import hashlib
from datetime import datetime, timezone

import pandas as pd

from src.common import get_logger, load_config, path, save_json
from src.ingestion.generate_data import generate

log = get_logger("ingestion")
TABLES = ["users", "items", "interactions"]


def _md5(p) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def run(cfg: dict | None = None, force_regenerate: bool = False) -> dict:
    cfg = cfg or load_config()
    raw = path(cfg, "raw")
    files = {t: raw / f"{t}.csv" for t in TABLES}

    if force_regenerate or not all(f.exists() for f in files.values()):
        log.info("Raw CSVs not found (or regeneration requested) -> generating synthetic data")
        for t, df in generate(cfg).items():
            df.to_csv(files[t], index=False)
    else:
        log.info("Using existing raw CSV files in %s", raw)

    manifest = {"ingested_at": datetime.now(timezone.utc).isoformat(), "source": "synthetic/CSV", "tables": {}}
    for t, f in files.items():
        df = pd.read_csv(f, dtype=str, keep_default_na=True)
        manifest["tables"][t] = {"file": str(f.relative_to(raw.parents[1])), "rows": len(df),
                                  "columns": list(df.columns), "md5": _md5(f)}
        log.info("Ingested %-12s rows=%d cols=%s", t, len(df), list(df.columns))
    save_json(manifest, raw / "ingestion_manifest.json")
    return {t: str(f) for t, f in files.items()}


if __name__ == "__main__":
    run()
