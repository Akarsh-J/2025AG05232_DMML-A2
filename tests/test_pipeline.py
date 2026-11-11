import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import load_config  # noqa: E402
from src.evaluation import metrics as M  # noqa: E402
from src.preparation.prepare import time_split  # noqa: E402
from src.validation.validate import _apply, interaction_rules  # noqa: E402


def test_precision_recall_ndcg():
    recs, rel = ["a", "b", "c", "d", "e"], {"a", "c", "z"}
    assert M.precision_at_k(recs, rel, 5) == 2 / 5
    assert M.recall_at_k(recs, rel, 5) == 2 / 3
    assert 0 < M.ndcg_at_k(recs, rel, 5) < 1
    assert M.ndcg_at_k(["a", "c", "z"], rel, 3) == 1.0       # perfect ranking


def test_coverage_diversity():
    assert M.coverage([["a", "b"], ["b", "c"]], 6) == 0.5
    cat = {"a": "x", "b": "x", "c": "y"}
    assert M.diversity([["a", "b"]], cat) == 0.0
    assert M.diversity([["a", "c"]], cat) == 1.0


def test_time_split_has_no_leakage():
    ts = pd.date_range("2024-01-01", periods=100)
    df = pd.DataFrame({"user_id": ["u"] * 100, "item_id": ["i"] * 100, "rating": 4.0, "timestamp": ts})
    train, test, *_ = time_split(df, 0.2)
    assert train.timestamp.max() < test.timestamp.min()


def test_validation_catches_bad_rows():
    cfg = load_config()
    df = pd.DataFrame({
        "user_id": ["u1", "u1", None, "u9", "u1"], "item_id": ["i1", "i1", "i1", "i1", "i1"],
        "rating": ["5", "5", "4", "3", "9"],
        "timestamp": ["2024-01-01"] * 4 + ["2040-01-01"]})
    df.loc[1, "timestamp"] = "2024-01-01"  # exact duplicate of row 0
    clean, bad, _ = _apply("interactions", df, interaction_rules(df, cfg, {"u1"}, {"i1"}))
    assert len(clean) == 1 and len(bad) == 4
