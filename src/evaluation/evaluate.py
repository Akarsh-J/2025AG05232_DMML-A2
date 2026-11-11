"""Stage 7 - Model evaluation on the time-based hold-out test set.

Relevant items for a user = test interactions with rating >= threshold.
Already-seen (train) items are excluded from recommendations.
Metrics: Precision@K, Recall@K, NDCG@K, Coverage, Diversity (+ RMSE/MAE for rating predictors).
Results are logged to the MLflow runs created by the training stage and written to
reports/model_results.{csv,md}.
"""
import pickle

import mlflow
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common import get_logger, load_config, load_json, path
from src.evaluation import metrics as M

log = get_logger("evaluation")


def evaluate_model(model, ks, test, maps, train_mat, item_cat, thr) -> dict:
    u2i, i2i = maps["user_to_idx"], maps["item_to_idx"]
    idx2item = {v: k for k, v in i2i.items()}
    rel = test[test.rating >= thr].groupby("user_id")["item_id"].apply(set)
    users = [u for u in rel.index if u in u2i]
    uidx = np.array([u2i[u] for u in users])
    out = {}
    for k in ks:
        recs_idx = model.recommend(uidx, k, train_mat)
        recs = [[idx2item[j] for j in row] for row in recs_idx]
        out[f"precision@{k}"] = float(np.mean([M.precision_at_k(r, rel[u], k) for r, u in zip(recs, users)]))
        out[f"recall@{k}"] = float(np.mean([M.recall_at_k(r, rel[u], k) for r, u in zip(recs, users)]))
        out[f"ndcg@{k}"] = float(np.mean([M.ndcg_at_k(r, rel[u], k) for r, u in zip(recs, users)]))
        out[f"coverage@{k}"] = M.coverage(recs, len(i2i))
        out[f"diversity@{k}"] = M.diversity(recs, item_cat)
    out["n_eval_users"] = len(users)
    # explicit rating prediction (RMSE / MAE) on all test rows
    if hasattr(model, "predict_rating"):
        u = test.user_id.map(u2i).to_numpy()
        i = test.item_id.map(i2i).to_numpy()
        pred = model.predict_rating(u, i)
        out["rmse"], out["mae"] = M.rmse(test.rating, pred), M.mae(test.rating, pred)
    return out


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    proc, mdir, rep = path(cfg, "processed"), path(cfg, "models"), path(cfg, "reports")
    tc = cfg["training"]
    thr = cfg["preparation"]["positive_rating_threshold"]
    test = pd.read_parquet(proc / "test.parquet")
    items = pd.read_parquet(proc / "items.parquet")
    maps = load_json(proc / "id_mappings.json")
    train_mat = sp.load_npz(proc / "user_item_matrix.npz").tocsr()
    item_cat = dict(zip(items.item_id, items.category))
    run_ids = load_json(mdir / "run_ids.json")

    mlflow.set_tracking_uri(f"sqlite:///{path(cfg, 'models').parent / 'mlflow.db'}")
    rows = {}
    for name, run_id in run_ids.items():
        with open(mdir / f"{name}.pkl", "rb") as f:
            model = pickle.load(f)
        res = evaluate_model(model, tc["top_k"], test, maps, train_mat, item_cat, thr)
        rows[name] = res
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in res.items() if k != "n_eval_users"})
        log.info("%s: %s", name, {k: round(v, 4) for k, v in res.items()})

    df = pd.DataFrame(rows).T
    df.index.name = "model"
    df.to_csv(rep / "model_results.csv")
    _write_md(df, tc, rep)
    return {"results": str(rep / "model_results.csv")}


def _write_md(df: pd.DataFrame, tc: dict, rep) -> None:
    lines = ["# Model Evaluation Results", "",
             f"Time-based hold-out test set. Relevant = rating >= threshold. Evaluated on {int(df['n_eval_users'].iloc[0])} users.", ""]
    for k in tc["top_k"]:
        cols = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"coverage@{k}", f"diversity@{k}"]
        lines += [f"## K = {k}", "", "| Model | Precision | Recall | NDCG | Coverage | Diversity |", "|---|---:|---:|---:|---:|---:|"]
        for m, r in df[cols].iterrows():
            lines.append(f"| {m} | " + " | ".join(f"{v:.4f}" for v in r) + " |")
        lines.append("")
    lines += ["## Rating prediction (explicit ratings)", "", "| Model | RMSE | MAE |", "|---|---:|---:|"]
    for m, r in df[["rmse", "mae"]].iterrows():
        lines.append(f"| {m} | " + " | ".join("n/a" if pd.isna(v) else f"{v:.4f}" for v in r) + " |")
    (rep / "model_results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
