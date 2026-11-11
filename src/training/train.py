"""Stage 6 - Model training with MLflow experiment tracking.

Trains: popularity baseline, item-kNN CF, SVD matrix factorisation.
For every model MLflow records parameters, training metadata, and the pickled model artifact.
Metrics are logged by the evaluation stage into the same run (run ids are stored in models/run_ids.json).
"""
import pickle

import mlflow
import pandas as pd
import scipy.sparse as sp

from src.common import get_logger, load_config, load_json, path, save_json
from src.features.feature_store import FeatureStore
from src.training.models import ItemKNNRecommender, PopularityRecommender, SVDRecommender

log = get_logger("training")


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    proc, mdir = path(cfg, "processed"), path(cfg, "models")
    tc = cfg["training"]
    maps = load_json(proc / "id_mappings.json")
    ratings = sp.load_npz(proc / "user_item_matrix.npz").tocsr()
    binary = sp.load_npz(proc / "user_item_binary.npz").tocsr()
    train = pd.read_parquet(proc / "train.parquet")
    fs = FeatureStore(cfg)                                    # same feature layer used at serving time

    mlflow.set_tracking_uri(f"sqlite:///{mdir.parent / 'mlflow.db'}")
    mlflow.set_experiment(tc["mlflow_experiment"])

    specs = {
        "popularity_baseline": (lambda: PopularityRecommender().fit(fs.item_features, maps["item_to_idx"], train.rating.mean()), {"strategy": "popularity x weighted rating"}),
        "item_knn_cf": (lambda: ItemKNNRecommender(**tc["item_knn"]).fit(binary), tc["item_knn"]),
        "svd_matrix_factorization": (lambda: SVDRecommender(**tc["svd"], **tc["mf_rating"], seed=cfg["seed"]).fit(binary, ratings),
                                     {**tc["svd"], **tc["mf_rating"]}),
    }
    run_ids = {}
    for name, (build, params) in specs.items():
        with mlflow.start_run(run_name=name) as r:
            mlflow.log_params({**params, "seed": cfg["seed"], "n_train_rows": len(train),
                               "positive_threshold": cfg["preparation"]["positive_rating_threshold"]})
            model = build()
            with open(mdir / f"{name}.pkl", "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(str(mdir / f"{name}.pkl"), artifact_path="model")
            run_ids[name] = r.info.run_id
            log.info("Trained %s (run %s)", name, r.info.run_id[:8])
    save_json(run_ids, mdir / "run_ids.json")
    return {"models": list(run_ids)}


if __name__ == "__main__":
    run()
