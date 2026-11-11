"""Stage 8 - Orchestration with Prefect.

Task dependency graph (arrows = data/ordering dependency, enforced by passing each task's
output into the next one):

  ingest -> validate -> prepare -> transform -> build_features -> train -> evaluate -> report

Run once:        python dags/recommendation_flow.py
Run on schedule: python dags/recommendation_flow.py --serve      (daily 02:00, cron "0 2 * * *")
Reproducibility: all randomness is seeded via configs/config.yaml; every task is idempotent
                 (re-running overwrites its outputs), and failed tasks are retried once.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")  # no telemetry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefect import flow, get_run_logger, task  # noqa: E402

from src.common import load_config  # noqa: E402
from src.evaluation import evaluate as evaluation_stage  # noqa: E402
from src.features import feature_store  # noqa: E402
from src.ingestion import ingest as ingestion_stage  # noqa: E402
from src.preparation import prepare as preparation_stage  # noqa: E402
from src.training import train as training_stage  # noqa: E402
from src.transformation import transform as transformation_stage  # noqa: E402
from src.validation import validate as validation_stage  # noqa: E402

RETRY = dict(retries=1, retry_delay_seconds=5)


@task(name="1_ingest", **RETRY)
def ingest(cfg):
    return ingestion_stage.run(cfg)


@task(name="2_validate", **RETRY)
def validate(cfg, upstream):
    return validation_stage.run(cfg)


@task(name="3_prepare", **RETRY)
def prepare(cfg, upstream):
    return preparation_stage.run(cfg)


@task(name="4_transform", **RETRY)
def transform(cfg, upstream):
    return transformation_stage.run(cfg)


@task(name="5_build_features", **RETRY)
def build_features(cfg, upstream):
    return feature_store.run(cfg)


@task(name="6_train_models", **RETRY)
def train(cfg, upstream):
    return training_stage.run(cfg)


@task(name="7_evaluate_models", **RETRY)
def evaluate(cfg, upstream):
    return evaluation_stage.run(cfg)


@task(name="8_report")
def report(cfg, upstream):
    log = get_run_logger()
    root = Path(__file__).resolve().parents[1]
    for f in ["data_validation_report.md", "feature_catalog.md", "model_results.md"]:
        assert (root / cfg["paths"]["reports"] / f).exists(), f"missing report {f}"
    log.info("All reports generated in %s/", cfg["paths"]["reports"])
    return "ok"


@flow(name="recommendation-pipeline", log_prints=True)
def recommendation_pipeline(config_path: str | None = None):
    cfg = load_config(config_path)
    a = ingest(cfg)
    b = validate(cfg, a)
    c = prepare(cfg, b)
    d = transform(cfg, c)
    e = build_features(cfg, d)
    f = train(cfg, e)
    g = evaluate(cfg, f)
    return report(cfg, g)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true", help="run as a daily scheduled deployment")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    if args.serve:
        recommendation_pipeline.serve(name="daily-recsys", cron="0 2 * * *")
    else:
        recommendation_pipeline(args.config)
