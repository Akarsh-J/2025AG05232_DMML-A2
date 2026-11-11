"""Convenience entry point:  python run_pipeline.py   (runs the full Prefect flow end-to-end)."""
from dags.recommendation_flow import recommendation_pipeline

if __name__ == "__main__":
    recommendation_pipeline()
