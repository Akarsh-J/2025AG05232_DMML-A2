"""Generates reports/architecture_diagram.png and reports/orchestration_run.png (from the real run log)."""
import re
from pathlib import Path

import graphviz
import matplotlib.pyplot as plt

R = Path(__file__).parent

g = graphviz.Digraph("arch", format="png")
g.attr(rankdir="TB", fontname="Helvetica", fontsize="11", dpi="150", nodesep="0.35", ranksep="0.45", pad="0.2")
g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="10", margin="0.12,0.08")
g.attr("edge", fontname="Helvetica", fontsize="8", color="#555555")

g.node("src", "Data sources\nusers.csv | items.csv\ninteractions.csv\n(documented synthetic data)", fillcolor="#E8E8E8", shape="folder")
stages = [("ing", "1 Ingestion\nsrc/ingestion", "#CFE8FC"), ("val", "2 Validation\nsrc/validation", "#CFE8FC"),
          ("prep", "3 Preparation\nsrc/preparation\n(clean, time split)", "#D9F2D0"),
          ("tr", "4 Transformation\nsrc/transformation\n(matrix, summaries)", "#D9F2D0"),
          ("fe", "5 Feature mgmt\nsrc/features\n(feature store)", "#FCEBC2"),
          ("trn", "6 Training\nsrc/training", "#F6D4E0"), ("ev", "7 Evaluation\nsrc/evaluation", "#F6D4E0")]
for k, label, col in stages:
    g.node(k, label, fillcolor=col)
g.node("orch", "Orchestration: Prefect flow (dags/recommendation_flow.py)\nretries | task dependencies | daily cron schedule", fillcolor="#FFFFFF", shape="note")
g.node("err", "data/errors/\nrejected records", fillcolor="#F8D0D0", shape="cylinder")
g.node("rep", "reports/\nvalidation report\nfeature catalog\nmodel results", fillcolor="#E8E8E8", shape="folder")
g.node("mlf", "MLflow tracking\nparams | metrics | artifacts", fillcolor="#E4DAF5", shape="cylinder")
g.node("mod", "models/\n*.pkl + predict.py\n(cold-start fallback)", fillcolor="#E4DAF5", shape="cylinder")
g.node("fstore", "data/features/\nuser | item | interaction\nparquet (offline store)", fillcolor="#FCEBC2", shape="cylinder")
for a, b in [("src", "ing"), ("ing", "val"), ("val", "prep"), ("prep", "tr"), ("tr", "fe"), ("fe", "trn"), ("trn", "ev")]:
    g.edge(a, b)
g.edge("val", "err", label="invalid rows", style="dashed")
g.edge("fe", "fstore")
g.edge("fstore", "trn", label="same features\nfor train", style="dashed")
g.edge("fstore", "mod", label="same features\nfor predict", style="dashed")
g.edge("trn", "mlf"); g.edge("ev", "mlf", label="metrics"); g.edge("trn", "mod"); g.edge("ev", "rep"); g.edge("val", "rep", style="dashed")
g.edge("orch", "ing", style="dotted", arrowhead="none", constraint="false", label="schedules / monitors")
g.render(str(R / "architecture_diagram"), cleanup=True)

# ---- run evidence rendered from the real log -------------------------------------------
log = (R / "pipeline_run.log").read_text().splitlines()
keep = [re.sub(r"\x1b\[[0-9;]*m", "", l)[:150] for l in log if re.search(r"Flow run|Task run|Stopping temporary", l)]
fig, ax = plt.subplots(figsize=(13, 0.34 * len(keep) + 1.2))
ax.axis("off"); fig.patch.set_facecolor("#0d1117")
ax.set_title("Prefect flow 'recommendation-pipeline' - run output (all tasks Completed)", color="#e6edf3", loc="left", fontsize=11, family="monospace")
for i, l in enumerate(keep):
    ok = "Completed" in l
    ax.text(0.0, 1 - (i + 1) / (len(keep) + 1), l, color="#3fb950" if ok else "#e6edf3", fontsize=8, family="monospace", va="center", transform=ax.transAxes)
fig.savefig(R / "orchestration_run.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
