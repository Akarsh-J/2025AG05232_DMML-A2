"""Builds reports/final_report.pdf (2-3 pages) from the real pipeline outputs."""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

R = Path(__file__).parent
res = pd.read_csv(R / "model_results.csv", index_col=0)
val = json.loads((R / "data_validation_report.json").read_text())["totals"]

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=11.8, spaceAfter=4)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, spaceAfter=2)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11, spaceBefore=6, spaceAfter=3, textColor=colors.HexColor("#1f3a5f"))
small = ParagraphStyle("s", parent=body, fontSize=7.8, leading=9.5, textColor=colors.HexColor("#444444"))
bul = ParagraphStyle("bul", parent=body, leftIndent=10, bulletIndent=0, spaceAfter=2)


def table(data, widths, header=True):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    st = [("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st))
    return t


def b(txt):
    return Paragraph(txt, bul, bulletText="\u2022")


S = []
S += [Paragraph("End-to-End Data Management Pipeline for a Recommendation System", h1),
      Paragraph("Data Management for Machine Learning - Assignment 2 | Name: ____________ | Roll No: ____________ | "
                "GitHub: ____________", small), Spacer(1, 4)]

S += [Paragraph("1. Problem and data", h2),
      Paragraph("The goal is a reproducible pipeline that turns raw user, item and interaction data into evaluated "
                "recommendation models. I use a <b>documented synthetic e-commerce dataset</b> (2,000 users, 500 items in 10 "
                "categories, about 52k ratings from 1 to 5 over 2023-2024), generated from seeded latent factors so users have "
                "category preferences on top of a popularity skew. About 3% of records are deliberately corrupted (missing values, "
                "duplicates, wrong types, out-of-range ratings, invalid/future timestamps, orphan IDs) so that validation is "
                "meaningful rather than a no-op. The generator is isolated, so real CSVs (e.g. MovieLens converted to the same "
                "columns) can be dropped into <i>data/raw</i> without code changes.", body)]

bul_small = ParagraphStyle("bs", parent=bul, fontSize=8.6, leading=11)
decisions = [
    Paragraph("<b>Modular stages</b> (<i>src/ingestion ... evaluation</i>), each with a <i>run(cfg)</i> function so it can run alone or inside the "
              "flow. All parameters live in <i>configs/config.yaml</i>; all randomness is seeded, and a full re-run reproduced identical metrics.", bul_small, bulletText="\u2022"),
    Paragraph("<b>Validation with severities.</b> 23 rule checks. <i>Reject</i> rules (invalid keys, ratings, timestamps, duplicates, referential integrity) "
              "move rows to <i>data/errors/</i> with the reasons; <i>warn</i> rules (missing age/country/category) keep the row and are imputed in preparation.", bul_small, bulletText="\u2022"),
    Paragraph("<b>Leakage control.</b> A time-based split (latest 20% of interactions to test, with an assertion that max(train) &lt; min(test)); "
              "derived datasets and features are computed from train only, \"as of\" the last train timestamp.", bul_small, bulletText="\u2022"),
    Paragraph("<b>Feature layer.</b> A single feature registry documents every feature and generates the catalog; the same functions serve training and prediction, "
              "and the offline parquet store has lookup methods. Prediction uses these features and falls back to popularity for cold-start users.", bul_small, bulletText="\u2022"),
    Paragraph("<b>Orchestration.</b> A Prefect flow with 8 tasks chained by data dependency, one retry each, idempotent outputs and a daily cron option.", bul_small, bulletText="\u2022"),
]
img = Image(str(R / "architecture_diagram.png"), width=7.6 * cm, height=7.6 * cm * 1765 / 1218)
lay = Table([[img, decisions]], colWidths=[8.0 * cm, 9.6 * cm])
lay.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
S += [Paragraph("2. Architecture and design decisions", h2), lay]

v = val
S += [Paragraph("3. Tools", h2),
      Paragraph("Python, Pandas, NumPy/SciPy (sparse matrices), scikit-learn (TruncatedSVD), Parquet/CSV storage, a custom rule-based validation layer, "
                "a custom feature store, MLflow (SQLite backend) for parameters, metrics and model artifacts, Prefect for orchestration, pytest for unit tests, "
                "Graphviz/Matplotlib for diagrams. I chose custom validation and feature code over Great Expectations/Feast to keep the dependency footprint "
                "small and the logic transparent for a project of this size.", body),
      Paragraph("4. Data validation results", h2),
      table([["Table", "Input rows", "Valid", "Rejected", "Rejected %"]] +
            [[t, v[t]["input"], v[t]["valid"], v[t]["rejected"], f"{100 * v[t]['rejected'] / v[t]['input']:.2f}"] for t in v],
            [3.2 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm]),
      Spacer(1, 3),
      Paragraph("Interactions are the dirtiest table (largest rejection share), mostly from orphan item/user IDs, duplicates, future timestamps and "
                "invalid ratings. Rejecting a user for a bad field also orphans that user's interactions, which is a deliberate conservative choice. "
                "The per-check counts are in <i>reports/data_validation_report.md</i>.", body)]

S += [Paragraph("5. Features", h2),
      Paragraph("19 documented features across user (8), item (7) and user-item pair (4) levels, for example <i>user_avg_rating, user_activity_count, "
                "user_preferred_category, user_days_since_last_interaction, item_popularity, item_avg_rating, item_weighted_rating</i> (Bayesian-smoothed), "
                "<i>category_match</i> and <i>recency_weight</i>. The full table is in <i>reports/feature_catalog.md</i>. The popularity baseline ranks with "
                "<i>item_popularity x item_weighted_rating</i> from the feature store.", body)]

S += [Paragraph("6. Models and results", h2),
      Paragraph("Three models were trained and tracked in MLflow: (1) <b>popularity baseline</b>; (2) <b>item-kNN collaborative filtering</b> "
                "(cosine similarity with shrinkage, 50 neighbours); (3) <b>SVD matrix factorization</b> (TruncatedSVD with 32 components for ranking, plus a biased MF "
                "trained by SGD for rating prediction). Relevant items are test ratings of 4 or higher, and a user's already-seen items are excluded from recommendations. "
                "Diversity is the share of item pairs in a list that belong to different categories.", body)]
names = {"popularity_baseline": "Popularity (baseline)", "item_knn_cf": "Item-kNN CF", "svd_matrix_factorization": "SVD MF"}
rows = [["Model", "P@10", "R@10", "NDCG@10", "Cov@10", "Div@10", "P@5", "NDCG@5", "RMSE", "MAE"]]
for m, r in res.iterrows():
    rows.append([names[m], f"{r['precision@10']:.3f}", f"{r['recall@10']:.3f}", f"{r['ndcg@10']:.3f}", f"{r['coverage@10']:.3f}",
                 f"{r['diversity@10']:.3f}", f"{r['precision@5']:.3f}", f"{r['ndcg@5']:.3f}",
                 "n/a" if pd.isna(r["rmse"]) else f"{r['rmse']:.3f}", "n/a" if pd.isna(r["mae"]) else f"{r['mae']:.3f}"])
S += [table(rows, [3.1 * cm] + [1.62 * cm] * 9), Spacer(1, 3),
      Paragraph(f"Evaluated on {int(res['n_eval_users'].iloc[0])} test users with at least one relevant held-out item. Baseline RMSE/MAE uses the item's smoothed mean rating.", small)]
k, base, sv = res.loc["item_knn_cf"], res.loc["popularity_baseline"], res.loc["svd_matrix_factorization"]
S += [Spacer(1, 3),
      b(f"Both personalised models clearly beat the baseline: item-kNN improves NDCG@10 from {base['ndcg@10']:.3f} to {k['ndcg@10']:.3f} "
        f"({k['ndcg@10'] / base['ndcg@10']:.1f}x) and Recall@10 from {base['recall@10']:.3f} to {k['recall@10']:.3f}."),
      b(f"Item-kNN is best on accuracy; SVD MF is slightly weaker on ranking but has the best catalogue coverage ({sv['coverage@10']:.2f} vs {base['coverage@10']:.2f} for popularity) "
        f"and gives better rating predictions (RMSE {sv['rmse']:.3f} vs {base['rmse']:.3f}, MAE {sv['mae']:.3f} vs {base['mae']:.3f})."),
      b(f"The baseline has the highest category diversity ({base['diversity@10']:.2f}) simply because globally popular items span many categories, while personalised lists "
        f"concentrate on a user's preferred categories ({k['diversity@10']:.2f} for kNN). This accuracy-diversity trade-off is expected and would need re-ranking to balance.")]

S += [Paragraph("7. Limitations and future work", h2),
      b("Synthetic data: absolute scores are not comparable with real benchmarks, and the generator's structure (category-driven preferences) favours collaborative methods."),
      b("Single global time split and no hyper-parameter search; cold-start users and items are excluded from offline evaluation and only covered by a popularity fallback."),
      b("Diversity is category-based; richer content embeddings, novelty and fairness metrics, and an ALS/LightFM hybrid could be added."),
      b("Prefect runs locally; production would add data-quality gating (fail the flow above a rejection threshold), drift monitoring and a model registry.")]

doc = SimpleDocTemplate(str(R / "final_report.pdf"), pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm,
                        topMargin=1.5 * cm, bottomMargin=1.4 * cm, title="Recommendation Pipeline - Final Report")
doc.build(S)
print("built")
