# Model Evaluation Results

Time-based hold-out test set. Relevant = rating >= threshold. Evaluated on 1612 users.

## K = 5

| Model | Precision | Recall | NDCG | Coverage | Diversity |
|---|---:|---:|---:|---:|---:|
| popularity_baseline | 0.0596 | 0.1028 | 0.0966 | 0.0388 | 0.8239 |
| item_knn_cf | 0.1599 | 0.2417 | 0.2361 | 0.3204 | 0.2726 |
| svd_matrix_factorization | 0.1185 | 0.1731 | 0.1725 | 0.4204 | 0.4861 |

## K = 10

| Model | Precision | Recall | NDCG | Coverage | Diversity |
|---|---:|---:|---:|---:|---:|
| popularity_baseline | 0.0484 | 0.1643 | 0.1190 | 0.0592 | 0.8352 |
| item_knn_cf | 0.1192 | 0.3621 | 0.2767 | 0.4755 | 0.3121 |
| svd_matrix_factorization | 0.0916 | 0.2769 | 0.2080 | 0.5449 | 0.5340 |

## Rating prediction (explicit ratings)

| Model | RMSE | MAE |
|---|---:|---:|
| popularity_baseline | 0.9024 | 0.7341 |
| item_knn_cf | n/a | n/a |
| svd_matrix_factorization | 0.8191 | 0.6510 |
