# Feature Catalog

Generated from `FEATURE_REGISTRY` in `src/features/feature_store.py`. Features are computed from the **train split only**, "as of" the last train timestamp, with the same code path used for training and prediction.

| Entity | Feature | Type | Description | Logic |
|---|---|---|---|---|
| user | `user_avg_rating` | float | Average rating the user has given | `mean(rating) over user's train interactions` |
| user | `user_rating_std` | float | Rating variability (0 if single rating) | `std(rating)` |
| user | `user_activity_count` | int | Number of interactions by the user | `count(interactions)` |
| user | `user_positive_rate` | float | Share of ratings considered positive | `mean(rating >= positive_threshold)` |
| user | `user_preferred_category` | string | Category the user interacted with most (tie -> highest avg rating) | `mode(item.category)` |
| user | `user_n_categories` | int | Number of distinct categories explored | `nunique(item.category)` |
| user | `user_days_since_last_interaction` | float | Days between user's last interaction and as_of date | `as_of - max(timestamp)` |
| user | `user_tenure_days` | int | Days since signup at as_of date | `as_of - signup_date` |
| item | `item_popularity` | int | Number of interactions the item received | `count(interactions)` |
| item | `item_avg_rating` | float | Average rating received (raw) | `mean(rating)` |
| item | `item_weighted_rating` | float | Bayesian-smoothed average rating (shrunk toward global mean) | `(n*avg + m*global_avg)/(n+m)` |
| item | `item_positive_rate` | float | Share of positive ratings | `mean(rating >= positive_threshold)` |
| item | `item_days_since_last_interaction` | float | Days since the item was last interacted with | `as_of - max(timestamp)` |
| item | `item_category` | string | Item category (standardised) | `items.category` |
| item | `item_price_bucket` | string | Price quartile bucket (low/mid/high/premium) | `qcut(price, 4)` |
| pair | `category_match` | int | 1 if item category == user's preferred category | `item_category == user_preferred_category` |
| pair | `rating_gap_vs_user_avg` | float | Item avg rating minus user avg rating | `item_avg_rating - user_avg_rating` |
| pair | `days_since_interaction` | float | Days between this interaction and as_of (training pairs) | `as_of - timestamp` |
| pair | `recency_weight` | float | Exponential time-decay weight of an interaction | `0.5 ** (days / half_life)` |
