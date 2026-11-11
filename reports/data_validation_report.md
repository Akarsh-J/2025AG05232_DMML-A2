# Data Validation Report

Generated automatically by `src/validation/validate.py`. `reject` = row moved to `data/errors/`; `warn` = row kept, repaired in preparation.

## Row counts

| Table | Input rows | Valid rows | Rejected rows | Rejected % |
|---|---:|---:|---:|---:|
| users | 2020 | 1970 | 50 | 2.48 |
| items | 505 | 490 | 15 | 2.97 |
| interactions | 53042 | 48562 | 4480 | 8.45 |

## Check results

| Table | Check | Severity | Violations | % of rows |
|---|---|---|---:|---:|
| users | missing_user_id | reject | 0 | 0.0 |
| users | duplicate_user_id | reject | 20 | 0.99 |
| users | age_wrong_type | reject | 20 | 0.99 |
| users | age_out_of_range | reject | 10 | 0.495 |
| users | invalid_signup_date | reject | 0 | 0.0 |
| users | missing_age | warn | 82 | 4.059 |
| users | missing_country | warn | 60 | 2.97 |
| items | missing_item_id | reject | 0 | 0.0 |
| items | duplicate_item_id | reject | 5 | 0.99 |
| items | price_wrong_type | reject | 5 | 0.99 |
| items | negative_price | reject | 5 | 0.99 |
| items | missing_category | warn | 21 | 4.158 |
| interactions | missing_user_id | reject | 234 | 0.441 |
| interactions | missing_item_id | reject | 0 | 0.0 |
| interactions | missing_rating | reject | 317 | 0.598 |
| interactions | rating_wrong_type | reject | 158 | 0.298 |
| interactions | rating_out_of_range | reject | 391 | 0.737 |
| interactions | invalid_timestamp_format | reject | 158 | 0.298 |
| interactions | timestamp_in_future | reject | 318 | 0.6 |
| interactions | timestamp_too_old | reject | 79 | 0.149 |
| interactions | duplicate_interaction | reject | 551 | 1.039 |
| interactions | orphan_user_id (referential integrity) | reject | 1012 | 1.908 |
| interactions | orphan_item_id (referential integrity) | reject | 1380 | 2.602 |

Note: one row can violate several checks, so violations can exceed rejected rows.
