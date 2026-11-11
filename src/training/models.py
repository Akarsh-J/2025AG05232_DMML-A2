"""Recommender models. All expose  recommend(user_idx, k, seen)  and (optionally)  predict_rating."""
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def _top_k(scores: np.ndarray, seen: sp.csr_matrix, k: int) -> np.ndarray:
    """Mask already-seen items, return top-k item indices per row of `scores`."""
    scores = scores.copy()
    seen = seen.tocoo()
    scores[seen.row, seen.col] = -np.inf
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.take_along_axis(scores, part, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(part, order, axis=1)


class PopularityRecommender:
    """Baseline: same ranking for everyone = Bayesian-weighted rating x log(popularity).
    Uses the item features from the feature store (item_weighted_rating, item_popularity)."""
    name = "popularity_baseline"

    def fit(self, item_features, item_to_idx, global_mean):
        n = len(item_to_idx)
        self.score = np.zeros(n)
        self.item_mean = np.full(n, global_mean)
        for r in item_features.itertuples():
            i = item_to_idx[r.item_id]
            self.score[i] = r.item_popularity * r.item_weighted_rating  # popularity-weighted quality
            self.item_mean[i] = r.item_weighted_rating
        self.global_mean = global_mean
        return self

    def recommend(self, user_idx, k, seen):
        scores = np.tile(self.score, (len(user_idx), 1))
        return _top_k(scores, seen[user_idx], k)

    def predict_rating(self, u, i):
        return self.item_mean[i]


class ItemKNNRecommender:
    """Personalised item-based collaborative filtering (cosine similarity, shrinkage, top-N neighbours)."""
    name = "item_knn_cf"

    def __init__(self, n_neighbors=50, shrinkage=10):
        self.n_neighbors, self.shrinkage = n_neighbors, shrinkage

    def fit(self, binary: sp.csr_matrix):
        m = binary.tocsr().astype(float)
        co = (m.T @ m).toarray()                                    # co-occurrence counts
        norm = normalize(m, axis=0)
        sim = (norm.T @ norm).toarray() * (co / (co + self.shrinkage))
        np.fill_diagonal(sim, 0)
        keep = np.argpartition(-sim, self.n_neighbors, axis=1)[:, :self.n_neighbors]
        sparse_sim = np.zeros_like(sim)
        rows = np.arange(sim.shape[0])[:, None]
        sparse_sim[rows, keep] = sim[rows, keep]
        self.sim = sparse_sim
        self.binary = m
        return self

    def recommend(self, user_idx, k, seen):
        scores = np.asarray((self.binary[user_idx] @ self.sim))
        return _top_k(scores, seen[user_idx], k)


class SVDRecommender:
    """Personalised matrix factorisation.
    * ranking:  TruncatedSVD on the binary positive-interaction matrix (PureSVD)
    * ratings:  biased MF  r_hat = mu + b_u + b_i + p_u . q_i  trained with SGD on explicit ratings"""
    name = "svd_matrix_factorization"

    def __init__(self, n_components=32, n_factors=16, epochs=25, lr=0.01, reg=0.05, seed=42):
        self.n_components, self.n_factors, self.epochs, self.lr, self.reg, self.seed = \
            n_components, n_factors, epochs, lr, reg, seed

    def fit(self, binary: sp.csr_matrix, ratings: sp.csr_matrix):
        svd = TruncatedSVD(self.n_components, random_state=self.seed)
        self.user_f = svd.fit_transform(binary)
        self.item_f = svd.components_.T
        self._fit_rating_mf(ratings.tocoo())
        return self

    def _fit_rating_mf(self, coo):
        rng = np.random.default_rng(self.seed)
        u, i, r = coo.row, coo.col, coo.data
        n_u, n_i = coo.shape
        self.mu = r.mean()
        self.bu, self.bi = np.zeros(n_u), np.zeros(n_i)
        self.P = rng.normal(0, 0.1, (n_u, self.n_factors))
        self.Q = rng.normal(0, 0.1, (n_i, self.n_factors))
        idx = np.arange(len(r))
        for _ in range(self.epochs):
            rng.shuffle(idx)
            for t in idx:
                a, b = u[t], i[t]
                err = r[t] - (self.mu + self.bu[a] + self.bi[b] + self.P[a] @ self.Q[b])
                self.bu[a] += self.lr * (err - self.reg * self.bu[a])
                self.bi[b] += self.lr * (err - self.reg * self.bi[b])
                pa = self.P[a].copy()
                self.P[a] += self.lr * (err * self.Q[b] - self.reg * pa)
                self.Q[b] += self.lr * (err * pa - self.reg * self.Q[b])

    def recommend(self, user_idx, k, seen):
        scores = self.user_f[user_idx] @ self.item_f.T
        return _top_k(scores, seen[user_idx], k)

    def predict_rating(self, u, i):
        pred = self.mu + self.bu[u] + self.bi[i] + np.einsum("ij,ij->i", self.P[u], self.Q[i])
        return np.clip(pred, 1, 5)
