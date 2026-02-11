

from __future__ import annotations



import math

from dataclasses import dataclass

from typing import Iterable, List, Sequence, Tuple, Dict, Any



import numpy as np

import pandas as pd





def _hash_feat(s: str, dim: int) -> int:

    return (hash(s) % dim + dim) % dim





def featurize_rows(df: pd.DataFrame, *, origin: str, dest: str, date_col: str | None, dim: int = 512) -> np.ndarray:

    """Feature hashing for simple shift classification.

    Features:
      - origin, dest (categorical)
      - origin-dest pair
      - day-of-week, month (if date available)
    """

    n = len(df)

    X = np.zeros((n, dim), dtype=np.float32)



    o = df[origin].astype(str).str.upper().str.strip()

    d = df[dest].astype(str).str.upper().str.strip()



    for i in range(n):

        feats = [

            f"o={o.iat[i]}",

            f"d={d.iat[i]}",

            f"od={o.iat[i]}_{d.iat[i]}",

        ]

        if date_col is not None and date_col in df.columns:

            dt = pd.to_datetime(df[date_col].iat[i], errors="coerce")

            if not pd.isna(dt):

                feats.append(f"m={int(dt.month)}")

                feats.append(f"dow={int(dt.dayofweek)}")

        for f in feats:

            j = _hash_feat(f, dim)

            X[i, j] += 1.0



    X[:, 0] += 1.0

    return X





@dataclass

class ShiftClassifier:

    """Tiny logistic regression trained with SGD for importance weighting."""



    dim: int = 512

    lr: float = 0.2

    iters: int = 200

    l2: float = 1e-3

    seed: int = 0

    w: np.ndarray | None = None



    def fit(self, X: np.ndarray, y: np.ndarray) -> "ShiftClassifier":

        rng = np.random.default_rng(self.seed)

        n, d = X.shape

        w = np.zeros(d, dtype=np.float32)



        for _ in range(self.iters):

            idx = rng.integers(0, n, size=min(n, 1024))

            Xb = X[idx]

            yb = y[idx].astype(np.float32)



            logits = Xb @ w

            p = 1.0 / (1.0 + np.exp(-logits))

            grad = (Xb.T @ (p - yb)) / len(idx) + self.l2 * w

            w -= self.lr * grad

        self.w = w

        return self



    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        if self.w is None:

            raise ValueError("Classifier not fitted.")

        logits = X @ self.w

        p = 1.0 / (1.0 + np.exp(-logits))

        return np.clip(p, 1e-6, 1 - 1e-6)



    def importance_weights(self, X: np.ndarray, *, prior_test: float) -> np.ndarray:

        """Return w(x) proportional to p(test|x) / p(calib|x)."""

        p = self.predict_proba(X)

        odds = p / (1.0 - p)



        prior_cal = max(1e-6, 1.0 - prior_test)

        odds *= prior_cal / max(1e-6, prior_test)



        return np.clip(odds, 0.05, 20.0)

