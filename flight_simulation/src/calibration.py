from __future__ import annotations



import math

from dataclasses import dataclass

from typing import List, Sequence, Tuple, Iterable





def _quantile(xs: Sequence[float], q: float) -> float:

    if not xs:

        return float("nan")

    ys = sorted(float(x) for x in xs)

    if q <= 0:

        return ys[0]

    if q >= 1:

        return ys[-1]

    pos = (len(ys) - 1) * q

    lo = int(math.floor(pos))

    hi = int(math.ceil(pos))

    if lo == hi:

        return ys[lo]

    w = pos - lo

    return ys[lo] * (1 - w) + ys[hi] * w





@dataclass(frozen=True)

class ConformalInterval:

    """Conformalized prediction interval using CQR-style scores."""



    alpha: float

    score_quantile: float



    def apply(self, q_lo: float, q_hi: float) -> Tuple[float, float]:

        s = float(self.score_quantile)

        return float(q_lo - s), float(q_hi + s)



    def __getitem__(self, key: str):



        if key in ("q", "score_quantile"):

            return float(self.score_quantile)

        if key == "alpha":

            return float(self.alpha)

        raise KeyError(key)





def conformalize_intervals(

    y_calib: Sequence[float],

    q_lo_calib: Sequence[float],

    q_hi_calib: Sequence[float],

    *,

    alpha: float,

) -> ConformalInterval:

    if len(y_calib) != len(q_lo_calib) or len(y_calib) != len(q_hi_calib):

        raise ValueError("Calibration arrays must have the same length.")

    if not (0 < alpha < 1):

        raise ValueError("alpha must be in (0, 1).")



    scores: List[float] = []

    for y, lo, hi in zip(y_calib, q_lo_calib, q_hi_calib):

        y = float(y)

        lo = float(lo)

        hi = float(hi)

        scores.append(max(lo - y, y - hi, 0.0))



    s_q = _quantile(scores, 1.0 - alpha)

    return ConformalInterval(alpha=alpha, score_quantile=float(s_q))





def interval_coverage(y: Sequence[float], q_lo: Sequence[float], q_hi: Sequence[float]) -> float:

    if len(y) != len(q_lo) or len(y) != len(q_hi) or len(y) == 0:

        return float("nan")

    covered = 0

    for yi, lo, hi in zip(y, q_lo, q_hi):

        if float(lo) <= float(yi) <= float(hi):

            covered += 1

    return covered / len(y)





def build_baseline_intervals_from_samples(

    sample_sets: Iterable[Sequence[float]] | Sequence[float],

    *,

    alpha: float,

) -> Tuple[List[float], List[float]]:

    """Compute (1-alpha) prediction intervals from sample draws.

    Accepts either:
      - an iterable of sample-sets (shape: n_obs x n_draws)
      - a single sample-set (shape: n_draws)

    Returns two lists (lo, hi) with length n_obs (or length 1 for a single set).
    """

    lo_q = alpha / 2.0

    hi_q = 1.0 - alpha / 2.0





    try:

        import numpy as _np

        if isinstance(sample_sets, _np.ndarray) and sample_sets.ndim == 1:

            return [float(_np.quantile(sample_sets, lo_q))], [float(_np.quantile(sample_sets, hi_q))]

    except Exception:

        pass





    try:

        it = iter(sample_sets)

    except TypeError:



        v = float(sample_sets)

        return [v], [v]



    try:

        first = next(it)

    except StopIteration:

        return [], []





    if isinstance(first, (int, float)):

        all_vals = [float(first)] + [float(x) for x in it]

        return [_quantile(all_vals, lo_q)], [_quantile(all_vals, hi_q)]





    q_lo: List[float] = []

    q_hi: List[float] = []

    q_lo.append(_quantile(first, lo_q))

    q_hi.append(_quantile(first, hi_q))

    for s in it:

        q_lo.append(_quantile(s, lo_q))

        q_hi.append(_quantile(s, hi_q))

    return q_lo, q_hi





def _weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:

    if len(values) == 0:

        return float("nan")

    if len(values) != len(weights):

        raise ValueError("values and weights must have the same length.")

    if q <= 0:

        return float(min(values))

    if q >= 1:

        return float(max(values))

    pairs = sorted(((float(v), float(w)) for v, w in zip(values, weights)), key=lambda t: t[0])

    tot_w = sum(max(0.0, w) for _, w in pairs)

    if tot_w <= 0:



        return _quantile([v for v, _ in pairs], q)

    thresh = q * tot_w

    c = 0.0

    for v, w in pairs:

        w = max(0.0, w)

        c += w

        if c >= thresh:

            return float(v)

    return float(pairs[-1][0])





def weighted_conformalize_intervals(

    y_calib: Sequence[float],

    q_lo_calib: Sequence[float],

    q_hi_calib: Sequence[float],

    weights: Sequence[float],

    *,

    alpha: float,

) -> ConformalInterval:

    """Weighted conformal calibration for covariate/time shift.

    Computes the (1-alpha) score quantile using a weighted quantile of conformity scores.
    """

    if len(y_calib) != len(q_lo_calib) or len(y_calib) != len(q_hi_calib) or len(y_calib) != len(weights):

        raise ValueError("Calibration arrays must have the same length.")

    scores: List[float] = []

    for y, lo, hi in zip(y_calib, q_lo_calib, q_hi_calib):

        y = float(y); lo = float(lo); hi = float(hi)

        scores.append(max(lo - y, y - hi, 0.0))

    s_q = _weighted_quantile(scores, list(weights), 1.0 - alpha)

    return ConformalInterval(alpha=alpha, score_quantile=float(s_q))





def build_conformal_intervals_from_samples(

    y_calib: Sequence[float],

    sample_sets_calib: Iterable[Sequence[float]] | Sequence[float],

    sample_sets_test: Iterable[Sequence[float]] | Sequence[float],

    *,

    alpha: float,

) -> Tuple[List[float], List[float], ConformalInterval]:

    """Baseline intervals from samples + conformal expansion calibrated on y_calib."""

    lo_cal, hi_cal = build_baseline_intervals_from_samples(sample_sets_calib, alpha=alpha)

    lo_te, hi_te = build_baseline_intervals_from_samples(sample_sets_test, alpha=alpha)



    conf = conformalize_intervals(y_calib, lo_cal, hi_cal, alpha=alpha)

    q = float(conf.score_quantile)

    lo_adj = [float(x) - q for x in lo_te]

    hi_adj = [float(x) + q for x in hi_te]

    return lo_adj, hi_adj, conf





def build_weighted_conformal_intervals_from_samples(

    y_calib: Sequence[float],

    sample_sets_calib: Iterable[Sequence[float]] | Sequence[float],

    sample_sets_test: Iterable[Sequence[float]] | Sequence[float],

    weights: Sequence[float],

    *,

    alpha: float,

) -> Tuple[List[float], List[float], ConformalInterval]:

    lo_cal, hi_cal = build_baseline_intervals_from_samples(sample_sets_calib, alpha=alpha)

    lo_te, hi_te = build_baseline_intervals_from_samples(sample_sets_test, alpha=alpha)



    conf = weighted_conformalize_intervals(y_calib, lo_cal, hi_cal, weights, alpha=alpha)

    q = float(conf.score_quantile)

    lo_adj = [float(x) - q for x in lo_te]

    hi_adj = [float(x) + q for x in hi_te]

    return lo_adj, hi_adj, conf





def pinball_loss(y: Sequence[float], q_pred: Sequence[float], tau: float) -> float:

    if len(y) != len(q_pred) or len(y) == 0:

        return float("nan")

    tot = 0.0

    for yi, qi in zip(y, q_pred):

        u = float(yi) - float(qi)

        tot += max(tau * u, (tau - 1.0) * u)

    return tot / len(y)





def median_from_samples(sample_sets: Iterable[Sequence[float]]) -> List[float]:

    return [float(_quantile(s, 0.5)) for s in sample_sets]

