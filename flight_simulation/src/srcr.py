from __future__ import annotations



import math

import random

from dataclasses import dataclass

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple



import pandas as pd



from .shift import ShiftClassifier, featurize_rows



from .calibration import (

    build_baseline_intervals_from_samples,

    build_conformal_intervals_from_samples,

    build_weighted_conformal_intervals_from_samples,

    conformalize_intervals,

    weighted_conformalize_intervals,

    interval_coverage,

)





def _clean_iata(x: object) -> str:

    s = str(x).strip().upper()

    return s





def _parse_delay_minutes(x: object) -> float:

    try:

        v = float(x)

    except Exception:

        return 0.0



    return max(0.0, v)





@dataclass(frozen=True)

class SRCRSplit:

    train: pd.DataFrame

    calib: pd.DataFrame

    test: pd.DataFrame

    has_time: bool





class ShiftRobustConformalDelayModel:

    """
    Shift-Robust Conformal Risk (SRCR) delay model.

    - Base model: empirical/quantile summaries + bootstrap sampling by (origin, dest).
    - Calibration: conformalized prediction intervals using nonconformity scores.
    - Shift-robustness: time-split evaluation; conservative per-origin calibration fallback.

    This is designed to work with public BTS-style CSVs or simple user logs.
    """



    def __init__(self, *, alpha: float = 0.1, min_group: int = 30, seed: int = 0):

        self.alpha = float(alpha)

        self.min_group = int(min_group)

        self._rng = random.Random(seed)



        self._global_delays: List[float] = []

        self._od_delays: Dict[Tuple[str, str], List[float]] = {}

        self._o_delays: Dict[str, List[float]] = {}





        self._r_global: float = 0.0

        self._r_o: Dict[str, float] = {}



    @staticmethod

    def detect_schema(df: pd.DataFrame) -> Dict[str, str]:

        cols = {c.lower(): c for c in df.columns}

        def pick(*names: str) -> Optional[str]:

            for n in names:

                if n.lower() in cols:

                    return cols[n.lower()]

            return None



        origin = pick("origin", "orig", "origin_airport", "origin_iata", "from", "from_airport", "ORIGIN")

        dest = pick("dest", "destination", "dst", "to", "to_airport", "DEST")

        delay = pick("arr_delay", "arrival_delay", "delay", "ARR_DELAY")

        date = pick("fl_date", "flight_date", "date", "FL_DATE")



        out = {}

        if origin: out["origin"] = origin

        if dest: out["dest"] = dest

        if delay: out["delay"] = delay

        if date: out["date"] = date

        return out



    @staticmethod

    def time_split(df: pd.DataFrame, date_col: str) -> SRCRSplit:

        d = df.copy()

        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")

        d = d.dropna(subset=[date_col])

        if d.empty:



            n = len(df)

            a = int(0.6 * n)

            b = int(0.8 * n)

            return SRCRSplit(train=df.iloc[:a], calib=df.iloc[a:b], test=df.iloc[b:], has_time=False)



        d = d.sort_values(date_col)

        n = len(d)

        a = int(0.6 * n)

        b = int(0.8 * n)

        return SRCRSplit(train=d.iloc[:a].copy(), calib=d.iloc[a:b].copy(), test=d.iloc[b:].copy(), has_time=True)



    def fit(self, df_raw: pd.DataFrame) -> "ShiftRobustConformalDelayModel":

        schema = self.detect_schema(df_raw)

        if not {"origin", "dest", "delay"}.issubset(schema):

            raise ValueError("Delay CSV must include origin, dest, and delay columns (schema autodetection failed).")



        ocol, dcol, ycol = schema["origin"], schema["dest"], schema["delay"]



        df = df_raw[[ocol, dcol, ycol] + ([schema["date"]] if "date" in schema else [])].copy()

        df[ocol] = df[ocol].map(_clean_iata)

        df[dcol] = df[dcol].map(_clean_iata)

        df[ycol] = df[ycol].map(_parse_delay_minutes)





        df = df[df[ocol].str.len() == 3]

        df = df[df[dcol].str.len() == 3]

        df = df.dropna(subset=[ocol, dcol])



        self._global_delays = df[ycol].astype(float).tolist()

        self._od_delays.clear()

        self._o_delays.clear()



        for (o, d), g in df.groupby([ocol, dcol]):

            ys = g[ycol].astype(float).tolist()

            if ys:

                self._od_delays[(o, d)] = ys

        for o, g in df.groupby(ocol):

            ys = g[ycol].astype(float).tolist()

            if ys:

                self._o_delays[o] = ys





        if "date" in schema:

            split = self.time_split(df, schema["date"])

            self._fit_calibration(split.calib, ocol, dcol, ycol)

        else:



            d2 = df.sample(frac=1.0, random_state=0)

            n = len(d2)

            b = int(0.8 * n)

            self._fit_calibration(d2.iloc[:b], ocol, dcol, ycol)



        return self



    def _bootstrap_samples(self, ys: Sequence[float], n: int, rng: random.Random) -> List[float]:

        if not ys:

            return [0.0] * n

        return [float(ys[rng.randrange(0, len(ys))]) for _ in range(n)]



    def samples_for(self, origin: str, dest: str, n: int, rng: Optional[random.Random] = None) -> List[float]:

        rng = rng or self._rng

        o, d = _clean_iata(origin), _clean_iata(dest)

        ys = self._od_delays.get((o, d))

        if ys and len(ys) >= self.min_group:

            return self._bootstrap_samples(ys, n, rng)

        ys = self._o_delays.get(o)

        if ys and len(ys) >= self.min_group:

            return self._bootstrap_samples(ys, n, rng)

        return self._bootstrap_samples(self._global_delays, n, rng)



    def interval_for(self, origin: str, dest: str, *, n: int = 800, alpha: Optional[float] = None) -> Tuple[float, float]:

        a = self.alpha if alpha is None else float(alpha)

        sims = self.samples_for(origin, dest, n=n)

        lo, hi = build_baseline_intervals_from_samples(sims, alpha=a)



        o = _clean_iata(origin)

        r = max(self._r_global, self._r_o.get(o, 0.0))

        return float(lo) - r, float(hi) + r



    def on_time_prob(self, origin: str, dest: str, *, threshold_min: float = 15.0, n: int = 2000) -> float:

        sims = self.samples_for(origin, dest, n=n)

        ok = sum(1 for x in sims if float(x) <= threshold_min)

        return ok / max(1, len(sims))



    def _fit_calibration(self, calib_df: pd.DataFrame, ocol: str, dcol: str, ycol: str) -> None:





        alpha = self.alpha





        scores_global: List[float] = []

        scores_by_o: Dict[str, List[float]] = {}



        rng = random.Random(12345)



        for _, r in calib_df.iterrows():

            o = _clean_iata(r[ocol])

            d = _clean_iata(r[dcol])

            y = float(r[ycol])



            sims = self.samples_for(o, d, n=400, rng=rng)

            lo, hi = build_baseline_intervals_from_samples(sims, alpha=alpha)

            lo_f, hi_f = float(lo), float(hi)

            s = max(lo_f - y, y - hi_f, 0.0)



            scores_global.append(s)

            scores_by_o.setdefault(o, []).append(s)





        if scores_global:

            self._r_global = float(pd.Series(scores_global).quantile(1.0 - alpha))

        else:

            self._r_global = 0.0



        self._r_o = {}

        for o, sc in scores_by_o.items():

            if len(sc) >= max(30, self.min_group // 2):

                self._r_o[o] = float(pd.Series(sc).quantile(1.0 - alpha))



    def evaluate_shift(self, df_raw: pd.DataFrame) -> Dict[str, float]:

        """Evaluate calibration under a time-based distribution shift.

        Returns baseline vs vanilla conformal vs weighted conformal coverage on the shifted test split.
        """

        schema = self.detect_schema(df_raw)

        if not {"origin", "dest", "delay"}.issubset(schema):

            raise ValueError("Delay CSV must include origin, dest, and delay columns.")

        if "date" not in schema:

            raise ValueError("Need a date column to evaluate distribution shift.")



        ocol, dcol, ycol, tcol = schema["origin"], schema["dest"], schema["delay"], schema["date"]

        df = df_raw[[ocol, dcol, ycol, tcol]].copy()

        df[ocol] = df[ocol].map(_clean_iata)

        df[dcol] = df[dcol].map(_clean_iata)

        df[ycol] = df[ycol].map(_parse_delay_minutes)



        split = self.time_split(df, tcol)



        max_cal = 1200

        max_te = 1200

        cal = split.calib.sample(n=min(max_cal, len(split.calib)), random_state=123) if len(split.calib) > max_cal else split.calib

        te = split.test.sample(n=min(max_te, len(split.test)), random_state=456) if len(split.test) > max_te else split.test



        rng = random.Random(999)

        n_draws = 350



        y_cal: List[float] = []

        lo_cal: List[float] = []

        hi_cal: List[float] = []



        for _, r in cal.iterrows():

            o = _clean_iata(r[ocol]); d = _clean_iata(r[dcol]); y = float(r[ycol])

            sims = self.samples_for(o, d, n=n_draws, rng=rng)

            lo, hi = build_baseline_intervals_from_samples(sims, alpha=self.alpha)

            y_cal.append(y); lo_cal.append(float(lo)); hi_cal.append(float(hi))



        y_te: List[float] = []

        lo_te: List[float] = []

        hi_te: List[float] = []

        for _, r in te.iterrows():

            o = _clean_iata(r[ocol]); d = _clean_iata(r[dcol]); y = float(r[ycol])

            sims = self.samples_for(o, d, n=n_draws, rng=rng)

            lo, hi = build_baseline_intervals_from_samples(sims, alpha=self.alpha)

            y_te.append(y); lo_te.append(float(lo)); hi_te.append(float(hi))



        conf = conformalize_intervals(y_cal, lo_cal, hi_cal, alpha=self.alpha)

        q = float(conf.score_quantile)

        lo_te_conf = [x - q for x in lo_te]

        hi_te_conf = [x + q for x in hi_te]



        prior_test = len(te) / max(1.0, (len(cal) + len(te)))

        X_cal = featurize_rows(cal, origin=ocol, dest=dcol, date_col=tcol, dim=512)

        X_te = featurize_rows(te, origin=ocol, dest=dcol, date_col=tcol, dim=512)



        X = np.vstack([X_cal, X_te])

        y_lbl = np.concatenate([np.zeros(len(X_cal), dtype=np.int32), np.ones(len(X_te), dtype=np.int32)])

        clf = ShiftClassifier(dim=512, lr=0.15, iters=220, l2=1e-3, seed=42).fit(X, y_lbl)

        w_cal = clf.importance_weights(X_cal, prior_test=prior_test).tolist()



        conf_w = weighted_conformalize_intervals(y_cal, lo_cal, hi_cal, w_cal, alpha=self.alpha)

        qw = float(conf_w.score_quantile)

        lo_te_w = [x - qw for x in lo_te]

        hi_te_w = [x + qw for x in hi_te]



        cov_base = interval_coverage(y_te, lo_te, hi_te)

        cov_conf = interval_coverage(y_te, lo_te_conf, hi_te_conf)

        cov_w = interval_coverage(y_te, lo_te_w, hi_te_w)



        return {

            "coverage_baseline": float(cov_base),

            "coverage_conformal": float(cov_conf),

            "coverage_weighted": float(cov_w),

            "n_test": float(len(y_te)),

            "has_time_split": 1.0,

        }

