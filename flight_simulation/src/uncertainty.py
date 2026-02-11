from __future__ import annotations



import math

import random

from dataclasses import dataclass

from typing import Dict, Optional, Sequence, Tuple





@dataclass(frozen=True)

class MixtureZeroLogNormal:

    """Compact heavy-tail model for nonnegative delays.

    Delay is 0 with probability p0, else LogNormal(mu, sigma) minutes.
    """



    p0: float

    mu: float

    sigma: float



    def sample(self, rng: random.Random) -> float:

        if rng.random() < self.p0:

            return 0.0

        z = rng.gauss(self.mu, self.sigma)

        return float(max(0.0, math.exp(z)))



    @staticmethod

    def fit(samples_min: Sequence[float]) -> "MixtureZeroLogNormal":

        xs = [float(x) for x in samples_min if x is not None and x >= 0]

        if not xs:

            return MixtureZeroLogNormal(p0=1.0, mu=math.log(1.0), sigma=0.1)



        zeros = sum(1 for x in xs if x <= 1e-12)

        p0 = zeros / len(xs)

        pos = [x for x in xs if x > 1e-12]

        if len(pos) < 2:

            return MixtureZeroLogNormal(p0=min(0.99, p0), mu=math.log(max(1.0, (pos[0] if pos else 1.0))), sigma=0.25)



        logs = [math.log(x) for x in pos]

        m = sum(logs) / len(logs)

        v = sum((t - m) ** 2 for t in logs) / max(1, len(logs) - 1)

        sigma = float(max(1e-3, math.sqrt(v)))

        mu = float(m)

        p0 = float(min(0.995, max(0.0, p0)))

        return MixtureZeroLogNormal(p0=p0, mu=mu, sigma=sigma)





@dataclass(frozen=True)

class UncertaintyParams:

    fuel_price_sigma: float = 0.10

    demand_sigma: float = 0.07

    seasonal_low: float = 0.92

    seasonal_mode: float = 1.02

    seasonal_high: float = 1.28





class UncertaintyModel:

    """Sampling utilities used across the project.

    Designed so you can later swap in a learned model without changing the
    optimization/simulation code.
    """



    def __init__(

        self,

        delay_model_by_airport: Optional[Dict[str, MixtureZeroLogNormal]] = None,

        params: Optional[UncertaintyParams] = None,

    ):

        self.delay_model_by_airport = delay_model_by_airport or {}

        self.params = params or UncertaintyParams()



    def delay_minutes(self, rng: random.Random, airport_iata: str) -> float:

        m = self.delay_model_by_airport.get(airport_iata)

        if m is None:

            m = MixtureZeroLogNormal(p0=0.55, mu=math.log(12.0), sigma=0.65)

        return m.sample(rng)



    def seasonal_multiplier(self, rng: random.Random) -> float:

        p = self.params

        return float(rng.triangular(p.seasonal_low, p.seasonal_mode, p.seasonal_high))



    def fuel_price_multiplier(self, rng: random.Random) -> float:

        sigma = float(max(1e-6, self.params.fuel_price_sigma))

        return float(math.exp(rng.gauss(0.0, sigma)))



    def demand_multiplier(self, rng: random.Random) -> float:

        sigma = float(max(1e-6, self.params.demand_sigma))

        x = rng.gauss(0.0, sigma)

        return float(max(0.6, min(1.6, 1.0 + x)))





def synthesize_delay_history(

    airports: Sequence[str],

    *,

    n_per_airport: int = 600,

    seed: int = 0,

    shift: float = 0.0,

) -> Dict[str, Tuple[MixtureZeroLogNormal, list[float]]]:

    """Synthetic "historical" delay samples per airport (for demo + calibration)."""

    rng = random.Random(seed)

    out: Dict[str, Tuple[MixtureZeroLogNormal, list[float]]] = {}

    for a in airports:

        base_p0 = 0.45 + 0.25 * rng.random()

        base_mu = math.log(10.0 + 18.0 * rng.random()) + shift

        base_sigma = 0.45 + 0.55 * rng.random()

        true_m = MixtureZeroLogNormal(p0=base_p0, mu=base_mu, sigma=base_sigma)

        xs = [true_m.sample(rng) for _ in range(n_per_airport)]

        fitted = MixtureZeroLogNormal.fit(xs)

        out[a] = (fitted, xs)

    return out





import pandas as pd

from datetime import datetime



def _find_col(df: "pd.DataFrame", candidates: list[str]) -> str | None:

    cols = {c.lower(): c for c in df.columns}

    for cand in candidates:

        if cand.lower() in cols:

            return cols[cand.lower()]

    return None



def fit_delay_models_from_dataframe(

    df: "pd.DataFrame",

    *,

    airport_col_candidates: tuple[list[str], list[str]] = (["origin", "ORIGIN", "Origin", "origin_iata"], ["dest", "DEST", "Dest", "destination", "dest_iata"]),

    delay_col_candidates: list[str] = ["arr_delay", "ARR_DELAY", "ArrDelay", "arrival_delay", "dep_delay", "DEP_DELAY", "DepDelay", "departure_delay"],

    date_col_candidates: list[str] = ["flight_date", "FL_DATE", "date", "Date", "FLIGHT_DATE"],

    train_frac: float = 0.7,

    min_samples: int = 80,

    seed: int = 0,

) -> tuple[dict[str, MixtureZeroLogNormal], dict[str, list[float]], dict[str, list[float]]]:

    """Fit per-airport heavy-tail delay models from a user-provided CSV.

    Returns:
      models: airport -> MixtureZeroLogNormal
      train_samples: airport -> list[delay_minutes]
      test_samples: airport -> list[delay_minutes]

    Supported schemas (examples):
      - ORIGIN, DEST, ARR_DELAY, FL_DATE  (BTS style)
      - origin, dest, arr_delay, flight_date
      - origin_iata, dest_iata, arrival_delay, date
    """

    import random as _random



    df = df.copy()

    ocol = _find_col(df, airport_col_candidates[0])

    dcol = _find_col(df, airport_col_candidates[1])

    delcol = _find_col(df, delay_col_candidates)

    if ocol is None or dcol is None or delcol is None:

        missing = []

        if ocol is None: missing.append("origin")

        if dcol is None: missing.append("dest")

        if delcol is None: missing.append("delay")

        raise ValueError(f"CSV missing required columns: {', '.join(missing)}")





    df[ocol] = df[ocol].astype(str).str.strip().str.upper()

    df[dcol] = df[dcol].astype(str).str.strip().str.upper()

    df[delcol] = pd.to_numeric(df[delcol], errors="coerce")





    df = df[df[ocol].str.len() == 3]

    df = df[df[dcol].str.len() == 3]

    df = df[df[delcol].notna()]





    df[delcol] = df[delcol].clip(lower=0.0)



    datecol = _find_col(df, date_col_candidates)

    if datecol is not None:



        df[datecol] = pd.to_datetime(df[datecol], errors="coerce")

        df = df[df[datecol].notna()].sort_values(datecol)

        n_train = int(max(1, min(len(df) - 1, round(train_frac * len(df)))))

        train_df = df.iloc[:n_train]

        test_df = df.iloc[n_train:]

    else:

        rng = _random.Random(seed)

        idx = list(df.index)

        rng.shuffle(idx)

        n_train = int(max(1, min(len(idx) - 1, round(train_frac * len(idx)))))

        train_df = df.loc[idx[:n_train]]

        test_df = df.loc[idx[n_train:]]



    train_by_airport: dict[str, list[float]] = {}

    test_by_airport: dict[str, list[float]] = {}



    def _add(dct: dict[str, list[float]], airport: str, val: float):

        dct.setdefault(airport, []).append(float(val))



    for _, r in train_df.iterrows():

        delay = float(r[delcol])

        _add(train_by_airport, str(r[ocol]), delay)

        _add(train_by_airport, str(r[dcol]), delay)

    for _, r in test_df.iterrows():

        delay = float(r[delcol])

        _add(test_by_airport, str(r[ocol]), delay)

        _add(test_by_airport, str(r[dcol]), delay)



    models: dict[str, MixtureZeroLogNormal] = {}

    for ap, xs in train_by_airport.items():

        if len(xs) >= min_samples:

            models[ap] = MixtureZeroLogNormal.fit(xs)



    return models, train_by_airport, test_by_airport

