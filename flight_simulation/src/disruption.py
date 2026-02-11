from __future__ import annotations

from dataclasses import dataclass

from typing import Dict, Iterable, Tuple

import numpy as np



@dataclass(frozen=True)

class DisruptionParams:

    airport_closure_prob: float

    atc_delay_mean_min: float

    atc_delay_sigma_min: float

    mx_event_prob: float

    mx_delay_mean_min: float

    mx_delay_sigma_min: float



def default_disruption_params() -> DisruptionParams:

    return DisruptionParams(

        airport_closure_prob=0.005,

        atc_delay_mean_min=12.0,

        atc_delay_sigma_min=10.0,

        mx_event_prob=0.02,

        mx_delay_mean_min=35.0,

        mx_delay_sigma_min=25.0,

    )



def sample_airport_closures(airports: Iterable[str], params: DisruptionParams, rng: np.random.Generator) -> Dict[str, bool]:

    p = max(0.0, min(1.0, params.airport_closure_prob))

    return {a: bool(rng.random() < p) for a in airports}



def sample_delay_minutes(params: DisruptionParams, rng: np.random.Generator) -> float:



    base = float(max(0.0, rng.normal(loc=params.atc_delay_mean_min, scale=max(1e-6, params.atc_delay_sigma_min))))

    if rng.random() < max(0.0, min(1.0, params.mx_event_prob)):

        base += float(max(0.0, rng.normal(loc=params.mx_delay_mean_min, scale=max(1e-6, params.mx_delay_sigma_min))))

    return base

