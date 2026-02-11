from __future__ import annotations

from dataclasses import dataclass

from typing import Dict, Tuple

import math

import numpy as np



@dataclass(frozen=True)

class DemandParams:

    base_pax: float

    price_elasticity: float

    shock_sigma: float



def expected_pax(distance_km: float, price_usd: float, params: DemandParams) -> float:



    dist_factor = 1.0 + 0.15 * math.log1p(max(0.0, distance_km) / 500.0)

    price_factor = (max(1e-6, price_usd) / 200.0) ** params.price_elasticity

    return max(0.0, params.base_pax * dist_factor * price_factor)



def sample_pax(distance_km: float, price_usd: float, params: DemandParams, rng: np.random.Generator) -> int:

    mu = expected_pax(distance_km, price_usd, params)



    shock = float(rng.lognormal(mean=0.0, sigma=max(0.0, params.shock_sigma)))

    lam = max(0.0, mu * shock)

    return int(rng.poisson(lam=lam))



def default_demand_params() -> DemandParams:

    return DemandParams(base_pax=120.0, price_elasticity=-0.9, shock_sigma=0.25)

