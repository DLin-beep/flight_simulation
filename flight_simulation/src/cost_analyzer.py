from __future__ import annotations



import math

import random

import statistics

from dataclasses import dataclass

from typing import Dict, List, Optional, Sequence, Tuple, Mapping



from .uncertainty import UncertaintyModel





def _quantile(xs: Sequence[float], q: float) -> float:

    ys = sorted(float(x) for x in xs)

    if not ys:

        return float("nan")

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





def _cvar(xs: Sequence[float], p: float) -> float:

    ys = sorted(float(x) for x in xs)

    if not ys:

        return float("nan")

    if not (0 < p < 1):

        raise ValueError("p must be in (0,1)")

    k = int(math.floor((len(ys) - 1) * p))

    tail = ys[k:]

    return float(sum(tail) / len(tail))





@dataclass(frozen=True)

class RouteSimulationStats:

    mean: float

    median: float

    p10: float

    p90: float

    cvar95: float

    stdev: float





@dataclass(frozen=True)

class CostSummary:

    mean_usd: float

    p90_usd: float

    cvar_usd: float

    mean_min: float

    p90_min: float

    mean_co2_kg: float

    p90_co2_kg: float

    on_time_prob: float





class FlightCostAnalyzer:

    """Cost + time + CO₂ model for a route.

    This is intentionally **transparent** (so you can explain it in interviews)
    while still including modern ingredients:

      - Monte Carlo uncertainty
      - tail-risk metrics (CVaR)
      - airport-specific heavy-tail delays (pluggable)
      - emissions estimate
    """



    FLIGHT_SPEED_KMH = 900.0

    BASE_FUEL_BURN_L_PER_KM = 4.5



    def __init__(

        self,

        *,

        edge_distance_km: Mapping[Tuple[str, str], float] | None = None,

        stop_penalty_usd: float = 0.0,

        uncertainty: UncertaintyModel | None = None,

        seed: Optional[int] = None,

    ):

        self._rng = random.Random(seed)

        self.edge_distance_km = dict(edge_distance_km) if edge_distance_km is not None else {}

        self.stop_penalty_usd = float(stop_penalty_usd)

        self.uncertainty = uncertainty or UncertaintyModel()





        self.params = {

            "domestic": {

                "fuel_price": (0.90, 0.08),

                "op_cost": (0.22, 0.04),

                "ac_cost": (0.14, 0.03),

                "leg_fuel_overhead_l": 280.0,

                "leg_overhead_minutes": 18.0,

                "hourly_delay_cost": 3500.0,

                "airport_fee": 450.0,

                "turnaround_fee": 700.0,

            },

            "international": {

                "fuel_price": (0.95, 0.09),

                "op_cost": (0.28, 0.05),

                "ac_cost": (0.17, 0.03),

                "leg_fuel_overhead_l": 420.0,

                "leg_overhead_minutes": 22.0,

                "hourly_delay_cost": 5200.0,

                "airport_fee": 900.0,

                "turnaround_fee": 1200.0,

            },

        }



    def route_type(self, distance_km: float) -> str:

        return "domestic" if distance_km < 3500 else "international"



    def path_distance_km(self, path: Sequence[str]) -> float:

        dist = 0.0

        for u, v in zip(path, path[1:]):

            w = self.edge_distance_km.get((u, v))

            if w is None:

                return float("inf")

            dist += float(w)

        return dist



    def summarize_path(self, path: Sequence[str], *, samples: int = 2000, seed: Optional[int] = None, delay_sampler: Optional[callable] = None) -> CostSummary:

        dist_km = self.path_distance_km(path)

        if not math.isfinite(dist_km):

            raise ValueError("Path contains unknown edges; cannot compute distance.")

        fuel_liters = dist_km * self.BASE_FUEL_BURN_L_PER_KM

        sim = self.simulate_route(dist_km, path, fuel_liters, n_samples=int(samples), seed=seed, delay_sampler=delay_sampler)

        cs = sim["samples"]["total_cost"]

        ts = sim["samples"]["total_time_min"]

        es = sim["samples"]["total_co2_kg"]

        ds = sim["samples"]["total_delay_min"]





        on_time_prob = float(sum(1 for d in ds if d <= 15.0) / len(ds)) if ds else float("nan")

        return CostSummary(

            mean_usd=float(sum(cs) / len(cs)),

            p90_usd=float(_quantile(cs, 0.90)),

            cvar_usd=float(_cvar(cs, 0.95)),

            mean_min=float(sum(ts) / len(ts)),

            p90_min=float(_quantile(ts, 0.90)),

            mean_co2_kg=float(sum(es) / len(es)),

            p90_co2_kg=float(_quantile(es, 0.90)),

            on_time_prob=on_time_prob,

        )



    def simulate_route(

        self,

        distance_km: float,

        route: Sequence[str],

        fuel_consumption_liters: float,

        *,

        n_samples: int = 2000,

        seed: Optional[int] = None,

        delay_sampler: Optional[callable] = None,

    ) -> Dict:

        if n_samples <= 0:

            raise ValueError("n_samples must be positive.")



        rng = random.Random(seed) if seed is not None else self._rng

        route_type = self.route_type(distance_km)

        p = self.params[route_type]

        legs = max(1, len(route) - 1)

        intermediate_stops = max(0, len(route) - 2)



        efficiency_multiplier = 0.90 if intermediate_stops == 0 else (1.00 + 0.07 * intermediate_stops)



        mu_fuel, sd_fuel = p["fuel_price"]



        def sample_seasonal() -> float:

            return self.uncertainty.seasonal_multiplier(rng)



        def sample_fuel_price() -> float:

            z = rng.gauss(0.0, sd_fuel / max(1e-6, mu_fuel))

            baseline = max(0.55, min(2.20, mu_fuel * math.exp(z)))

            return float(baseline * self.uncertainty.fuel_price_multiplier(rng))



        mu_op, sd_op = p["op_cost"]

        def sample_op_cost_per_km() -> float:

            return max(0.05, rng.gauss(mu_op, sd_op))



        mu_ac, sd_ac = p["ac_cost"]

        def sample_ac_cost_per_km() -> float:

            return max(0.03, rng.gauss(mu_ac, sd_ac))



        def sample_delay_minutes_total() -> float:

            if delay_sampler is not None:

                try:

                    v = delay_sampler(rng, route, distance_km)

                    return float(v)

                except Exception:



                    pass

            total = 0.0

            for a in route[:-1]:

                total += self.uncertainty.delay_minutes(rng, str(a))

                if rng.random() < 0.025:

                    total += rng.uniform(60.0, 220.0)

            total += 0.25 * self.uncertainty.delay_minutes(rng, str(route[-1]))

            return float(total)



        fuel_cost_s: List[float] = []

        operating_cost_s: List[float] = []

        aircraft_cost_s: List[float] = []

        airport_fees_s: List[float] = []

        total_cost_s: List[float] = []

        total_time_min_s: List[float] = []

        total_co2_kg_s: List[float] = []

        total_delay_min_s: List[float] = []

        seasonal_s: List[float] = []



        cruise_time_hours = distance_km / self.FLIGHT_SPEED_KMH

        overhead_hours = (p["leg_overhead_minutes"] * legs) / 60.0

        base_time_hours = cruise_time_hours + overhead_hours



        fuel_liters = fuel_consumption_liters + p["leg_fuel_overhead_l"] * legs

        airport_fees_base = p["airport_fee"] * (legs + 1) + p["turnaround_fee"] * intermediate_stops



        for _ in range(n_samples):

            seasonal = sample_seasonal()

            seasonal_s.append(seasonal)



            fuel_price = sample_fuel_price()

            op_per_km = sample_op_cost_per_km()

            ac_per_km = sample_ac_cost_per_km()



            delay_minutes = sample_delay_minutes_total()

            delay_hours = delay_minutes / 60.0

            delay_cost = delay_hours * p["hourly_delay_cost"]



            fuel_liters_eff = fuel_liters * efficiency_multiplier

            fuel_cost = fuel_liters_eff * fuel_price

            operating_cost = distance_km * op_per_km + delay_cost

            aircraft_cost = distance_km * ac_per_km

            airport_fees = airport_fees_base * rng.triangular(0.90, 1.00, 1.18)



            stop_penalty = self.stop_penalty_usd * intermediate_stops

            total_cost = (fuel_cost + operating_cost + aircraft_cost + airport_fees + stop_penalty) * seasonal

            total_time_min = (base_time_hours + delay_hours) * 60.0



            fuel_kg = fuel_liters_eff * 0.8

            co2_kg = 3.16 * fuel_kg



            fuel_cost_s.append(fuel_cost)

            operating_cost_s.append(operating_cost)

            aircraft_cost_s.append(aircraft_cost)

            airport_fees_s.append(airport_fees)

            total_cost_s.append(total_cost)

            total_time_min_s.append(total_time_min)

            total_co2_kg_s.append(co2_kg)

            total_delay_min_s.append(delay_minutes)



        def stats(xs: List[float]) -> RouteSimulationStats:

            return RouteSimulationStats(

                mean=float(statistics.fmean(xs)),

                median=float(_quantile(xs, 0.50)),

                p10=float(_quantile(xs, 0.10)),

                p90=float(_quantile(xs, 0.90)),

                cvar95=float(_cvar(xs, 0.95)),

                stdev=float(statistics.pstdev(xs)),

            )



        return {

            "route_type": route_type,

            "efficiency_multiplier": float(efficiency_multiplier),

            "seasonal_multiplier_mean": float(statistics.fmean(seasonal_s)),

            "n_samples": int(n_samples),

            "seed": seed,

            "samples": {

                "fuel_cost": fuel_cost_s,

                "operating_cost": operating_cost_s,

                "aircraft_cost": aircraft_cost_s,

                "airport_fees": airport_fees_s,

                "total_cost": total_cost_s,

                "total_time_min": total_time_min_s,

                "total_co2_kg": total_co2_kg_s,

                "total_delay_min": total_delay_min_s,

            },

            "total_cost_stats": stats(total_cost_s),

            "total_time_stats": stats(total_time_min_s),

        }

