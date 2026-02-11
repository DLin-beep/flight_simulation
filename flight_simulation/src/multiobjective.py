from __future__ import annotations



"""Multi-objective route selection utilities.

This module is deliberately small and dependency-free so it can be reused both
from the CLI and the Streamlit UI.

We treat a route as a point in objective space:
  - mean cost (USD)
  - tail risk: CVaR95 cost (USD)
  - mean CO2 (kg)

and provide:
  - Pareto frontier extraction
  - lightweight feasibility filtering (budgets / minimum on-time probability)
  - interpretable weighted selection over the frontier
"""



from dataclasses import dataclass

from typing import Dict, List, Sequence, Tuple





@dataclass(frozen=True)

class RouteMetrics:

    path: Tuple[str, ...]

    distance_km: float

    mean_cost_usd: float

    p90_cost_usd: float

    cvar95_cost_usd: float

    mean_time_min: float

    p90_time_min: float

    mean_co2_kg: float

    p90_co2_kg: float

    on_time_prob: float



    def dominates(self, other: "RouteMetrics") -> bool:

        """True if this route is no-worse on all primary objectives and better on ≥1."""

        a = (self.mean_cost_usd, self.cvar95_cost_usd, self.mean_co2_kg)

        b = (other.mean_cost_usd, other.cvar95_cost_usd, other.mean_co2_kg)

        le = all(x <= y for x, y in zip(a, b))

        lt = any(x < y for x, y in zip(a, b))

        return le and lt





def pareto_frontier(metrics: Sequence[RouteMetrics]) -> List[RouteMetrics]:

    """Compute the Pareto frontier w.r.t. (mean cost, CVaR95 cost, mean CO2)."""

    frontier: List[RouteMetrics] = []

    for m in metrics:

        dominated = False

        new_frontier: List[RouteMetrics] = []

        for f in frontier:

            if f.dominates(m):

                dominated = True

                new_frontier.append(f)

            elif m.dominates(f):

                continue

            else:

                new_frontier.append(f)

        if not dominated:

            new_frontier.append(m)

        frontier = new_frontier

    return frontier





def pick_by_weights(frontier: Sequence[RouteMetrics], *, w_cost: float, w_risk: float, w_co2: float) -> RouteMetrics:

    """Pick a single route from a frontier using normalized weighted objectives."""

    if not frontier:

        raise ValueError("frontier is empty")



    def norm(xs: List[float]) -> List[float]:

        lo = min(xs)

        hi = max(xs)

        if hi - lo < 1e-12:

            return [0.0 for _ in xs]

        return [(x - lo) / (hi - lo) for x in xs]



    costs = [m.mean_cost_usd for m in frontier]

    risks = [m.cvar95_cost_usd for m in frontier]

    co2s = [m.mean_co2_kg for m in frontier]

    nc = norm(costs)

    nr = norm(risks)

    ne = norm(co2s)



    best_i = 0

    best_score = float("inf")

    for i, (a, b, c) in enumerate(zip(nc, nr, ne)):

        s = float(w_cost) * a + float(w_risk) * b + float(w_co2) * c

        if s < best_score:

            best_score = s

            best_i = i

    return frontier[best_i]





def feasible_set(

    metrics: Sequence[RouteMetrics],

    *,

    max_mean_co2_kg: float | None = None,

    max_cvar95_cost_usd: float | None = None,

    min_on_time_prob: float | None = None,

) -> List[RouteMetrics]:

    """Filter a set of routes by simple decision constraints."""

    out: List[RouteMetrics] = []

    for m in metrics:

        if max_mean_co2_kg is not None and m.mean_co2_kg > max_mean_co2_kg:

            continue

        if max_cvar95_cost_usd is not None and m.cvar95_cost_usd > max_cvar95_cost_usd:

            continue

        if min_on_time_prob is not None and m.on_time_prob < min_on_time_prob:

            continue

        out.append(m)

    return out





def explain_weights(preset: str) -> Dict[str, float]:

    """Optional helper for UI/reporting."""

    if preset == "Lowest cost":

        return {"w_cost": 0.70, "w_risk": 0.20, "w_co2": 0.10}

    if preset == "Most reliable":

        return {"w_cost": 0.25, "w_risk": 0.65, "w_co2": 0.10}

    if preset == "Lowest emissions":

        return {"w_cost": 0.25, "w_risk": 0.15, "w_co2": 0.60}

    return {"w_cost": 0.50, "w_risk": 0.30, "w_co2": 0.20}

