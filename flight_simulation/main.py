

"""Flight Simulation (airline-oriented) — CLI entrypoint.

Design goals:
- Reproducible routing + uncertainty simulation (no GUI required).
- Airline-style KPIs: cost, time, CO2, load factor, revenue, profit.
- A clean baseline that can be extended with real ops data.

Quick start:
    pip install -r requirements.txt
    python main.py route --orig JFK --dest LHR --objective cvar --k 8 --samples 4000
    python main.py simulate --orig JFK --dest LHR --aircraft B789 --samples 2000
"""



from __future__ import annotations



import argparse

import hashlib

from dataclasses import dataclass

from typing import Dict, List, Sequence, Tuple, Optional



import numpy as np



from src.data_loader import FlightDataLoader

from src.route_finder import RouteFinder, PathResult

from src.cost_analyzer import FlightCostAnalyzer, CostSummary

from src.fleet import default_demo_fleet

from src.demand import default_demand_params

from src.revenue import default_revenue_model

from src.disruption import default_disruption_params

from src.airline_sim import default_sim_params, simulate_leg





def _stable_seed(*parts: str) -> int:

    h = hashlib.md5("||".join(parts).encode("utf-8")).hexdigest()

    return int(h[:8], 16)





def _format_path(path: Sequence[str]) -> str:

    return " -> ".join(path)





def _route_candidates(rf: RouteFinder, orig: str, dest: str, k: int) -> List[PathResult]:



    return rf.k_shortest_paths(orig, dest, k=k)





def cmd_route(args: argparse.Namespace) -> int:

    loader = FlightDataLoader(allow_download=not args.offline)

    net = loader.load_network()



    rf = RouteFinder(net.graph)

    candidates = _route_candidates(rf, args.orig, args.dest, args.k)

    if not candidates:

        print(f"No route found from {args.orig} to {args.dest}.")

        return 2



    analyzer = FlightCostAnalyzer(

        edge_distance_km=net.edge_distance_km,

        stop_penalty_usd=float(args.stop_penalty),

        seed=_stable_seed(args.orig, args.dest, args.objective, str(args.samples)),

    )



    scored: List[Tuple[float, PathResult, CostSummary]] = []

    for pr in candidates:

        summary = analyzer.summarize_path(pr.path, samples=int(args.samples))

        score = summary.cvar_usd if args.objective == "cvar" else summary.mean_usd

        scored.append((score, pr, summary))



    scored.sort(key=lambda t: t[0])



    print(f"Objective: {args.objective.upper()}  (k={args.k}, samples={args.samples})")

    for rank, (score, pr, s) in enumerate(scored[: min(len(scored), 10)], start=1):

        print("\n" + "=" * 70)

        print(f"#{rank}  score={score:,.0f} USD")

        print(f"Path: {_format_path(pr.path)}")

        print(f"Distance: {pr.distance_km:,.0f} km | Stops: {max(0, len(pr.path)-2)}")

        print(f"Cost: mean={s.mean_usd:,.0f}  p90={s.p90_usd:,.0f}  CVaR95={s.cvar_usd:,.0f}")

        print(f"Time: mean={s.mean_min:,.0f} min  p90={s.p90_min:,.0f} min")

    return 0





def cmd_simulate(args: argparse.Namespace) -> int:

    loader = FlightDataLoader(allow_download=not args.offline)

    net = loader.load_network()

    rf = RouteFinder(net.graph)





    best = rf.shortest_path(args.orig, args.dest)

    if best is None:

        print(f"No route found from {args.orig} to {args.dest}.")

        return 2



    fleet = default_demo_fleet()

    if args.aircraft not in fleet.types:

        print(f"Unknown aircraft type '{args.aircraft}'. Available: {', '.join(sorted(fleet.types))}")

        return 2



    demand = default_demand_params()

    revenue_model = default_revenue_model()

    disruption = default_disruption_params()

    sim_params = default_sim_params()



    rng = np.random.default_rng(_stable_seed(args.orig, args.dest, args.aircraft, str(args.samples)))

    aircraft = fleet.get(args.aircraft)





    profits: List[float] = []

    costs: List[float] = []

    revs: List[float] = []

    co2: List[float] = []

    delays: List[float] = []



    for _ in range(int(args.samples)):

        tot_profit = tot_cost = tot_rev = tot_co2 = tot_delay = 0.0

        for u, v in zip(best.path, best.path[1:]):

            d = net.edge_distance_km[(u, v)]

            kpi = simulate_leg(u, v, d, aircraft, sim_params, demand, revenue_model, disruption, rng)

            tot_profit += kpi.profit_usd

            tot_cost += kpi.cost_usd

            tot_rev += kpi.revenue_usd

            tot_co2 += kpi.co2_kg

            tot_delay += kpi.delay_min

        profits.append(tot_profit); costs.append(tot_cost); revs.append(tot_rev); co2.append(tot_co2); delays.append(tot_delay)



    def q(xs: List[float], p: float) -> float:

        xs2 = sorted(xs)

        if not xs2:

            return float("nan")

        i = int(round((len(xs2)-1)*p))

        return xs2[i]



    def cvar_loss(xs: List[float], alpha: float = 0.95) -> float:



        losses = sorted([-x for x in xs])

        if not losses:

            return float("nan")

        k = max(1, int((1-alpha)*len(losses)))

        return sum(losses[:k]) / k



    print("=" * 70)

    print(f"Simulated route (distance-optimal): {_format_path(best.path)}")

    print(f"Aircraft: {aircraft.code} (seats={aircraft.seats}) | samples={args.samples}")

    print("-" * 70)

    print(f"Revenue: mean={np.mean(revs):,.0f} USD  p10={q(revs,0.10):,.0f}  p90={q(revs,0.90):,.0f}")

    print(f"Cost:    mean={np.mean(costs):,.0f} USD  p10={q(costs,0.10):,.0f}  p90={q(costs,0.90):,.0f}")

    print(f"Profit:  mean={np.mean(profits):,.0f} USD  p10={q(profits,0.10):,.0f}  p90={q(profits,0.90):,.0f}")

    print(f"Profit tail risk (CVaR loss 95%): {cvar_loss(profits):,.0f} USD")

    print(f"CO2:     mean={np.mean(co2):,.0f} kg")

    print(f"Delay:   mean={np.mean(delays):,.1f} min")

    return 0





def build_parser() -> argparse.ArgumentParser:

    p = argparse.ArgumentParser(prog="flight-simulation", description="Airline-oriented flight routing & simulation")

    sub = p.add_subparsers(dest="cmd", required=True)



    p_route = sub.add_parser("route", help="Find k candidate routes and score under uncertainty")

    p_route.add_argument("--orig", required=True, help="Origin IATA (e.g., JFK)")

    p_route.add_argument("--dest", required=True, help="Destination IATA (e.g., LHR)")

    p_route.add_argument("--objective", choices=["mean", "cvar"], default="cvar")

    p_route.add_argument("--k", type=int, default=8)

    p_route.add_argument("--samples", type=int, default=4000)

    p_route.add_argument("--stop-penalty", type=float, default=250.0)

    p_route.add_argument("--offline", action="store_true", help="Use bundled sample data only (no downloads)")

    p_route.set_defaults(func=cmd_route)



    p_sim = sub.add_parser("simulate", help="Simulate airline KPIs (revenue/cost/profit/CO2) on a route")

    p_sim.add_argument("--orig", required=True)

    p_sim.add_argument("--dest", required=True)

    p_sim.add_argument("--aircraft", default="B789")

    p_sim.add_argument("--samples", type=int, default=2000)

    p_sim.add_argument("--offline", action="store_true")

    p_sim.set_defaults(func=cmd_simulate)



    return p





def main() -> int:

    args = build_parser().parse_args()

    return int(args.func(args))





if __name__ == "__main__":

    raise SystemExit(main())

