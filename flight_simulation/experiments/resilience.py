

"""
Resilience experiment: connectivity + detour under random airport closures.

Example:
    python experiments/resilience.py --orig JFK --dest LHR --trials 200 --closure_prob 0.01 --seed 0
"""

from __future__ import annotations



import sys

from pathlib import Path







_ROOT = Path(__file__).resolve().parents[1]

if str(_ROOT) not in sys.path:

    sys.path.insert(0, str(_ROOT))



import argparse

import random

from typing import List, Set, Tuple



from src.data_loader import FlightDataLoader

from src.route_finder import RouteFinder





def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--orig", required=True, help="Origin airport IATA (e.g., JFK)")

    ap.add_argument("--dest", required=True, help="Destination airport IATA (e.g., LHR)")

    ap.add_argument("--trials", type=int, default=200, help="Number of Monte Carlo trials")

    ap.add_argument("--closure_prob", type=float, default=0.01, help="Probability an airport is closed in a trial")

    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--stop_penalty_km", type=float, default=250.0, help="Per-leg penalty for routing (km equivalent)")

    args = ap.parse_args()



    loader = FlightDataLoader()

    loader.load_airport_data()

    net = loader.build_flight_network()

    rf = RouteFinder(net.graph, net.edge_distance_km)



    orig = args.orig.strip().upper()

    dest = args.dest.strip().upper()



    base = rf.dijkstra(orig, dest, per_leg_penalty_km=args.stop_penalty_km)

    if base is None:

        raise SystemExit(f"No baseline route found from {orig} to {dest}.")



    base_dist = rf.path_distance_km(base.path)



    rng = random.Random(args.seed)

    nodes = list(net.graph.keys())

    nodes_set = set(nodes)



    connected = 0

    detour_ratios: List[float] = []



    for _ in range(args.trials):

        closed: Set[str] = set()

        for n in nodes:

            if n in (orig, dest):

                continue

            if rng.random() < args.closure_prob:

                closed.add(n)





        if orig in closed or dest in closed:

            continue



        pr = rf.dijkstra(orig, dest, per_leg_penalty_km=args.stop_penalty_km, banned_nodes=closed)

        if pr is None:

            continue



        connected += 1

        d = rf.path_distance_km(pr.path)

        if base_dist > 0:

            detour_ratios.append(d / base_dist)



    connectivity = connected / max(1, args.trials)

    if detour_ratios:

        avg_detour = sum(detour_ratios) / len(detour_ratios)

        p90_detour = sorted(detour_ratios)[int(0.9 * (len(detour_ratios) - 1))]

    else:

        avg_detour = float("nan")

        p90_detour = float("nan")



    print("=== Resilience experiment ===")

    print(f"OD pair: {orig} -> {dest}")

    print(f"Trials: {args.trials}")

    print(f"Closure probability: {args.closure_prob}")

    print(f"Routing stop penalty (km): {args.stop_penalty_km}")

    print(f"Baseline distance (km): {base_dist:,.0f}")

    print(f"Connectivity: {connectivity:.3f}")

    print(f"Avg detour ratio (given connected): {avg_detour:.3f}")

    print(f"p90 detour ratio (given connected): {p90_detour:.3f}")





if __name__ == "__main__":

    main()

