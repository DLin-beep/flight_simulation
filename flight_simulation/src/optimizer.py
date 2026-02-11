from __future__ import annotations

from dataclasses import dataclass

from typing import Dict, List, Optional, Tuple

import numpy as np



from .fleet import Fleet, AircraftType



@dataclass(frozen=True)

class LegAssignment:

    leg: Tuple[str, str]

    aircraft_type: str

    pax_cap: int



def assign_aircraft_greedy(legs: List[Tuple[str, str]], demand_forecast: Dict[Tuple[str, str], int], fleet: Fleet) -> List[LegAssignment]:

    """
    Simple airline-ish baseline:
    - Choose smallest aircraft that satisfies forecast, subject to availability.
    - If none available, choose largest available.
    This is not an ops optimizer; it's a clean, explainable baseline.
    """



    types_sorted = sorted(fleet.types.values(), key=lambda t: t.seats)

    remaining = dict(fleet.counts)

    out: List[LegAssignment] = []



    for leg in legs:

        need = int(max(0, demand_forecast.get(leg, 0)))

        chosen: Optional[AircraftType] = None





        for t in types_sorted:

            if remaining.get(t.code, 0) > 0 and t.seats >= need:

                chosen = t

                break





        if chosen is None:

            for t in reversed(types_sorted):

                if remaining.get(t.code, 0) > 0:

                    chosen = t

                    break





        if chosen is None:

            chosen = types_sorted[0]



        remaining[chosen.code] = max(0, remaining.get(chosen.code, 0) - 1)

        out.append(LegAssignment(leg=leg, aircraft_type=chosen.code, pax_cap=chosen.seats))



    return out

