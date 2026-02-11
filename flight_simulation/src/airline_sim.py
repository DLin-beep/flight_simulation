from __future__ import annotations

from dataclasses import dataclass

from typing import Dict, List, Tuple, Optional

import numpy as np



from .fleet import AircraftType

from .demand import DemandParams, sample_pax

from .revenue import RevenueModel, ticket_price_usd, revenue_usd

from .metrics import estimate_emissions

from .disruption import DisruptionParams, sample_delay_minutes



@dataclass(frozen=True)

class LegKpis:

    src: str

    dst: str

    distance_km: float

    aircraft: str

    pax: int

    capacity: int

    load_factor: float

    block_time_min: float

    delay_min: float

    fuel_kg: float

    co2_kg: float

    revenue_usd: float

    cost_usd: float

    profit_usd: float



@dataclass(frozen=True)

class SimulationParams:

    fuel_price_usd_per_kg: float

    crew_usd_per_block_hour: float

    mx_usd_per_leg: float

    airport_fee_usd: float

    delay_cost_usd_per_min: float



def default_sim_params() -> SimulationParams:

    return SimulationParams(

        fuel_price_usd_per_kg=0.95,

        crew_usd_per_block_hour=620.0,

        mx_usd_per_leg=850.0,

        airport_fee_usd=420.0,

        delay_cost_usd_per_min=18.0,

    )



def simulate_leg(

    src: str,

    dst: str,

    distance_km: float,

    aircraft: AircraftType,

    sim: SimulationParams,

    demand: DemandParams,

    revenue_model: RevenueModel,

    disruption: DisruptionParams,

    rng: np.random.Generator,

) -> LegKpis:



    price = ticket_price_usd(distance_km, revenue_model)

    pax = sample_pax(distance_km, price, demand, rng)

    pax = min(pax, aircraft.seats)

    lf = (pax / aircraft.seats) if aircraft.seats else 0.0





    cruise_hr = distance_km / max(1e-6, aircraft.block_speed_kmh)

    block_min = 18.0 + 60.0 * cruise_hr

    delay_min = sample_delay_minutes(disruption, rng)





    em = estimate_emissions(distance_km, aircraft.burn_kg_per_km)

    fuel_cost = em.fuel_kg * sim.fuel_price_usd_per_kg





    crew_cost = (block_min / 60.0) * sim.crew_usd_per_block_hour

    delay_cost = delay_min * sim.delay_cost_usd_per_min

    cost = fuel_cost + crew_cost + delay_cost + sim.mx_usd_per_leg + sim.airport_fee_usd



    rev = revenue_usd(pax, price, revenue_model)

    profit = rev - cost



    return LegKpis(

        src=src, dst=dst, distance_km=distance_km, aircraft=aircraft.code,

        pax=pax, capacity=aircraft.seats, load_factor=lf,

        block_time_min=block_min, delay_min=delay_min,

        fuel_kg=em.fuel_kg, co2_kg=em.co2_kg,

        revenue_usd=rev, cost_usd=cost, profit_usd=profit

    )

