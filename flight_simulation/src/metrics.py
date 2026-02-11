from __future__ import annotations

from dataclasses import dataclass



@dataclass(frozen=True)

class EmissionsEstimate:

    fuel_kg: float

    co2_kg: float



def fuel_kg_from_distance_km(distance_km: float, burn_kg_per_km: float) -> float:

    return max(0.0, distance_km) * max(0.0, burn_kg_per_km)



def co2_kg_from_fuel_kg(fuel_kg: float, co2_factor: float = 3.16) -> float:



    return max(0.0, fuel_kg) * max(0.0, co2_factor)



def estimate_emissions(distance_km: float, burn_kg_per_km: float) -> EmissionsEstimate:

    fuel = fuel_kg_from_distance_km(distance_km, burn_kg_per_km)

    return EmissionsEstimate(fuel_kg=fuel, co2_kg=co2_kg_from_fuel_kg(fuel))

