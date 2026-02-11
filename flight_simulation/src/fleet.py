from __future__ import annotations

from dataclasses import dataclass

from typing import Dict, Iterable, List, Optional



@dataclass(frozen=True)

class AircraftType:

    code: str

    seats: int

    burn_kg_per_km: float

    block_speed_kmh: float



@dataclass(frozen=True)

class Fleet:

    types: Dict[str, AircraftType]

    counts: Dict[str, int]



    def total_tails(self) -> int:

        return int(sum(self.counts.values()))



    def get(self, code: str) -> AircraftType:

        return self.types[code]



def default_demo_fleet() -> Fleet:



    types = {

        "A320": AircraftType("A320", seats=180, burn_kg_per_km=2.6, block_speed_kmh=780),

        "B738": AircraftType("B738", seats=160, burn_kg_per_km=2.5, block_speed_kmh=770),

        "B789": AircraftType("B789", seats=290, burn_kg_per_km=5.8, block_speed_kmh=830),

        "A321": AircraftType("A321", seats=200, burn_kg_per_km=2.8, block_speed_kmh=790),

    }

    counts = {"A320": 25, "B738": 30, "B789": 6, "A321": 12}

    return Fleet(types=types, counts=counts)

