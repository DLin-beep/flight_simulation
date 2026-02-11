from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

from typing import Dict, Iterable, List, Tuple, Optional

import pandas as pd



@dataclass(frozen=True)

class Airport:

    iata: str

    name: str

    city: str

    country: str

    lat: float

    lon: float



def load_airports_csv(path: Path) -> Dict[str, Airport]:

    df = pd.read_csv(path)

    required = {"iata", "name", "city", "country", "lat", "lon"}

    missing = required - set(df.columns)

    if missing:

        raise ValueError(f"airports CSV missing columns: {sorted(missing)}")

    out: Dict[str, Airport] = {}

    for _, r in df.iterrows():

        iata = str(r["iata"]).strip().upper()

        if not iata or iata == "NAN":

            continue

        out[iata] = Airport(

            iata=iata,

            name=str(r["name"]),

            city=str(r["city"]),

            country=str(r["country"]),

            lat=float(r["lat"]),

            lon=float(r["lon"]),

        )

    return out



def load_routes_csv(path: Path) -> List[Tuple[str, str, str]]:

    df = pd.read_csv(path)

    required = {"src", "dst", "airline"}

    missing = required - set(df.columns)

    if missing:

        raise ValueError(f"routes CSV missing columns: {sorted(missing)}")

    routes: List[Tuple[str, str, str]] = []

    for _, r in df.iterrows():

        routes.append((str(r["src"]).strip().upper(), str(r["dst"]).strip().upper(), str(r["airline"]).strip().upper()))

    return routes

