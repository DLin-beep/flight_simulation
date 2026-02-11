from __future__ import annotations



import math

import pickle

import urllib.request

from collections import defaultdict

from dataclasses import dataclass

from pathlib import Path

from typing import Dict, Iterable, List, Tuple, Optional



import pandas as pd





OPENFLIGHTS_AIRPORTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"

OPENFLIGHTS_ROUTES_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"





def _haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:

    """Great-circle distance between two points (km)."""

    r = 6371.0

    phi1 = math.radians(a_lat)

    phi2 = math.radians(b_lat)

    dphi = math.radians(b_lat - a_lat)

    dlmb = math.radians(b_lon - a_lon)

    s = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2.0) ** 2

    return 2.0 * r * math.asin(min(1.0, math.sqrt(s)))





def _download_if_missing(url: str, path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0:

        return

    try:

        urllib.request.urlretrieve(url, path.as_posix())

    except Exception as e:

        raise RuntimeError(

            f"Failed to download {url}. If you're offline, place the file at {path} and re-run."

        ) from e





@dataclass(frozen=True)

class FlightNetwork:

    """Graph + metadata used by the route finder."""

    graph: Dict[str, List[Tuple[str, float]]]

    edge_distance_km: Dict[Tuple[str, str], float]

    airport_locations: Dict[str, Tuple[float, float]]





class FlightDataLoader:

    """
    Loads airport + route data (OpenFlights), with simple on-disk caching.

    Data are cached in `.cache/openflights/` so the GUI starts faster after the first run.
    """



    def __init__(self, cache_dir: Optional[Path] = None, data_dir: Optional[Path] = None, allow_download: bool = True):

        self.cache_dir = cache_dir or (Path(__file__).resolve().parents[1] / ".cache" / "openflights")

        self.data_dir = data_dir or (Path(__file__).resolve().parents[1] / "data")

        self.allow_download = allow_download

        self.airports_df: Optional[pd.DataFrame] = None

        self.airport_locations: Dict[str, Tuple[float, float]] = {}

        self._network: Optional[FlightNetwork] = None



    def load_airport_data(self) -> pd.DataFrame:

        """Load airports into a DataFrame and populate airport_locations.

        Prefers bundled CSVs in `data/` (offline-friendly). Falls back to OpenFlights
        download + cache when allowed.
        """



        sample_path = (self.data_dir / "sample_airports.csv")

        if sample_path.exists():

            df = pd.read_csv(sample_path)

            df["iata"] = df["iata"].astype(str).str.strip().str.upper()

            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")

            df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

            df.dropna(subset=["iata", "lat", "lon"], inplace=True)



            self.airports_df = df

            self.airport_locations = {

                str(r["iata"]).upper(): (float(r["lat"]), float(r["lon"]))

                for _, r in df.iterrows()

            }

            return df



        airports_path = self.cache_dir / "airports.dat"

        if not self.allow_download:

            raise RuntimeError("Airport data not found locally and downloads are disabled. Use --offline with bundled CSVs.")

        _download_if_missing(OPENFLIGHTS_AIRPORTS_URL, airports_path)



        cols = [

            "airport_id", "name", "city", "country", "iata", "icao", "latitude", "longitude",

            "altitude", "timezone", "dst", "tz_timezone", "type", "source"

        ]

        df = pd.read_csv(airports_path, header=None, names=cols, index_col=False)



        df = df[(df["iata"] != "\\N") & df["iata"].notna() & df["latitude"].notna() & df["longitude"].notna()].copy()

        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")

        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

        df.dropna(subset=["latitude", "longitude"], inplace=True)



        df["iata"] = df["iata"].astype(str).str.strip().str.upper()

        df["city"] = df["city"].astype(str)

        df["country"] = df["country"].astype(str)



        self.airports_df = df

        self.airport_locations = {r["iata"]: (float(r["latitude"]), float(r["longitude"])) for _, r in df.iterrows()}

        return df



    def build_flight_network(self, force_rebuild: bool = False) -> FlightNetwork:

        if self.airports_df is None:

            raise RuntimeError("Call load_airport_data() before build_flight_network().")



        graph_cache = self.cache_dir / "flight_network.pkl"

        if (not force_rebuild) and graph_cache.exists():

            try:

                with graph_cache.open("rb") as f:

                    net = pickle.load(f)



                if isinstance(net, FlightNetwork) and net.graph and net.airport_locations:

                    self._network = net

                    return net

            except Exception:

                pass





        sample_routes = (self.data_dir / "sample_routes.csv")

        if sample_routes.exists():

            routes_df = pd.read_csv(sample_routes)

        else:

            routes_path = self.cache_dir / "routes.dat"

            if not self.allow_download:

                raise RuntimeError("Route data not found locally and downloads are disabled. Use --offline with bundled CSVs.")

            _download_if_missing(OPENFLIGHTS_ROUTES_URL, routes_path)







        if "src_iata" not in routes_df.columns and {"src", "dst"}.issubset(set(routes_df.columns)):

            routes_df = routes_df.rename(columns={"src": "src_iata", "dst": "dst_iata"})



        valid_iata = set(self.airports_df["iata"])



        edges: Dict[Tuple[str, str], float] = {}



        for _, row in routes_df.iterrows():

            src = str(row["src_iata"]).strip().upper()

            dst = str(row["dst_iata"]).strip().upper()

            if src not in valid_iata or dst not in valid_iata:

                continue

            if src not in self.airport_locations or dst not in self.airport_locations:

                continue



            (slat, slon) = self.airport_locations[src]

            (dlat, dlon) = self.airport_locations[dst]

            dist = _haversine_km(slat, slon, dlat, dlon)



            key = (src, dst)

            prev = edges.get(key)

            if prev is None or dist < prev:

                edges[key] = dist



        adj: Dict[str, List[Tuple[str, float]]] = {iata: [] for iata in valid_iata}

        for (u, v), d in edges.items():

            adj[u].append((v, d))



        net = FlightNetwork(graph=adj, edge_distance_km=edges, airport_locations=self.airport_locations)



        self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:

            with graph_cache.open("wb") as f:

                pickle.dump(net, f)

        except Exception:

            pass



        self._network = net

        return net





    def load_network(self, *, force_rebuild: bool = False) -> FlightNetwork:

        """Convenience: load airports (if needed) and build the FlightNetwork."""

        if self.airports_df is None:

            self.load_airport_data()

        return self.build_flight_network(force_rebuild=force_rebuild)



    def get_city_suggestions(self) -> List[str]:

        if self.airports_df is None:

            raise RuntimeError("Call load_airport_data() first.")

        cities = self.airports_df["city"].str.lower().unique()

        return sorted([city.title() for city in cities if pd.notna(city)])



    def find_airports_by_city(self, city_name: str) -> List[str]:

        if self.airports_df is None:

            raise RuntimeError("Call load_airport_data() first.")

        city_name = city_name.strip().lower()

        return (

            self.airports_df[self.airports_df["city"].str.lower() == city_name]["iata"]

            .astype(str).str.upper().tolist()

        )



    def airport_info_map(self) -> Dict[str, Tuple[str, str]]:

        """iata -> (city, country)"""

        if self.airports_df is None:

            raise RuntimeError("Call load_airport_data() first.")

        return {r["iata"]: (r["city"], r["country"]) for _, r in self.airports_df.iterrows()}

