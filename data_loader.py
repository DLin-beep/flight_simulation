import pandas as pd
from geopy.distance import geodesic
from collections import defaultdict
import math

class FlightDataLoader:
    
    def __init__(self):
        self.airports_df = None
        self.routes_graph = None
        self.airport_locations = {}
        
    def load_airport_data(self):
        cols = ["airport_id","name","city","country","iata","icao","latitude","longitude","altitude","timezone","dst","tz_timezone","type","source"]
        df = pd.read_csv("https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat", 
                        header=None, names=cols, index_col=False)
        
        df = df[(df["iata"] != "\\N") & df["iata"].notna() & 
                df["latitude"].notna() & df["longitude"].notna()].copy()
        
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df.dropna(subset=["latitude","longitude"], inplace=True)
        
        self.airports_df = df
        self.airport_locations = {r["iata"]: (r["latitude"], r["longitude"]) for _, r in df.iterrows()}
        return df
    
    def build_flight_network(self):
        cols = ["airline","airline_id","src_iata","src_id","dst_iata","dst_id","codeshare","stops","equipment"]
        routes_df = pd.read_csv("https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat", 
                               header=None, names=cols)
        
        valid_iata = set(self.airports_df["iata"])
        graph = defaultdict(dict)
        
        for _, row in routes_df.iterrows():
            src = row["src_iata"]
            dst = row["dst_iata"]
            
            if src in valid_iata and dst in valid_iata:
                distance = geodesic(self.airport_locations[src], self.airport_locations[dst]).kilometers
                if distance < graph[src].get(dst, math.inf):
                    graph[src][dst] = distance
        
        self.routes_graph = {s: list(d.items()) for s, d in graph.items()}
        return self.routes_graph
    
    def get_city_suggestions(self):
        cities = self.airports_df["city"].str.lower().unique()
        return sorted([city.title() for city in cities if pd.notna(city)])
    
    def find_airports_by_city(self, city_name):
        return self.airports_df[self.airports_df["city"].str.lower() == city_name.lower()]["iata"].tolist() 
