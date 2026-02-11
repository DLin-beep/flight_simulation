import math

from src.data_loader import FlightDataLoader

from src.route_finder import RouteFinder

from src.cost_analyzer import FlightCostAnalyzer



def test_offline_network_builds():

    loader = FlightDataLoader(allow_download=False)

    net = loader.load_network()

    assert net.graph

    assert net.edge_distance_km

    assert net.airport_locations



def test_shortest_path_exists_offline():

    loader = FlightDataLoader(allow_download=False)

    net = loader.load_network()

    rf = RouteFinder(net.graph)

    pr = rf.shortest_path("JFK", "LHR")

    assert pr is not None

    assert pr.path[0] == "JFK"

    assert pr.path[-1] == "LHR"

    assert pr.distance_km > 0



def test_cost_summary_fields_finite():

    loader = FlightDataLoader(allow_download=False)

    net = loader.load_network()

    rf = RouteFinder(net.graph)

    pr = rf.shortest_path("JFK", "LHR")

    analyzer = FlightCostAnalyzer(edge_distance_km=net.edge_distance_km, stop_penalty_usd=5000.0, seed=123)

    s = analyzer.summarize_path(pr.path, samples=500)

    assert math.isfinite(s.mean_usd)

    assert math.isfinite(s.p90_usd)

    assert math.isfinite(s.cvar_usd)

    assert s.p90_usd >= s.mean_usd * 0.8

