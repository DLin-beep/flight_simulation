"""
Flight Route Finder (GUI module)

What’s new in this version:
- Fixed project layout: a real `src/` package now exists (so imports work).
- K-shortest candidate routes (Yen’s algorithm) so you can compare alternatives.
- Monte Carlo uncertainty + risk metrics (p90, CVaR95) for cost/time.
- Risk-aware optimization option: pick routes that minimize tail-risk, not just distance.

Run:
    python gui.py
"""

from __future__ import annotations



import hashlib

import math

import tkinter as tk

from tkinter import ttk



from src.data_loader import FlightDataLoader

from src.route_finder import RouteFinder

from src.cost_analyzer import FlightCostAnalyzer

from src.world_map import WorldMapRenderer





def _stable_seed(*parts: str) -> int:

    h = hashlib.md5("||".join(parts).encode("utf-8")).hexdigest()



    return int(h[:8], 16) % (2**31 - 1)





class FlightRouteApp:

    def __init__(self, root: tk.Tk):

        self.root = root

        self.root.title("Flight Route Finder (Risk-Aware)")



        self.data_loader = FlightDataLoader()

        self.cost_analyzer = FlightCostAnalyzer(seed=42)



        self.df = self.data_loader.load_airport_data()

        net = self.data_loader.build_flight_network()

        self.network = net

        self.graph = net.graph

        self.route_finder = RouteFinder(self.graph, net.edge_distance_km)



        self.city_suggestions = self.data_loader.get_city_suggestions()

        self.info_map = self.data_loader.airport_info_map()



        self.create_widgets()







    def create_widgets(self):

        input_frame = ttk.Frame(self.root, padding=10)

        input_frame.pack(fill=tk.X)



        ttk.Label(input_frame, text="Origin City:").grid(row=0, column=0, padx=5, sticky=tk.E)

        self.origin_entry = ttk.Entry(input_frame, width=28)

        self.origin_entry.grid(row=0, column=1, padx=5)

        self.origin_entry.bind("<KeyRelease>", lambda e: self.show_suggestions(self.origin_entry, self.origin_listbox))



        ttk.Label(input_frame, text="Destination City:").grid(row=0, column=2, padx=5, sticky=tk.E)

        self.dest_entry = ttk.Entry(input_frame, width=28)

        self.dest_entry.grid(row=0, column=3, padx=5)

        self.dest_entry.bind("<KeyRelease>", lambda e: self.show_suggestions(self.dest_entry, self.dest_listbox))



        search_btn = ttk.Button(input_frame, text="Find Route", command=self.search_route)

        search_btn.grid(row=0, column=4, padx=5)





        self.origin_listbox = tk.Listbox(input_frame, height=5)

        self.origin_listbox.grid(row=1, column=1, padx=5, sticky="ew")

        self.origin_listbox.bind("<<ListboxSelect>>", lambda e: self.select_suggestion(self.origin_entry, self.origin_listbox))



        self.dest_listbox = tk.Listbox(input_frame, height=5)

        self.dest_listbox.grid(row=1, column=3, padx=5, sticky="ew")

        self.dest_listbox.bind("<<ListboxSelect>>", lambda e: self.select_suggestion(self.dest_entry, self.dest_listbox))





        options = ttk.LabelFrame(self.root, text="Optimization & Simulation Options", padding=10)

        options.pack(fill=tk.X, padx=10)



        ttk.Label(options, text="Optimize for:").grid(row=0, column=0, padx=5, sticky=tk.E)

        self.objective_var = tk.StringVar(value="Shortest distance")

        self.objective_box = ttk.Combobox(

            options,

            textvariable=self.objective_var,

            state="readonly",

            values=[

                "Shortest distance",

                "Lowest expected cost",

                "Lowest tail-risk cost (CVaR95)",

            ],

            width=30,

        )

        self.objective_box.grid(row=0, column=1, padx=5, sticky=tk.W)



        ttk.Label(options, text="Stop penalty (km):").grid(row=0, column=2, padx=5, sticky=tk.E)

        self.stop_penalty_var = tk.IntVar(value=250)

        ttk.Spinbox(options, from_=0, to=2000, textvariable=self.stop_penalty_var, width=8).grid(row=0, column=3, padx=5, sticky=tk.W)



        ttk.Label(options, text="MC samples:").grid(row=0, column=4, padx=5, sticky=tk.E)

        self.samples_var = tk.IntVar(value=2000)

        ttk.Spinbox(options, from_=200, to=20000, increment=200, textvariable=self.samples_var, width=8).grid(row=0, column=5, padx=5, sticky=tk.W)



        ttk.Label(options, text="Seed:").grid(row=0, column=6, padx=5, sticky=tk.E)

        self.seed_entry = ttk.Entry(options, width=10)

        self.seed_entry.insert(0, "42")

        self.seed_entry.grid(row=0, column=7, padx=5, sticky=tk.W)





        self.result_frame = ttk.Frame(self.root, padding=10)

        self.result_frame.pack(fill=tk.BOTH, expand=True)



        left = ttk.Frame(self.result_frame)

        left.pack(side=tk.LEFT, fill=tk.Y)



        ttk.Label(left, text="Route (airports):").pack(anchor="w")

        self.route_list = tk.Listbox(left, width=46)

        self.route_list.pack(fill=tk.Y, expand=True)



        ttk.Label(left, text="Top candidates (summary):").pack(anchor="w", pady=(10, 0))

        self.candidate_list = tk.Listbox(left, width=46, height=8)

        self.candidate_list.pack(fill=tk.X)



        self.canvas = tk.Canvas(self.result_frame, bg="lightblue", width=860, height=520)

        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)



        self.map_renderer = WorldMapRenderer(self.canvas)





        self.create_stats_frame()



        self.status_label = ttk.Label(self.root, text="Ready")

        self.status_label.pack(fill=tk.X)



    def create_stats_frame(self):

        self.stats_frame = ttk.Frame(self.root, padding=10)

        self.stats_frame.pack(fill=tk.X)



        basic_frame = ttk.LabelFrame(self.stats_frame, text="Flight Information", padding=5)

        basic_frame.pack(fill=tk.X, pady=5)



        self.distance_var = tk.StringVar()

        self.flight_time_var = tk.StringVar()

        self.route_type_var = tk.StringVar()

        self.stops_var = tk.StringVar()



        ttk.Label(basic_frame, text="Distance:").grid(row=0, column=0, padx=5, sticky=tk.E)

        ttk.Label(basic_frame, textvariable=self.distance_var).grid(row=0, column=1, padx=5, sticky=tk.W)

        ttk.Label(basic_frame, text="Cruise Time:").grid(row=0, column=2, padx=5, sticky=tk.E)

        ttk.Label(basic_frame, textvariable=self.flight_time_var).grid(row=0, column=3, padx=5, sticky=tk.W)

        ttk.Label(basic_frame, text="Route Type:").grid(row=0, column=4, padx=5, sticky=tk.E)

        ttk.Label(basic_frame, textvariable=self.route_type_var).grid(row=0, column=5, padx=5, sticky=tk.W)

        ttk.Label(basic_frame, text="Stops:").grid(row=0, column=6, padx=5, sticky=tk.E)

        ttk.Label(basic_frame, textvariable=self.stops_var).grid(row=0, column=7, padx=5, sticky=tk.W)



        cost_frame = ttk.LabelFrame(self.stats_frame, text="Cost (p10 / median / p90)", padding=5)

        cost_frame.pack(fill=tk.X, pady=5)



        self.fuel_cost_var = tk.StringVar()

        self.operating_cost_var = tk.StringVar()

        self.aircraft_cost_var = tk.StringVar()

        self.airport_fees_var = tk.StringVar()

        self.total_cost_var = tk.StringVar()

        self.ticket_price_var = tk.StringVar()



        ttk.Label(cost_frame, text="Fuel:").grid(row=0, column=0, padx=5, sticky=tk.E)

        ttk.Label(cost_frame, textvariable=self.fuel_cost_var).grid(row=0, column=1, padx=5, sticky=tk.W)

        ttk.Label(cost_frame, text="Ops:").grid(row=0, column=2, padx=5, sticky=tk.E)

        ttk.Label(cost_frame, textvariable=self.operating_cost_var).grid(row=0, column=3, padx=5, sticky=tk.W)

        ttk.Label(cost_frame, text="Aircraft:").grid(row=0, column=4, padx=5, sticky=tk.E)

        ttk.Label(cost_frame, textvariable=self.aircraft_cost_var).grid(row=0, column=5, padx=5, sticky=tk.W)

        ttk.Label(cost_frame, text="Fees:").grid(row=0, column=6, padx=5, sticky=tk.E)

        ttk.Label(cost_frame, textvariable=self.airport_fees_var).grid(row=0, column=7, padx=5, sticky=tk.W)



        ttk.Label(cost_frame, text="Total:").grid(row=1, column=0, padx=5, sticky=tk.E)

        ttk.Label(cost_frame, textvariable=self.total_cost_var).grid(row=1, column=1, padx=5, sticky=tk.W)

        ttk.Label(cost_frame, text="Ticket (per person):").grid(row=1, column=2, padx=5, sticky=tk.E)

        ttk.Label(cost_frame, textvariable=self.ticket_price_var).grid(row=1, column=3, padx=5, sticky=tk.W)



        risk_frame = ttk.LabelFrame(self.stats_frame, text="Risk Metrics (Monte Carlo)", padding=5)

        risk_frame.pack(fill=tk.X, pady=5)



        self.cost_mean_var = tk.StringVar()

        self.cost_p90_var = tk.StringVar()

        self.cost_cvar_var = tk.StringVar()



        self.time_mean_var = tk.StringVar()

        self.time_p90_var = tk.StringVar()

        self.time_cvar_var = tk.StringVar()



        ttk.Label(risk_frame, text="Cost mean:").grid(row=0, column=0, padx=5, sticky=tk.E)

        ttk.Label(risk_frame, textvariable=self.cost_mean_var).grid(row=0, column=1, padx=5, sticky=tk.W)

        ttk.Label(risk_frame, text="Cost p90:").grid(row=0, column=2, padx=5, sticky=tk.E)

        ttk.Label(risk_frame, textvariable=self.cost_p90_var).grid(row=0, column=3, padx=5, sticky=tk.W)

        ttk.Label(risk_frame, text="Cost CVaR95:").grid(row=0, column=4, padx=5, sticky=tk.E)

        ttk.Label(risk_frame, textvariable=self.cost_cvar_var).grid(row=0, column=5, padx=5, sticky=tk.W)



        ttk.Label(risk_frame, text="Time mean:").grid(row=1, column=0, padx=5, sticky=tk.E)

        ttk.Label(risk_frame, textvariable=self.time_mean_var).grid(row=1, column=1, padx=5, sticky=tk.W)

        ttk.Label(risk_frame, text="Time p90:").grid(row=1, column=2, padx=5, sticky=tk.E)

        ttk.Label(risk_frame, textvariable=self.time_p90_var).grid(row=1, column=3, padx=5, sticky=tk.W)

        ttk.Label(risk_frame, text="Time CVaR95:").grid(row=1, column=4, padx=5, sticky=tk.E)

        ttk.Label(risk_frame, textvariable=self.time_cvar_var).grid(row=1, column=5, padx=5, sticky=tk.W)







    def show_suggestions(self, entry_widget: ttk.Entry, listbox_widget: tk.Listbox):

        value = entry_widget.get().lower().strip()

        listbox_widget.delete(0, tk.END)

        if not value:

            return

        suggestions = [city for city in self.city_suggestions if city.lower().startswith(value)]

        for suggestion in suggestions[:10]:

            listbox_widget.insert(tk.END, suggestion)



    def select_suggestion(self, entry_widget: ttk.Entry, listbox_widget: tk.Listbox):

        if listbox_widget.curselection():

            selected = listbox_widget.get(listbox_widget.curselection())

            entry_widget.delete(0, tk.END)

            entry_widget.insert(0, selected)

            listbox_widget.delete(0, tk.END)







    def search_route(self):

        self.status_label.config(text="Searching...")

        self.root.update_idletasks()



        origin_city = self.origin_entry.get().strip()

        destination_city = self.dest_entry.get().strip()



        start_airports = self.data_loader.find_airports_by_city(origin_city)

        end_airports = self.data_loader.find_airports_by_city(destination_city)



        if not start_airports:

            self.status_label.config(text=f"No airports found for city '{origin_city}'.")

            return

        if not end_airports:

            self.status_label.config(text=f"No airports found for city '{destination_city}'.")

            return



        try:

            base_seed = int(self.seed_entry.get().strip())

        except Exception:

            base_seed = _stable_seed(origin_city, destination_city)



        n_samples = int(self.samples_var.get())

        stop_penalty = float(self.stop_penalty_var.get())





        candidates = self.route_finder.k_shortest_paths_multi(

            start_airports, end_airports, k=12, per_leg_penalty_km=stop_penalty

        )

        if not candidates:

            self.status_label.config(text="No route found.")

            return





        objective = self.objective_var.get()



        scored = []

        for idx, pr in enumerate(candidates):

            route = pr.path

            dist_km = self.route_finder.path_distance_km(route)

            if math.isinf(dist_km):

                continue



            metrics = self.cost_analyzer.calculate_flight_metrics(dist_km, route)

            seed_i = base_seed + 1000 * idx

            sim = self.cost_analyzer.simulate_route(

                dist_km, route, metrics["fuel_consumption_liters"], n_samples=n_samples, seed=seed_i

            )



            cost_stats = sim["total_cost_stats"]

            time_stats = sim["total_time_stats"]



            if objective == "Shortest distance":

                key = (dist_km, metrics["stops"])

            elif objective == "Lowest expected cost":

                key = (cost_stats.mean, dist_km)

            else:

                key = (cost_stats.cvar95, cost_stats.mean)



            scored.append((key, dist_km, route, metrics, sim, seed_i))



        if not scored:

            self.status_label.config(text="No valid routes after evaluation.")

            return



        scored.sort(key=lambda x: x[0])

        best = scored[0]

        _, best_dist_km, best_route, best_metrics, best_sim, best_seed = best





        self.candidate_list.delete(0, tk.END)

        for j, (_, d_km, r, m, sim, sseed) in enumerate(scored[:8], start=1):

            c = sim["total_cost_stats"]

            self.candidate_list.insert(

                tk.END,

                f"{j}. {r[0]}→{r[-1]} | {d_km:,.0f} km | stops={m['stops']} | mean=${c.mean:,.0f} | CVaR95=${c.cvar95:,.0f}"

            )





        self.route_list.delete(0, tk.END)

        for airport_code in best_route:

            city, country = self.info_map.get(airport_code, ("Unknown", "Unknown"))

            self.route_list.insert(tk.END, f"{airport_code} - {city}, {country}")





        cost_analysis = self.cost_analyzer.calculate_advanced_costs(

            best_dist_km,

            best_route,

            best_metrics["fuel_consumption_liters"],

            n_samples=n_samples,

            seed=best_seed,

        )



        self.update_flight_info(best_metrics, cost_analysis)

        self.update_cost_display(cost_analysis)



        self.draw_route(best_route)



        self.status_label.config(

            text=f"Done • {objective} • evaluated {len(scored)} routes • seed={best_seed}"

        )







    def update_flight_info(self, metrics, cost_analysis):

        self.distance_var.set(f"{metrics['distance_miles']:,.1f} mi ({metrics['distance_miles']/0.621371:,.0f} km)")

        self.flight_time_var.set(f"{metrics['flight_time_hours']:,.2f} h (cruise only)")

        self.route_type_var.set(str(cost_analysis["route_type"]).title())

        self.stops_var.set(f"{metrics['stops']} stop{'s' if metrics['stops'] != 1 else ''}")



    def _fmt_range(self, trio):

        lo, mid, hi = trio

        return f"${lo:,.0f} / ${mid:,.0f} / ${hi:,.0f}"



    def update_cost_display(self, cost_analysis):

        self.fuel_cost_var.set(self._fmt_range(cost_analysis["fuel_cost"]))

        self.operating_cost_var.set(self._fmt_range(cost_analysis["operating_cost"]))

        self.aircraft_cost_var.set(self._fmt_range(cost_analysis["aircraft_cost"]))

        self.airport_fees_var.set(self._fmt_range(cost_analysis["airport_fees"]))

        self.total_cost_var.set(self._fmt_range(cost_analysis["total_cost"]))





        _, total_expected, _ = cost_analysis["total_cost"]

        ticket_low, ticket_high = self.cost_analyzer.calculate_ticket_prices(total_expected)

        self.ticket_price_var.set(f"${ticket_low:,.0f}–${ticket_high:,.0f}")



        c = cost_analysis["total_cost_stats"]

        t = cost_analysis["total_time_stats"]



        self.cost_mean_var.set(f"${c.mean:,.0f}")

        self.cost_p90_var.set(f"${c.p90:,.0f}")

        self.cost_cvar_var.set(f"${c.cvar95:,.0f}")



        self.time_mean_var.set(f"{t.mean:,.2f} h")

        self.time_p90_var.set(f"{t.p90:,.2f} h")

        self.time_cvar_var.set(f"{t.cvar95:,.2f} h")



    def draw_route(self, route):

        self.canvas.delete("all")

        self.map_renderer.draw_world_map()

        self.map_renderer.draw_flight_route(route, self.df)





def main():

    root = tk.Tk()

    app = FlightRouteApp(root)

    root.mainloop()





