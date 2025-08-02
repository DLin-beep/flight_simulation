#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk
import math

from src.data_loader import FlightDataLoader
from src.route_finder import RouteFinder
from src.cost_analyzer import FlightCostAnalyzer
from src.world_map import WorldMapRenderer

class FlightRouteApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Flight Route Finder")
        
        self.data_loader = FlightDataLoader()
        self.cost_analyzer = FlightCostAnalyzer()
        
        self.df = self.data_loader.load_airport_data()
        self.graph = self.data_loader.build_flight_network()
        self.route_finder = RouteFinder(self.graph)
        
        self.create_widgets()
        self.city_suggestions = self.data_loader.get_city_suggestions()
    
    def create_widgets(self):
        input_frame = ttk.Frame(self.root, padding=10)
        input_frame.pack(fill=tk.X)
        
        ttk.Label(input_frame, text="Origin City:").grid(row=0, column=0, padx=5)
        self.origin_entry = ttk.Entry(input_frame, width=30)
        self.origin_entry.grid(row=0, column=1, padx=5)
        self.origin_entry.bind('<KeyRelease>', lambda e: self.show_suggestions(self.origin_entry, self.origin_listbox))
        
        self.origin_listbox = tk.Listbox(input_frame, height=5)
        self.origin_listbox.grid(row=1, column=1, padx=5, sticky='ew')
        self.origin_listbox.bind('<<ListboxSelect>>', lambda e: self.select_suggestion(self.origin_entry, self.origin_listbox))
        
        ttk.Label(input_frame, text="Destination City:").grid(row=0, column=2, padx=5)
        self.dest_entry = ttk.Entry(input_frame, width=30)
        self.dest_entry.grid(row=0, column=3, padx=5)
        self.dest_entry.bind('<KeyRelease>', lambda e: self.show_suggestions(self.dest_entry, self.dest_listbox))
        
        self.dest_listbox = tk.Listbox(input_frame, height=5)
        self.dest_listbox.grid(row=1, column=3, padx=5, sticky='ew')
        self.dest_listbox.bind('<<ListboxSelect>>', lambda e: self.select_suggestion(self.dest_entry, self.dest_listbox))
        
        search_btn = ttk.Button(input_frame, text="Find Route", command=self.search_route)
        search_btn.grid(row=0, column=4, padx=5)
        
        self.result_frame = ttk.Frame(self.root, padding=10)
        self.result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.route_list = tk.Listbox(self.result_frame, width=50)
        self.route_list.pack(side=tk.LEFT, fill=tk.Y)
        
        self.canvas = tk.Canvas(self.result_frame, bg="lightblue", width=800, height=500)
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
        ttk.Label(basic_frame, text="Flight Time:").grid(row=0, column=2, padx=5, sticky=tk.E)
        ttk.Label(basic_frame, textvariable=self.flight_time_var).grid(row=0, column=3, padx=5, sticky=tk.W)
        ttk.Label(basic_frame, text="Route Type:").grid(row=0, column=4, padx=5, sticky=tk.E)
        ttk.Label(basic_frame, textvariable=self.route_type_var).grid(row=0, column=5, padx=5, sticky=tk.W)
        ttk.Label(basic_frame, text="Stops:").grid(row=0, column=6, padx=5, sticky=tk.E)
        ttk.Label(basic_frame, textvariable=self.stops_var).grid(row=0, column=7, padx=5, sticky=tk.W)
        
        cost_frame = ttk.LabelFrame(self.stats_frame, text="Cost Analysis", padding=5)
        cost_frame.pack(fill=tk.X, pady=5)
        
        self.fuel_cost_var = tk.StringVar()
        self.operating_cost_var = tk.StringVar()
        self.aircraft_cost_var = tk.StringVar()
        self.airport_fees_var = tk.StringVar()
        self.total_cost_var = tk.StringVar()
        self.ticket_price_var = tk.StringVar()
        
        ttk.Label(cost_frame, text="Fuel Cost:").grid(row=0, column=0, padx=5, sticky=tk.E)
        ttk.Label(cost_frame, textvariable=self.fuel_cost_var).grid(row=0, column=1, padx=5, sticky=tk.W)
        ttk.Label(cost_frame, text="Operating Cost:").grid(row=0, column=2, padx=5, sticky=tk.E)
        ttk.Label(cost_frame, textvariable=self.operating_cost_var).grid(row=0, column=3, padx=5, sticky=tk.W)
        ttk.Label(cost_frame, text="Aircraft Cost:").grid(row=0, column=4, padx=5, sticky=tk.E)
        ttk.Label(cost_frame, textvariable=self.aircraft_cost_var).grid(row=0, column=5, padx=5, sticky=tk.W)
        ttk.Label(cost_frame, text="Airport Fees:").grid(row=0, column=6, padx=5, sticky=tk.E)
        ttk.Label(cost_frame, textvariable=self.airport_fees_var).grid(row=0, column=7, padx=5, sticky=tk.W)
        
        ttk.Label(cost_frame, text="Total Cost:").grid(row=1, column=0, padx=5, sticky=tk.E)
        ttk.Label(cost_frame, textvariable=self.total_cost_var).grid(row=1, column=1, padx=5, sticky=tk.W)
        ttk.Label(cost_frame, text="Ticket Price (per person):").grid(row=1, column=2, padx=5, sticky=tk.E)
        ttk.Label(cost_frame, textvariable=self.ticket_price_var).grid(row=1, column=3, padx=5, sticky=tk.W)
    
    def show_suggestions(self, entry_widget, listbox_widget):
        value = entry_widget.get().lower()
        if value == '':
            listbox_widget.delete(0, tk.END)
            return
        
        suggestions = [city for city in self.city_suggestions if city.lower().startswith(value)]
        
        listbox_widget.delete(0, tk.END)
        for suggestion in suggestions[:10]:
            listbox_widget.insert(tk.END, suggestion)
    
    def select_suggestion(self, entry_widget, listbox_widget):
        if listbox_widget.curselection():
            selected = listbox_widget.get(listbox_widget.curselection())
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, selected)
            listbox_widget.delete(0, tk.END)
    
    def search_route(self):
        self.status_label.config(text="Searching...")
        self.root.update_idletasks()
        
        origin_city = self.origin_entry.get().strip().lower()
        destination_city = self.dest_entry.get().strip().lower()
        
        start_airports = self.data_loader.find_airports_by_city(origin_city)
        end_airports = self.data_loader.find_airports_by_city(destination_city)
        
        if not start_airports:
            self.status_label.config(text=f"No airports found for city '{origin_city}'")
            return
        if not end_airports:
            self.status_label.config(text=f"No airports found for city '{destination_city}'")
            return
        
        dist_km, route = self.route_finder.optimized_dijkstra(start_airports, end_airports)
        
        if math.isinf(dist_km):
            self.status_label.config(text="No route found")
            return
        
        self.route_list.delete(0, tk.END)
        info_map = {r["iata"]: (r["city"], r["country"]) for _, r in self.df.iterrows()}
        for airport_code in route:
            city, country = info_map.get(airport_code, ("Unknown", "Unknown"))
            self.route_list.insert(tk.END, f"{airport_code} - {city}, {country}")
        
        metrics = self.cost_analyzer.calculate_flight_metrics(dist_km, route)
        fuel_consumption = metrics['fuel_consumption_liters']
        
        cost_analysis = self.cost_analyzer.calculate_advanced_costs(dist_km, route, fuel_consumption)
        
        self.update_flight_info(metrics, cost_analysis)
        self.update_cost_display(cost_analysis)
        
        self.draw_route(route)
        
        self.status_label.config(text="Done")
    
    def update_flight_info(self, metrics, cost_analysis):
        self.distance_var.set(f"{format(metrics['distance_miles'], ',.1f')} mi")
        self.flight_time_var.set(f"{format(metrics['flight_time_hours'], ',.1f')} hours")
        self.route_type_var.set(cost_analysis['route_type'].title())
        self.stops_var.set(f"{metrics['stops']} stop{'s' if metrics['stops'] != 1 else ''}")
    
    def update_cost_display(self, cost_analysis):
        fuel_low, fuel_expected, fuel_high = cost_analysis['fuel_cost']
        op_low, op_expected, op_high = cost_analysis['operating_cost']
        ac_low, ac_expected, ac_high = cost_analysis['aircraft_cost']
        af_low, af_expected, af_high = cost_analysis['airport_fees']
        total_low, total_expected, total_high = cost_analysis['total_cost']
        
        self.fuel_cost_var.set(f"${format(fuel_low, ',.0f')}-${format(fuel_high, ',.0f')}")
        self.operating_cost_var.set(f"${format(op_low, ',.0f')}-${format(op_high, ',.0f')}")
        self.aircraft_cost_var.set(f"${format(ac_low, ',.0f')}-${format(ac_high, ',.0f')}")
        self.airport_fees_var.set(f"${format(af_low, ',.0f')}-${format(af_high, ',.0f')}")
        self.total_cost_var.set(f"${format(total_low, ',.0f')}-${format(total_high, ',.0f')}")
        
        ticket_low, ticket_high = self.cost_analyzer.calculate_ticket_prices(total_expected)
        self.ticket_price_var.set(f"${format(ticket_low, ',.0f')}-${format(ticket_high, ',.0f')}")
    
    def draw_route(self, route):
        self.canvas.delete("all")
        self.map_renderer.draw_world_map()
        self.map_renderer.draw_flight_route(route, self.df)

def main():
    root = tk.Tk()
    app = FlightRouteApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 
