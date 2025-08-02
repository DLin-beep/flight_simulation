import random
import math

class FlightCostAnalyzer:
    
    def __init__(self):
        self.FLIGHT_SPEED_KMH = 900
        self.KM_TO_MILES = 0.621371
        self.FUEL_BURN_RATE_L_PER_KM = 5
        
        self.fuel_cost_per_l = {
            'domestic': (0.7, 0.9),
            'international': (0.8, 1.2),
            'long_haul': (0.9, 1.4)
        }
        
        self.operating_cost_per_km = {
            'domestic': (0.15, 0.25),
            'international': (0.20, 0.35),
            'long_haul': (0.25, 0.45)
        }
        
        self.aircraft_cost_per_km = {
            'domestic': (0.10, 0.18),
            'international': (0.15, 0.25),
            'long_haul': (0.20, 0.35)
        }
    
    def calculate_advanced_costs(self, distance_km, route, fuel_consumption):
        
        if distance_km < 1000:
            route_type = 'domestic'
        elif distance_km < 5000:
            route_type = 'international'
        else:
            route_type = 'long_haul'
        
        fuel_cost_range = self.fuel_cost_per_l[route_type]
        base_fuel_cost = fuel_consumption * random.uniform(*fuel_cost_range)
        
        op_cost_range = self.operating_cost_per_km[route_type]
        operating_cost = distance_km * random.uniform(*op_cost_range)
        
        ac_cost_range = self.aircraft_cost_per_km[route_type]
        aircraft_cost = distance_km * random.uniform(*ac_cost_range)
        
        airport_fees = len(route) * random.uniform(50, 200)
        
        seasonal_multiplier = random.uniform(0.9, 1.3)
        
        stops = len(route) - 1
        efficiency_multiplier = 1.0
        if stops == 0:
            efficiency_multiplier = 0.85
        elif stops > 2:
            efficiency_multiplier = 1.25
        
        total_cost = (base_fuel_cost + operating_cost + aircraft_cost + airport_fees) * seasonal_multiplier * efficiency_multiplier
        
        low_cost = total_cost * 0.85
        expected_cost = total_cost
        high_cost = total_cost * 1.25
        
        return {
            'fuel_cost': (base_fuel_cost * 0.85, base_fuel_cost, base_fuel_cost * 1.15),
            'operating_cost': (operating_cost * 0.9, operating_cost, operating_cost * 1.2),
            'aircraft_cost': (aircraft_cost * 0.9, aircraft_cost, aircraft_cost * 1.2),
            'airport_fees': (airport_fees * 0.8, airport_fees, airport_fees * 1.3),
            'total_cost': (low_cost, expected_cost, high_cost),
            'route_type': route_type,
            'efficiency_multiplier': efficiency_multiplier,
            'seasonal_multiplier': seasonal_multiplier
        }
    
    def calculate_flight_metrics(self, distance_km, route):
        dist_miles = distance_km * self.KM_TO_MILES
        flight_time = distance_km / self.FLIGHT_SPEED_KMH
        fuel_consumption = distance_km * self.FUEL_BURN_RATE_L_PER_KM
        stops = len(route) - 1 if len(route) > 1 else 0
        
        return {
            'distance_miles': dist_miles,
            'flight_time_hours': flight_time,
            'fuel_consumption_liters': fuel_consumption,
            'stops': stops
        }
    
    def calculate_ticket_prices(self, total_cost_expected, seats=150):
        profit_margin_low = 0.20
        profit_margin_high = 0.40
        
        ticket_low = (total_cost_expected * (1 + profit_margin_low)) / seats
        ticket_high = (total_cost_expected * (1 + profit_margin_high)) / seats
        
        return ticket_low, ticket_high 
