import math

class TrafficSimulation:
    def __init__(self, traffic_flow_per_min, bus_stop_time, bus_merge_time, passengers):
        self.traffic_flow = traffic_flow_per_min
        self.stop_time = bus_stop_time
        self.merge_time = bus_merge_time
        self.passengers = passengers
        
        # Traffic Constants
        # Saturation Flow: Max capacity of a single lane (approx 1800 cars/hour ~ 0.5 cars/sec)
        self.saturation_flow_rate = 1800 / 3600 

    def run_lane_stop_scenario(self):
        """
        Scenario A: Bus stops in the active lane.
        Uses Deterministic Queuing Theory (Triangle of Delay).
        """
        # 1. Arrival Rate (lambda): cars per second
        arrival_rate = self.traffic_flow / 60.0
        
        # 2. Check for Gridlock
        # If cars arrive faster than the road can handle (saturation), the queue never clears.
        if arrival_rate >= self.saturation_flow_rate:
            return {
                "scenario": "Lane Stop",
                "cost_cars": 99999, # Infinite delay
                "cost_bus": 0,
                "total_cost": 99999,
                "cars_queued": 999,
                "note": "GRIDLOCK: Traffic exceeds road capacity!"
            }

        # 3. Queue Build-up Phase
        # Max Queue Length (vehicles) = Arrival Rate * Stop Time
        q_max = arrival_rate * self.stop_time
        
        # 4. Queue Clearing Phase (Dissipation)
        # The queue shrinks at rate: (Saturation Flow - Arrival Rate)
        # Time to clear queue = Max Queue / (mu - lambda)
        clearing_rate = self.saturation_flow_rate - arrival_rate
        t_clear = q_max / clearing_rate
        
        # 5. Total Delay Calculation (Area of the triangle)
        # Total Delay = 0.5 * (Base) * (Height)
        # Base = Stop Time + Clearing Time
        # Height = Max Queue (in units of time/vehicles) -> actually easier to sum delays
        
        # Alternative Formula for Aggregate Delay (seconds * vehicles):
        # Total Delay = 0.5 * q_max * (self.stop_time + t_clear)
        total_delay_seconds = 0.5 * q_max * (self.stop_time + t_clear)
        
        return {
            "scenario": "Lane Stop",
            "cost_cars": total_delay_seconds,
            "cost_bus": 0,
            "total_cost": total_delay_seconds,
            "cars_queued": math.ceil(q_max),
            "note": f"Queue clears in {t_clear:.1f}s after bus leaves"
        }

    def run_bus_bay_scenario(self):
        """
        Scenario B: Bus pulls into a bay.
        Traffic flows freely, bus waits to merge.
        """
        # Cars are not affected (Cost ~ 0)
        # In reality, there is minor friction, but we assume 0 for contrast.
        time_lost_cars = 0
        
        # Bus passengers wait for merge
        # Cost = Passengers * Merge Time
        time_lost_passengers = self.passengers * self.merge_time
        
        return {
            "scenario": "Bus Bay",
            "cost_cars": time_lost_cars,
            "cost_bus": time_lost_passengers,
            "total_cost": time_lost_passengers,
            "cars_queued": 0, 
            "note": "Traffic flows freely"
        }