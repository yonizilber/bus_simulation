from dataclasses import dataclass

@dataclass
class SimConfig:
    # --- Traffic Generation ---
    road_length: float = 1000.0   # meters
    traffic_rate: float = 0.3     # cars/second
    total_cars_to_spawn: int = 50 # <--- NEW: Fixed Batch Size
    
    # --- Passenger Loads (The Modular Variables) ---
    pax_per_car: float = 1.2
    pax_per_bus: float = 50.0
    
    # --- Scenario Logic ---
    bus_stop_position: float = 300.0
    bus_dwell_time: float = 15.0  # seconds
    
    # --- Physics Limits ---
    speed_limit_car: float = 50.0 / 3.6 # m/s
    speed_limit_bus: float = 40.0 / 3.6 # m/s