import pandas as pd
import random
import numpy as np
from typing import List, Dict
from vehicle import Vehicle, Bus, BusState, CAR_PARAMS, VehicleParams
from config import SimConfig

class SimulationEngine:
    def __init__(self, scenario="A", config: SimConfig = None):
        self.scenario = scenario
        self.cfg = config if config else SimConfig()
        
        self.road_length = self.cfg.road_length
        self.dt = 0.1
        self.time = 0.0
        
        self.vehicles: List[Vehicle] = []
        self.frame_history = []
        # --- FIX: GLOBAL COUNTER ---
        # Keeps IDs unique forever (1, 2, 3... 999), preventing "Zig-Zags"
        self.global_id_counter = 0 
        self.cars_finished_count = 0
        self.last_spawn_time = 0.0
        self.traffic_rate = self.cfg.traffic_rate
        
        # Batch Mode tracking
        self.cars_spawned_count = 0
        

    def init_scenario(self):
        self.vehicles = []
        self.global_id_counter = 0 # Starts at 0, only goes up
        self.time = 0.0
        self.frame_history = []
        self.cars_spawned_count = 0
        self.cars_finished_count = 0 # <--- NEW COUNTER
        
        # Create Bus
        bus = Bus(v_id=999, position=50.0, velocity=10.0)
        
        bus.set_stop_schedule(
            location=self.cfg.bus_stop_position, 
            duration=self.cfg.bus_dwell_time
        )
        self.vehicles.append(bus)



    def step(self):
        self.time += self.dt
        self._try_spawn_car()

        # Sort by position
        self.vehicles.sort(key=lambda v: v.position, reverse=True)

        # --- NEW: Find the Bus Object ---
        # We need this object so we can pass it to the cars for the "Relative Check"
        the_bus = next((v for v in self.vehicles if isinstance(v, Bus)), None)

        for i, veh in enumerate(self.vehicles):
            lead_vehicle = None
            
            # 1. Look Ahead Logic
            if i > 0:
                potential_lead = self.vehicles[i-1]
                dist = potential_lead.position - veh.position - potential_lead.params.length

                # Ignore distant bus logic
                ignore_bus = False
                if isinstance(potential_lead, Bus):
                    if potential_lead.state == BusState.IN_BAY:
                        ignore_bus = True
                    elif dist > 100.0:
                        ignore_bus = True

                if ignore_bus:
                    if i > 1:
                        lead_vehicle = self._get_veh_state(self.vehicles[i-2])
                    else:
                        lead_vehicle = None
                else:
                    lead_vehicle = self._get_veh_state(potential_lead)



            # 2. Bus Logic
            if isinstance(veh, Bus):
                self._handle_bus_logic(veh)

            # 3. Physics Update
            # Case A: Totally Static
            if isinstance(veh, Bus) and veh.state in [BusState.IN_BAY, BusState.STOPPED_IN_LANE]:
                veh.velocity = 0.0
                veh.acceleration = 0.0

                # FORCE PRESENCE 0.0 HERE (Last moment enforcement)
                if veh.state == BusState.IN_BAY:
                    veh.presence_factor = 0.0
            
            # Case B: Waiting / Emergency
            elif isinstance(veh, Bus) and veh.state == BusState.WAITING_TO_MERGE:
                if veh.velocity > 0:
                    veh.acceleration = -4.5
                    veh.velocity += veh.acceleration * self.dt
                    if veh.velocity < 0: veh.velocity = 0
                else:
                    veh.velocity = 0.0
                    veh.acceleration = 0.0
            
            # Case C: Moving (Cars & Moving Bus)
            else:
                # --- THIS IS THE CRITICAL UPDATE ---
                # We pass 'the_bus' to the car so it can calculate relative position
                veh.update_physics(self.dt, lead_vehicle, bus=the_bus)

        self._record_frame()
        self._remove_finished_cars()

        
    def _handle_bus_logic(self, bus: Bus):
        dist_to_stop = bus.target_stop_x - bus.position
        
        # 1. ARRIVAL LOGIC
        if bus.state == BusState.MOVING and 0 < dist_to_stop < 50:
            bus.state = BusState.DECELERATING
        
        if bus.state == BusState.DECELERATING and dist_to_stop <= 0.5:
            bus.position = bus.target_stop_x
            bus.velocity = 0.0
            bus.dwell_timer = bus.stop_duration
            bus.state = BusState.STOPPED_IN_LANE if self.scenario == "A" else BusState.IN_BAY
            # bus.presence_factor = 0 if self.scenario == "B" else 1.0
            bus.presence_factor = 1.0 if self.scenario == "A" else 0.0

            # Reset Frustration Timer
            bus.wait_time = 0.0 

        # 2. DWELL LOGIC
        if bus.state in [BusState.IN_BAY, BusState.STOPPED_IN_LANE]:
            # Continuous Enforcement:
            if bus.state == BusState.IN_BAY:
                bus.presence_factor = 0.0 # Always 0 while in bay
            else:
                bus.presence_factor = 1.0 # Always 1 while in lane (Scenario A)

            bus.dwell_timer -= self.dt
            if bus.dwell_timer <= 0:
                if self.scenario == "A":
                    bus.state = BusState.MOVING
                else:
                    bus.state = BusState.WAITING_TO_MERGE
                    bus.presence_factor = 0.0 
                    bus.wait_time = 0.0 # Start counting wait time

        # 3. MERGE LOGIC (The Fix)
        if bus.state == BusState.WAITING_TO_MERGE:
            
            # A. Track Frustration
            if not hasattr(bus, 'wait_time'): bus.wait_time = 0.0
            bus.wait_time += self.dt
            
            # "Aggressive Mode" kicks in after 4 seconds of waiting
            is_aggressive = (bus.wait_time > 4.0)

            # B. Recovery from Shock
            if hasattr(bus, 'shock_timer') and bus.shock_timer > 0:
                bus.shock_timer -= self.dt
                # If shocked, brake hard
                if bus.velocity > 0:
                     bus.velocity = max(0, bus.velocity - 4.5 * self.dt)
                return # Still frozen

            # C. Check for NEW Shock (The "Maniac Check")
            # We only panic if we are actually sticking out (presence > 0)
            if bus.presence_factor > 0:
                for v in self.vehicles:
                    if v.id != bus.id and not isinstance(v, Bus):
                        # Detect Squeezing Car
                        if v.is_squeezing:
                            dist = v.position - bus.position
                            # Shock Zone: Car is alongside (-10m to +15m)
                            if -10 < dist < 15:
                                # LOGIC: 
                                # If we are Aggressive, we ignore the car (Don't Panic!)
                                # If we are Passive (wait < 4s), we Panic.
                                if not is_aggressive:
                                    bus.shock_timer = 2.0 # Freeze for 2s
                                    return
                                else:
                                    # We are aggressive; we ignore the squeeze and keep pushing.
                                    pass 

            # D. Nose Out (Increment Presence)
            # If aggressive, we nose out TWICE as fast
            increment = 0.10 if is_aggressive else 0.05
            
            if bus.presence_factor < 1.0:
                bus.presence_factor += increment
            
            # E. Commit & Switch State
            # Once presence is high (0.8), we own the lane.
            is_committed = (bus.presence_factor > 0.8)
            
            if is_committed:
                bus.state = BusState.MOVING 
                bus.presence_factor = 1.0
            # Note: We removed the pure Gap Check here to force reliance on Presence
            # This ensures the bus forces a yield instead of waiting for an empty road.

    # def _handle_bus_logic(self, bus: Bus):
    #     dist_to_stop = bus.target_stop_x - bus.position
        
    #     if bus.state == BusState.MOVING and 0 < dist_to_stop < 50:
    #         bus.state = BusState.DECELERATING
        
    #     if bus.state == BusState.DECELERATING and dist_to_stop <= 0.5:
    #         bus.position = bus.target_stop_x
    #         bus.velocity = 0.0
    #         bus.dwell_timer = bus.stop_duration
    #         bus.state = BusState.STOPPED_IN_LANE if self.scenario == "A" else BusState.IN_BAY
    #         bus.presence_factor = 1.0 if self.scenario == "A" else 0.0

    #     if bus.state in [BusState.IN_BAY, BusState.STOPPED_IN_LANE]:
    #         bus.dwell_timer -= self.dt
    #         if bus.dwell_timer <= 0:
    #             if self.scenario == "A":
    #                 bus.state = BusState.MOVING
    #             else:
    #                 bus.state = BusState.WAITING_TO_MERGE
    #                 bus.presence_factor = 0.0 

    #     # 4. MERGE LOGIC
    #     if bus.state == BusState.WAITING_TO_MERGE:
            
    #         # A. Check Shock Timer (Are we currently frozen?)
    #         if hasattr(bus, 'shock_timer') and bus.shock_timer > 0:
    #             bus.shock_timer -= self.dt
    #             # FORCE STOP while shocked
    #             bus.velocity = 0.0 
    #             # CRITICAL: Force the state name to be "SHOCKED" so Visualizer sees it
    #             # We can't change the Enum, but we can rely on the state staying 'WAITING'
    #             # and the Visualizer checking the timer, OR we hack the display state.
    #             # For now, let's trust the Visualizer checks "SHOCKED" in string 
    #             # or we add a specific flag.
    #             return 

    #         # B. Check for NEW Panic (Make this sensitive!)
    #         # If a car is "Squeezing" (Purple) nearby, panic!
    #         for v in self.vehicles:
    #             if v.id != bus.id and not isinstance(v, Bus):
    #                 # If car is Squeezing AND close to Bus
    #                 dist = v.position - bus.position
    #                 # If car is overtaking (-10m behind to +10m ahead)
    #                 if -10 < dist < 15 and v.is_squeezing:
    #                     bus.shock_timer = 2.0 # Freeze for 2s
    #                     # OPTIONAL: Print to debug
    #                     # print(f"Bus Shocked by Car {v.id}")
    #                     return

    #         # C. Nose Out (Normal)
    #         if bus.presence_factor < 1.0:
    #             bus.presence_factor += 0.05 
            
    #         is_committed = (bus.presence_factor > 0.8)
            
    #         # D. Switch State
    #         if is_committed or self._check_gap_condition(bus, req_ttc=2.0):
    #             bus.state = BusState.MOVING 
    #             bus.presence_factor = 1.0
        


    
    def _check_gap_condition(self, bus: Bus, req_ttc: float) -> bool:
        bus_index = self.vehicles.index(bus)
        if bus_index == len(self.vehicles) - 1:
            return True 
            
        follower = self.vehicles[bus_index + 1]
        gap = bus.position - bus.params.length - follower.position
        
        # YIELD LOGIC
        # If car is slow (yielding) AND we have physical room (>1m), GO.
        if bus.presence_factor > 0.1 and follower.velocity < 5.5 and gap > 1.0:
            return True

        # Standard Safety (for fast cars)
        if gap < 2.0: return False 
        
        approach_speed = follower.velocity - bus.velocity
        if approach_speed > 0:
            ttc = gap / approach_speed
            if ttc < req_ttc: 
                return False
                
        return True

    def _is_safe_to_merge(self, bus: Bus) -> bool:
        """Gap Acceptance Logic."""
        bus_index = self.vehicles.index(bus)
        if bus_index == len(self.vehicles) - 1:
            return True # No one behind
            
        follower = self.vehicles[bus_index + 1]
        gap = bus.position - bus.params.length - follower.position
        
        # 1. Hard Distance Check
        if gap < 15.0: return False
        
        # 2. Time To Collision Check
        approach_speed = follower.velocity - bus.velocity
        if approach_speed > 0:
            ttc = gap / approach_speed
            if ttc < 3.0: return False
                
        return True

    def _try_spawn_car(self):
        # Stop if batch is full
        if self.cars_spawned_count >= self.cfg.total_cars_to_spawn:
            return

        if self.time - self.last_spawn_time < 0.5: return
        
        # Check if the start area is blocked
        if len(self.vehicles) > 0:
            last_veh = self.vehicles[-1]
            if last_veh.position < 15.0: return 

        if random.random() < (self.traffic_rate * self.dt):
            # --- FIX STARTS HERE ---
            self.global_id_counter += 1
            new_id = self.global_id_counter 
            # Now every car gets a unique ID (1, 2, 3...) that is never reused.
            
            c = Vehicle(new_id, position=0.0, velocity=12.0 + random.uniform(-2, 2), params=CAR_PARAMS)
            self.vehicles.append(c)
            self.last_spawn_time = self.time
            self.cars_spawned_count += 1

    def all_vehicles_finished(self) -> bool:
        """Checks if the batch is done."""
        # 1. Have we spawned everyone?
        if self.cars_spawned_count < self.cfg.total_cars_to_spawn:
            return False
            
        # 2. Is the road empty? (Since we remove finished cars now)
        if len(self.vehicles) == 0:
            return True
            
        return False
            
        # Check if everyone spawned has reached the end
        for v in self.vehicles:
            if v.position < (self.road_length - 10):
                return False
        return True

    def _get_veh_state(self, veh):
        # Default presence for normal cars is 1.0 (Solid)
        presence = 1.0
        
        # Only Buses have variable presence
        if isinstance(veh, Bus):
            presence = veh.presence_factor
            
        return {
            'position': veh.position,
            'velocity': veh.velocity,
            'length': veh.params.length,
            'presence': presence
        }

    def _record_frame(self):
        for v in self.vehicles:
            # 1. Determine "State Description" for the Graph
            state_desc = "Cruising"
            color_code = "blue" # Default for cars
            
            if isinstance(v, Bus):
                state_desc = v.state.name # e.g. "WAITING_TO_MERGE"
                if v.state == BusState.IN_BAY: color_code = "gray"
                elif v.state == BusState.WAITING_TO_MERGE: color_code = "orange"
                elif v.state == BusState.MOVING: color_code = "red"
            
            # ADD THIS:
            if isinstance(v, Bus) and hasattr(v, 'shock_timer') and v.shock_timer > 0:
                state_desc += " (SHOCKED)" # This triggers the Black color in App.py
                
                # Add detail about shock
                if hasattr(v, 'shock_timer') and v.shock_timer > 0:
                    state_desc += " (SHOCKED)"
            else:
                # Car Logic
                if v.is_squeezing:
                    state_desc = "Squeezing!"
                    color_code = "purple" # Highlight aggressive cars
                elif v.reaction_timer > 0.5: # Arbitrary threshold for "processing"
                    state_desc = "Reacting..."

            
            # 2. Record Data
            self.frame_history.append({
                'time': round(self.time, 2),
                'id': v.id,
                'x': round(v.position, 2),
                'v': round(v.velocity, 2),
                'type': "Bus" if isinstance(v, Bus) else "Car",
                'state': state_desc,
                'color': color_code,
                'presence': getattr(v, 'presence_factor', 1.0)
            })

            
    def _remove_finished_cars(self):
        """Sink Logic: Remove cars that finished."""
        active_vehicles = []
        for v in self.vehicles:
            if v.position > self.road_length:
                self.cars_finished_count += 1
            else:
                active_vehicles.append(v)
        self.vehicles = active_vehicles


    def _is_emergency_stop_needed(self, bus: Bus) -> bool:
        """
        Returns True if a crash is imminent regardless of rules.
        """
        bus_index = self.vehicles.index(bus)
        if bus_index == len(self.vehicles) - 1:
            return False
            
        follower = self.vehicles[bus_index + 1]
        gap = bus.position - bus.params.length - follower.position
        
        # 1. Immediate Impact Danger (< 5m and fast)
        if gap < 5.0 and follower.velocity > 10.0:
            return True
            
        # 2. Critical TTC Danger (< 1.5s)
        approach_speed = follower.velocity - bus.velocity
        if approach_speed > 0:
            ttc = gap / approach_speed
            if ttc < 1.5:
                return True
                
        return False
            
    def get_results(self):
        return pd.DataFrame(self.frame_history)

class AnalyticsEngine:
    def __init__(self, dataframe: pd.DataFrame, config: SimConfig):
        self.df = dataframe
        self.cfg = config

    def compute_kpis(self):
        if self.df.empty:
            return {"avg_person_speed": 0.0, "throughput_people": 0}

        vehicles = self.df.groupby('id')
        
        total_w_dist = 0.0
        total_w_time = 0.0
        total_people_arrived = 0

        for v_id, trip in vehicles:
            v_type = trip.iloc[0]['type']
            pax = self.cfg.pax_per_bus if v_type == 'Bus' else self.cfg.pax_per_car
            
            dist = trip['x'].max() - trip['x'].min()
            duration = trip['time'].max() - trip['time'].min()
            
            if duration < 0.1: continue

            total_w_dist += (dist * pax)
            total_w_time += (duration * pax)
            
            # Count PEOPLE who finished
            if trip['x'].max() > (self.cfg.road_length - 20):
                total_people_arrived += pax

        if total_w_time > 0:
            avg_person_speed = (total_w_dist / total_w_time) * 3.6
        else:
            avg_person_speed = 0.0
            
        return {
            "avg_person_speed": avg_person_speed,
            "throughput_people": int(total_people_arrived) 
        }
class AnalyticsEngine:
    def __init__(self, dataframe, config: SimConfig):
        self.df = dataframe
        self.cfg = config

    def compute_kpis(self):
        if self.df.empty:
            return {"avg_person_speed": 0.0, "throughput_people": 0}

        vehicles = self.df.groupby('id')
        
        total_w_dist = 0.0
        total_w_time = 0.0
        total_people_arrived = 0

        for v_id, trip in vehicles:
            v_type = trip.iloc[0]['type']
            pax = self.cfg.pax_per_bus if v_type == 'Bus' else self.cfg.pax_per_car
            
            dist = trip['x'].max() - trip['x'].min()
            duration = trip['time'].max() - trip['time'].min()
            
            if duration < 0.1: continue

            total_w_dist += (dist * pax)
            total_w_time += (duration * pax)
            
            # Count PEOPLE who finished (New Logic)
            if trip['x'].max() > (self.cfg.road_length - 20):
                total_people_arrived += pax

        if total_w_time > 0:
            avg_person_speed = (total_w_dist / total_w_time) * 3.6
        else:
            avg_person_speed = 0.0
            
        return {
            "avg_person_speed": avg_person_speed,
            "throughput_people": int(total_people_arrived) 
        }