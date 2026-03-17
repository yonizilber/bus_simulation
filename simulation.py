import pandas as pd
import random
import numpy as np
import torch
from typing import List
from vehicle import Vehicle, Bus, BusState, CAR_PARAMS
from config import SimConfig
from rl.ppo_agent import PPOAgent

class SimulationEngine:
    def __init__(self, scenario="A", config: SimConfig = None):
        self.scenario = scenario
        self.cfg = config if config else SimConfig()
        self.road_length = self.cfg.road_length
        self.dt = 0.1
        self.time = 0.0
        self.vehicles: List[Vehicle] = []
        self.frame_history = []
        self.global_id_counter = 0 
        self.cars_finished_count = 0
        self.last_spawn_time = 0.0
        self.traffic_rate = self.cfg.traffic_rate
        self.cars_spawned_count = 0
        
        # --- LOAD THE TRAINED PYTORCH BRAIN ---
        self.brain = PPOAgent(filename="rl/checkpoints/ppo_bus_brain.pth")
        self.brain.load() # Loads the trained weights from your Hard Mode run

    def init_scenario(self):
        self.vehicles = []
        self.global_id_counter = 0 
        self.time = 0.0
        self.frame_history = []
        self.cars_spawned_count = 0
        self.cars_finished_count = 0 
        
        # FIX: Spawn the bus at the start line (0.0) so it has room to drive!
        bus = Bus(v_id=999, position=0.0, velocity=10.0, params=CAR_PARAMS)        
        
        # FIX: Force the bus stop to be at 300 meters so we can watch the approach
        bus.set_stop_schedule(300.0, self.cfg.bus_dwell_time) 
        
        self.vehicles.append(bus)

    def step(self):
        self.time += self.dt
        self._try_spawn_car()
        self.vehicles.sort(key=lambda v: v.position, reverse=True)
        the_bus = next((v for v in self.vehicles if isinstance(v, Bus)), None)

        for i, veh in enumerate(self.vehicles):
            # 1. FIND THE LEADER
            lead_vehicle = None
            if i > 0:
                potential_lead = self.vehicles[i-1]
                dist = potential_lead.position - veh.position - potential_lead.params.length
                ignore_bus = False
                
                if isinstance(potential_lead, Bus):
                    if potential_lead.state == BusState.IN_BAY or dist > 100.0:
                        ignore_bus = True

                if ignore_bus:
                    if i > 1: lead_vehicle = self._get_veh_state(self.vehicles[i-2])
                else:
                    lead_vehicle = self._get_veh_state(potential_lead)

            # 2. RUN LOGIC & TELEMETRY
            if isinstance(veh, Bus):
                self._handle_bus_logic(veh)
                
                # Hard stop the bus physically if it is in these states
                if veh.state in [BusState.IN_BAY, BusState.STOPPED_IN_LANE]:
                    veh.velocity = 0.0
                    veh.acceleration = 0.0
            else:
                # Car Telemetry
                if lead_vehicle:
                    gap = lead_vehicle['position'] - lead_vehicle['length'] - veh.position
                    veh.telemetry_leader = f"ID {lead_vehicle.get('id', '?')} ({gap:.1f}m gap)"
                else:
                    veh.telemetry_leader = "Clear Road"
                
                if the_bus:
                    dist_to_bus = the_bus.position - veh.position
                    if dist_to_bus > 100.0 or dist_to_bus < 0 or the_bus.state == BusState.IN_BAY:
                        veh.telemetry_bus = "Out of Sight"
                    else:
                        veh.telemetry_bus = f"{the_bus.presence_factor:.2f}"
                else:
                    veh.telemetry_bus = "No Bus"

            # 3. UPDATE PHYSICS (FOR EVERYONE!)
            # If the bus is MOVING or DECELERATING, it will use the IDM physics to drive forward
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
            bus.wait_time = 0.0 

        # 2. DWELL LOGIC
        if bus.state in [BusState.IN_BAY, BusState.STOPPED_IN_LANE]:
            bus.dwell_timer -= self.dt
            if bus.dwell_timer <= 0:
                if self.scenario == "A":
                    bus.state = BusState.MOVING
                else:
                    bus.state = BusState.WAITING_TO_MERGE
                    bus._merge_progress = 0.0 
                    bus.wait_time = 0.0 

        # 3. AI MERGE LOGIC (The PyTorch Brain)
        if bus.state == BusState.WAITING_TO_MERGE:
            bus.wait_time += self.dt
            bus.velocity = 0.0 
            
            # --- PHASE A: SENSE THE WORLD ---
            gap = 999.0
            nearest_speed = 0.0
            nearest_accel = 0.0
            
            for v in self.vehicles:
                if v.id != bus.id and not isinstance(v, Bus):
                    current_gap = bus.position - bus.params.length - v.position
                    if -10 < current_gap < 60:
                        if abs(current_gap) < abs(gap):
                            gap = current_gap
                            nearest_speed = v.velocity
                            nearest_accel = v.acceleration
            
            # --- PHASE B: ASK THE PYTORCH BRAIN ---
            state_array = np.array([gap, nearest_speed, nearest_accel, bus._merge_progress], dtype=np.float32)
            state_tensor = torch.FloatTensor(state_array).unsqueeze(0)
            
            # We use no_grad and only take the mean to eliminate the "Shaking Foot" randomness
            with torch.no_grad():
                action_mean, _, _ = self.brain.policy(state_tensor)
                pressure = action_mean.item()
            
            pressure = np.clip(pressure, 0.0, 1.0)
            delta = pressure * 0.10
            
            # --- PHASE C: THE PRODUCTION SHIELD ---
            if pressure > 0 and gap > -5.0:
                if gap < 5.0 and nearest_speed > 2.0:
                    delta = 0.0 
                elif pressure > 0.5 and gap < 15.0 and nearest_speed > 10.0:
                    delta = 0.0

            # --- PHASE D: EXECUTE ---
            old_progress = bus._merge_progress
            bus._merge_progress = min(1.0, bus._merge_progress + delta)
            
            # NEW: Translate sideways merging into forward motion (Phase 3)
            # The exit taper is 15 meters.
            progress_made = bus._merge_progress - old_progress
            forward_movement = progress_made * 15.0
            
            # Physically move the bus forward
            bus.position += forward_movement
            
            # Calculate the visual speed for the telemetry/charts
            if progress_made > 0:
                bus.velocity = forward_movement / self.dt
            else:
                bus.velocity = 0.0

            if bus._merge_progress >= 1.0:
                bus.state = BusState.MOVING
                bus._merge_progress = 1.0

    def _try_spawn_car(self):
        if self.cars_spawned_count >= self.cfg.total_cars_to_spawn: return
        if self.time - self.last_spawn_time < 0.5: return
        
        if len(self.vehicles) > 0:
            if self.vehicles[-1].position < 15.0: return 

        if random.random() < (self.traffic_rate * self.dt):
            self.global_id_counter += 1
            start_speed = 11.0 + random.uniform(-1, 1)
            c = Vehicle(self.global_id_counter, position=0.0, velocity=start_speed, params=CAR_PARAMS)
            self.vehicles.append(c)
            self.last_spawn_time = self.time
            self.cars_spawned_count += 1

    def all_vehicles_finished(self) -> bool:
        if self.cars_spawned_count < self.cfg.total_cars_to_spawn: return False
        if len(self.vehicles) == 0: return True
        return False

    def _get_veh_state(self, veh):
        return {
            'id': veh.id,
            'position': veh.position,
            'velocity': veh.velocity,
            'length': veh.params.length,
            'presence': getattr(veh, 'presence_factor', 1.0)
        }

    def _record_frame(self):
        for v in self.vehicles:
            state_desc = v.state.name if isinstance(v, Bus) else "Cruising"
            if not isinstance(v, Bus):
                if v.is_squeezing: state_desc = "Squeezing!"
                elif v.reaction_timer > 0: state_desc = "Reacting..."
            
            final_presence = getattr(v, 'presence_factor', 1.0)
            leader_info = getattr(v, 'telemetry_leader', 'N/A')
            bus_vis_info = getattr(v, 'telemetry_bus', 'N/A')

            self.frame_history.append({
                'time': self.time,
                'id': v.id,
                'type': "Bus" if isinstance(v, Bus) else "Car",
                'x': v.position,
                'v': v.velocity,
                'state': state_desc,
                'presence': final_presence,
                'leader_info': leader_info,
                'bus_visibility': bus_vis_info
            })

    def _remove_finished_cars(self):
        active = []
        for v in self.vehicles:
            if v.position > self.road_length:
                self.cars_finished_count += 1
            else:
                active.append(v)
        self.vehicles = active

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