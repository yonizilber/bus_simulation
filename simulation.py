import pandas as pd
import random
import numpy as np
import torch
from typing import List
from vehicle import Vehicle, Bus, BusState, CAR_PARAMS, sample_vehicle_params
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

    @staticmethod
    def _encode_brain_state(gap, nearest_speed, nearest_accel, merge_progress, nearest_width, relative_speed):
        gap = float(np.clip(gap, -10.0, 60.0))
        nearest_speed = float(np.clip(nearest_speed, 0.0, 20.0))
        nearest_accel = float(np.clip(nearest_accel, -9.0, 3.0))
        merge_progress = float(np.clip(merge_progress, 0.0, 1.0))
        nearest_width = float(np.clip(nearest_width, 1.70, 2.05))
        relative_speed = float(np.clip(relative_speed, -15.0, 15.0))

        gap_n = (gap + 10.0) / 70.0
        speed_n = nearest_speed / 20.0
        accel_n = nearest_accel / 9.0
        progress_n = merge_progress
        width_n = (nearest_width - 1.85) / 0.20
        rel_speed_n = relative_speed / 15.0

        return np.array([gap_n, speed_n, accel_n, progress_n, width_n, rel_speed_n], dtype=np.float32)

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
                lead_length_for_dist = potential_lead.params.length
                if isinstance(potential_lead, Bus):
                    lead_length_for_dist = potential_lead.projected_length_along_x()
                dist = potential_lead.position - veh.position - lead_length_for_dist
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
                    bus.merge_start_x = bus.position
                    bus.wait_time = 0.0 

        # 3. AI MERGE LOGIC (The PyTorch Brain)
        if bus.state == BusState.WAITING_TO_MERGE:
            bus.wait_time += self.dt
            bus.velocity = 0.0 
            
            # --- PHASE A: SENSE THE WORLD ---
            gap = 999.0
            nearest_speed = 0.0
            nearest_accel = 0.0
            nearest_width = 1.85
            
            for v in self.vehicles:
                if v.id != bus.id and not isinstance(v, Bus):
                    bus_length_x = bus.projected_length_along_x()
                    current_gap = bus.position - bus_length_x - v.position
                    if -10 < current_gap < 60:
                        if abs(current_gap) < abs(gap):
                            gap = current_gap
                            nearest_speed = v.velocity
                            nearest_accel = v.acceleration
                            nearest_width = v.params.width
            
            # --- PHASE B: ASK THE PYTORCH BRAIN ---
            relative_speed = bus.velocity - nearest_speed
            state_array = self._encode_brain_state(gap, nearest_speed, nearest_accel, bus._merge_progress, nearest_width, relative_speed)
            state_tensor = torch.FloatTensor(state_array).unsqueeze(0)
            
            # We use no_grad and only take the mean to eliminate the "Shaking Foot" randomness
            with torch.no_grad():
                action_mean, _, _ = self.brain.policy(state_tensor)
                pressure = action_mean.item()
            
            pressure = np.clip(pressure, 0.0, 1.0)
            delta = pressure * 0.10
            delta = min(delta, bus.max_merge_progress_step)
            
            # --- PHASE C: THE PRODUCTION SHIELD ---
            rel_speed = bus.velocity - nearest_speed
            adaptive_min_gap = 4.0 + max(0.0, nearest_width - 1.80) * 2.2
            if pressure > 0 and gap > -5.0:
                if 0.0 <= gap < 0.50:
                    delta = 0.0
                elif bus._merge_progress > 0.65 and 0.0 <= gap < 1.20 and rel_speed > 0.2:
                    delta = 0.0
                elif 0.0 <= gap < adaptive_min_gap and rel_speed > 1.2:
                    risk = np.clip((adaptive_min_gap - gap) / max(adaptive_min_gap, 1e-3), 0.0, 1.0)
                    delta *= (1.0 - 0.85 * risk)
                elif pressure > 0.65 and 0.0 <= gap < (adaptive_min_gap + 6.0) and rel_speed > 2.5:
                    delta *= 0.35

            if gap < 0.0 and bus._merge_progress > 0.60:
                delta = max(delta, 0.5 * bus.max_merge_progress_step)

            # --- PHASE D: EXECUTE ---
            old_x = bus.position
            bus._merge_progress = min(1.0, bus._merge_progress + delta)
            bus.position = bus.merge_start_x + (bus._merge_progress * bus.exit_taper_length)
            
            # Calculate the visual speed for the telemetry/charts
            progress_made_x = bus.position - old_x
            if progress_made_x > 0:
                bus.velocity = progress_made_x / self.dt
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
            _, sampled_params = sample_vehicle_params()
            start_speed = max(5.5, sampled_params.max_speed * random.uniform(0.68, 0.95))
            c = Vehicle(self.global_id_counter, position=0.0, velocity=start_speed, params=sampled_params)
            self.vehicles.append(c)
            self.last_spawn_time = self.time
            self.cars_spawned_count += 1

    def all_vehicles_finished(self) -> bool:
        if self.cars_spawned_count < self.cfg.total_cars_to_spawn: return False
        if len(self.vehicles) == 0: return True
        return False

    def _get_veh_state(self, veh):
        length_for_gap = veh.params.length
        if isinstance(veh, Bus):
            length_for_gap = veh.projected_length_along_x()

        return {
            'id': veh.id,
            'position': veh.position,
            'velocity': veh.velocity,
            'length': length_for_gap,
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
                'class': "Bus" if isinstance(v, Bus) else getattr(v, 'vehicle_class', 'Mid'),
                'x': v.position,
                'v': v.velocity,
                'length': v.params.length,
                'width': v.params.width,
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