import numpy as np
from dataclasses import dataclass
import random

@dataclass
class VehicleParams:
    # URBAN SETTINGS
    max_speed: float = 13.9      # 50 km/h (approx 14 m/s)
    max_accel: float = 2.0       # Standard city acceleration
    comfortable_brake: float = 2.0 
    min_gap: float = 2.0         
    time_headway: float = 1.5    
    length: float = 5.0          
    delta: float = 4.0           

    # Human Factors
    base_reaction_time: float = 0.5 
    shock_reaction_time: float = 1.0 

CAR_PARAMS = VehicleParams()

class Vehicle:
    def __init__(self, v_id: int, position: float, velocity: float, params: VehicleParams = None):
        self.id = v_id
        self.position = position
        self.velocity = velocity
        self.acceleration = 0.0
        self.params = params if params else CAR_PARAMS
        
        self.reaction_timer = 0.0
        self.last_presence_seen = 0.0
        self.is_squeezing = False 

    def update_physics(self, dt: float, leader=None, bus=None):
        # --- 1. SQUEEZE EXIT LOGIC ---
        if self.is_squeezing:
            if bus:
                safe_pass_point = bus.position + bus.params.length + 5.0
                if self.position > safe_pass_point:
                    self.is_squeezing = False # Safe exit
            elif leader is None:
                self.is_squeezing = False

        # --- 2. REACTION TIMER ---
        self.reaction_timer -= dt
        if leader:
            current_presence = leader.get('presence', 1.0)
            if current_presence < 1.0 and current_presence > self.last_presence_seen:
                diff = current_presence - self.last_presence_seen
                if diff > 0.1 and self.reaction_timer <= 0 and not self.is_squeezing:
                    self.reaction_timer += self.params.shock_reaction_time
            self.last_presence_seen = current_presence

        # --- 3. PHYSICS STEP ---
        if self.reaction_timer <= 0:
            self.acceleration = self._calculate_idm_accel(leader)
            self.reaction_timer = self.params.base_reaction_time + random.uniform(-0.1, 0.1)

        self.velocity += self.acceleration * dt
        if self.velocity < 0:
            self.velocity = 0.0
            self.acceleration = 0.0
        self.position += self.velocity * dt

    def _calculate_idm_accel(self, leader) -> float:
        p = self.params
        target_speed = p.max_speed

        if leader is not None:
            presence = leader.get('presence', 1.0)
            gap = leader['position'] - leader['length'] - self.position
            if gap < 0.1: gap = 0.1
            
            # Calculate Braking Need
            req_brake = (self.velocity ** 2) / (2 * gap)

            # --- PRESSURE REACTION LOGIC ---
            
            if presence == 0.0:
                self.is_squeezing = False
            elif presence == 1.0:
                self.is_squeezing = False
            
            else:
                # BUS IS ASSERTING DOMINANCE (> 0.85)
                if presence > 0.85:
                    # CRITICAL FIX: SURVIVAL CHECK
                    # Even if the bus claims the lane, if we can't stop, we MUST squeeze.
                    if req_brake > 6.0:
                        self.is_squeezing = True
                        # FIX: We keep presence at 1.0 because the car hasn't magically disappeared yet. 
                        # It is still physically occupying the space next to the bus!
                        presence = 1.0 
                        # We also force maximum emergency braking to try and survive
                        return -9.0
                    else:
                        self.is_squeezing = False
                        presence = 1.0 
                
                # BUS IS CREEPING (< 0.85)
                else:
                    panic_squeeze = (req_brake > 4.0)
                    opportunistic_squeeze = (gap < 20.0 and presence < 0.5)
                    keep_squeezing = self.is_squeezing

                    if panic_squeeze or opportunistic_squeeze or keep_squeezing:
                        self.is_squeezing = True 
                        target_speed = p.max_speed * 1.3 
                        presence = 0.0 
                    else:
                        self.is_squeezing = False
                        presence = 1.0 

        # --- STANDARD IDM MATH ---
        if self.velocity < 0.1:
            speed_ratio = 0
        else:
            speed_ratio = (self.velocity / target_speed)
        
        free_road_term = 1.0 - (speed_ratio ** p.delta)

        if leader is None:
            interaction_term = 0.0
        else:
            delta_v = self.velocity - leader['velocity']
            desired_gap = (p.min_gap + 
                           (self.velocity * p.time_headway) + 
                           (self.velocity * delta_v) / (2 * np.sqrt(p.max_accel * p.comfortable_brake)))
            interaction_term = (desired_gap / gap) ** 2
            
            leader_presence = 1.0 if leader is None else presence
            interaction_term *= leader_presence

        accel = p.max_accel * (free_road_term - interaction_term)
        return max(accel, -9.0)

class Bus(Vehicle):
    def __init__(self, v_id, position, velocity, params):
        super().__init__(v_id, position, velocity, params)
        self.state = BusState.MOVING
        self.stop_duration = 15.0
        self.target_stop_x = 300.0
        self.dwell_timer = 0.0
        self.wait_time = 0.0
        self._merge_progress = 0.0 

    def set_stop_schedule(self, location, duration):
        self.target_stop_x = location
        self.stop_duration = duration

    @property
    def presence_factor(self):
        if self.state == BusState.IN_BAY: return 0.0
        if self.state == BusState.STOPPED_IN_LANE: return 1.0
        if self.state == BusState.WAITING_TO_MERGE: return self._merge_progress
        return 1.0

from enum import Enum
class BusState(Enum):
    MOVING = 0
    DECELERATING = 1
    STOPPED_IN_LANE = 2
    IN_BAY = 3
    WAITING_TO_MERGE = 4