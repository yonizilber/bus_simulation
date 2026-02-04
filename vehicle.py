import numpy as np
from dataclasses import dataclass
import random

@dataclass
class VehicleParams:
    max_speed: float = 33.3         # 120 km/h
    max_accel: float = 2.0          # m/s^2 (Comfortable gas)
    comfortable_brake: float = 2.0  # m/s^2 (Comfortable brake)
    min_gap: float = 2.0            # meters (Stopped gap)
    time_headway: float = 1.5       # seconds (Following distance)
    length: float = 5.0             # meters
    delta: float = 4.0              # IDM exponent

    # Human Factors
    base_reaction_time: float = 0.5 # Normal reaction time (seconds)
    shock_reaction_time: float = 1.5 # Surprise reaction time (seconds)

# Standard Car Configuration
CAR_PARAMS = VehicleParams()

class Vehicle:
    def __init__(self, v_id: int, position: float, velocity: float, params: VehicleParams = None):
        self.id = v_id
        self.position = position
        self.velocity = velocity
        self.acceleration = 0.0
        self.params = params if params else CAR_PARAMS
        
        # Human State
        self.reaction_timer = 0.0
        self.last_presence_seen = 0.0
        self.is_squeezing = False # "Commitment" to pass the bus

    def update_physics(self, dt: float, leader=None, bus=None):
        # --- 1. RELATIVE POSITION LOGIC (The New Fix) ---
        # We check if we have actually passed the bus physically.
        # This replaces the old timer/cooldown logic.
        if self.is_squeezing:
            # If the simulation passed us the bus object:
            if bus:
                # Calculate the safe "all clear" point: Bus Position + Bus Length + 5m buffer
                safe_pass_point = bus.position + bus.params.length + 5.0
                
                # If we haven't reached that point yet, FORCE Squeezing to stay ON.
                if self.position <= safe_pass_point:
                    pass # Keep Squeezing!
                
                # If we HAVE passed it, we can relax.
                else:
                    # We start a small cooldown to transition smoothly back to normal
                    if not hasattr(self, 'squeeze_cooldown_timer'):
                        self.squeeze_cooldown_timer = 1.0 
            
            # Fallback: If no bus object exists (shouldn't happen), keep squeezing to be safe
            elif leader is None:
                pass 

        # --- Handle the Exit Timer (Smooth Transition) ---
        if hasattr(self, 'squeeze_cooldown_timer') and self.squeeze_cooldown_timer > 0:
            self.squeeze_cooldown_timer -= dt
            self.is_squeezing = True # Force ON
            if self.squeeze_cooldown_timer <= 0:
                self.is_squeezing = False # Finally OFF
                del self.squeeze_cooldown_timer

        # --- 2. REACTION TIMER LOGIC ---
        self.reaction_timer -= dt
        
        if leader:
            current_presence = leader.get('presence', 1.0)
        
        # --- 2. Update Reaction Timer (Standard Logic) ---
        self.reaction_timer -= dt
        
        # --- 3. Check for "Surprise" (Standard Logic) ---
        if leader:
            current_presence = leader.get('presence', 1.0)
            if current_presence - self.last_presence_seen > 0.1:
                self.reaction_timer += self.params.shock_reaction_time
            self.last_presence_seen = current_presence

        # --- 4. Human Decision Loop (Standard Logic) ---
        if self.reaction_timer <= 0:
            self.acceleration = self._calculate_idm_accel(leader)
            self.reaction_timer = self.params.base_reaction_time + random.uniform(-0.1, 0.1)

        # --- 5. Apply Physics (Standard Logic) ---
        self.velocity += self.acceleration * dt
        if self.velocity < 0:
            self.velocity = 0.0
            self.acceleration = 0.0
            
        self.position += self.velocity * dt

    def _calculate_idm_accel(self, leader) -> float:
        p = self.params
        target_speed = p.max_speed

        # --- FIX 1: BETTER EXIT LOGIC ---
        # Stop squeezing if:
        # A. The road is empty (leader is None)
        # B. We are following a REAL car (presence == 1.0), not a ghost bus.
        if self.is_squeezing:
            if leader is None or leader.get('presence', 1.0) == 1.0:
                self.is_squeezing = False
        
        # --- HUMAN LOGIC: The Squeeze (Dilemma Zone) ---
        if leader is not None:
            presence = leader.get('presence', 1.0)
            gap = leader['position'] - leader['length'] - self.position
            if gap < 0.1: gap = 0.1

            # Only consider squeezing if it's a "Ghost" (Merging Bus)
            if presence < 1.0:
                # 1. Calculate Required Braking
                req_brake = (self.velocity ** 2) / (2 * gap)
                
                # 2. Decision Matrix
                if self.is_squeezing:
                    # Check abort condition
                    if gap <= 5.0 and presence > 0.8:
                        self.is_squeezing = False 
                    else:
                        target_speed = p.max_speed * 1.1 
                        presence = 0.0 # Ignore bus completely

                elif req_brake > 3.0 and presence < 0.5:
                    self.is_squeezing = True 
                    target_speed = p.max_speed * 1.1
                    presence = 0.0
                else:
                    self.is_squeezing = False
                    # FIX: If we decide to YIELD, we must respect the bus!
                    # We override the ghost factor. We treat it as a real car (presence 1.0).
                    # Otherwise, we won't brake hard enough to actually stop.
                    presence = 1.0 

        # --- Standard IDM Calculation ---
        if self.velocity < 0.1:
            speed_ratio = 0
        else:
            speed_ratio = (self.velocity / target_speed)
        
        free_road_term = 1.0 - (speed_ratio ** p.delta)

        if leader is None:
            interaction_term = 0.0
        else:
            gap = leader['position'] - leader['length'] - self.position
            if gap < 0.1: gap = 0.1
            delta_v = self.velocity - leader['velocity']
            desired_gap = (p.min_gap + 
                           (self.velocity * p.time_headway) + 
                           (self.velocity * delta_v) / (2 * np.sqrt(p.max_accel * p.comfortable_brake)))
            
            interaction_term = (desired_gap / gap) ** 2
            
            # Use the Modified Presence (0.0 if squeezing, 1.0 if yielding)
            leader_presence = 1.0 if leader is None else presence
            
            interaction_term *= leader_presence

        accel = p.max_accel * (free_road_term - interaction_term)
        return max(accel, -9.0)



# Bus Logic (Simplified for import, state managed in SimulationEngine)
class Bus(Vehicle):
    def set_stop_schedule(self, location, duration):
        self.target_stop_x = location
        self.stop_duration = duration
        self.dwell_timer = 0.0
        self.state = BusState.MOVING
        self.presence_factor = 1.0
        self.shock_timer = 0.0 # Post-scare recovery timer



from enum import Enum
class BusState(Enum):
    MOVING = 0
    DECELERATING = 1
    STOPPED_IN_LANE = 2 # Scenario A
    IN_BAY = 3          # Scenario B
    WAITING_TO_MERGE = 4