import numpy as np
import random
import sys
import os

# Ensure we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vehicle import Vehicle, Bus, BusState, CAR_PARAMS, sample_vehicle_params

class BusMergeEnv:
    def __init__(self):
        self.dt = 0.1
        self.max_steps = 500
        
        # PyTorch Network Dimensions
        self.state_size = 6  # [Gap, Speed, Accel, Merge_Progress, Nearest_Car_Width, Relative_Speed]
        self.action_size = 1 # [Merge_Pressure] -> Continuous float from -1.0 to 1.0
        
        self.bus = None
        self.vehicles = []
        self.current_step = 0
        self.last_intervened = False
        self.last_shield_condition = ""
        self.episode_shields = 0
        self.last_success = False
        self.last_crash = False
        self.last_timeout = False

    def _encode_state(self, gap, nearest_speed, nearest_accel, merge_progress, nearest_width, relative_speed):
        """Keep RL features in compact ranges to reduce training oscillations."""
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

    def _scan_nearest_car(self):
        """Return nearest relevant car context in physical units."""
        gap = 999.0
        nearest_speed = 0.0
        nearest_accel = 0.0
        nearest_width = 1.85

        for v in self.vehicles:
            current_gap = self.bus.position - self.bus.projected_length_along_x() - v.position
            if -10 < current_gap < 60:
                if abs(current_gap) < abs(gap):
                    gap = current_gap
                    nearest_speed = v.velocity
                    nearest_accel = v.acceleration
                    nearest_width = v.params.width

        relative_speed = self.bus.velocity - nearest_speed
        return gap, nearest_speed, nearest_accel, nearest_width, relative_speed

    def reset(self):
        self.current_step = 0
        self.episode_shields = 0
        self.last_shield_condition = ""
        
        # 1. Spawn Bus directly in the bay (skip the driving phase)
        self.bus = Bus(v_id=999, position=300.0, velocity=0.0, params=CAR_PARAMS)
        self.bus.state = BusState.WAITING_TO_MERGE
        self.bus._merge_progress = 0.0
        self.bus.merge_start_x = self.bus.position
        
        # 2. Spawn a random stream of 1 to 3 cars approaching
        self.vehicles = []
        num_cars = random.randint(1, 3)
        for i in range(num_cars):
            # Space them out realistically
            dist = random.uniform(10.0, 50.0) + (i * 25.0) 
            _, sampled_params = sample_vehicle_params()
            speed = max(4.0, sampled_params.max_speed * random.uniform(0.55, 0.95))
            car = Vehicle(v_id=i, position=300.0 - dist, velocity=speed, params=sampled_params)
            self.vehicles.append(car)
            
        return self._get_state()

    def step(self, action_value):
        self.current_step += 1
        self.last_intervened = False
        self.last_shield_condition = ""
        self.last_success = False
        self.last_crash = False
        self.last_timeout = False
        
        # --- 1. PARSE CONTINUOUS ACTION ---
        # The Neural Network outputs a float between -1.0 and 1.0.
        # -1.0 means "Wait", 1.0 means "Push maximum".
        # We clip it to [0.0, 1.0] and scale it to our delta max of 0.10.
        pressure = np.clip(action_value, 0.0, 1.0)
        delta = pressure * 0.10
        delta = min(delta, self.bus.max_merge_progress_step)
        raw_delta = delta
        
        # --- 2. THE SAFETY SHIELD ---
        # We still prevent the AI from forcing physically impossible crashes during training
        nearest_gap, nearest_speed, _, nearest_width, rel_speed = self._scan_nearest_car()
        intervened = False
        adaptive_min_gap = 4.0 + max(0.0, nearest_width - 1.80) * 2.2
        emergency_gap = 0.50
        
        if pressure > 0 and nearest_gap > -5.0:
            if 0.0 <= nearest_gap < emergency_gap:
                if self.bus._merge_progress >= 0.30:
                    delta = self.bus.max_merge_progress_step
                    intervened = True
                    self.last_shield_condition = "emergency_forward"
                else:
                    delta = 0.0
                    intervened = True
                    self.last_shield_condition = "emergency_stop"
            elif self.bus._merge_progress > 0.65 and 0.0 <= nearest_gap < 1.20 and rel_speed > 0.2:
                delta = 0.0
                intervened = True
                self.last_shield_condition = "late_overlap"
            elif 0.0 <= nearest_gap < adaptive_min_gap and rel_speed > 1.2:
                risk = np.clip((adaptive_min_gap - nearest_gap) / max(adaptive_min_gap, 1e-3), 0.0, 1.0)
                delta *= (1.0 - 0.85 * risk)
                intervened = True
                self.last_shield_condition = "adaptive_gap"
            elif pressure > 0.65 and 0.0 <= nearest_gap < (adaptive_min_gap + 6.0) and rel_speed > 2.5:
                delta *= 0.35
                intervened = True
                self.last_shield_condition = "high_pressure"

        # If overlap already started late in merge, prioritize clearing forward.
        if nearest_gap < 0.0 and self.bus._merge_progress > 0.60:
            delta = max(delta, 0.5 * self.bus.max_merge_progress_step)

        self.last_intervened = intervened
        if intervened:
            self.episode_shields += 1
        intervention_strength = 0.0 if raw_delta <= 1e-8 else np.clip((raw_delta - delta) / raw_delta, 0.0, 1.0)

        # Apply movement
        old_x = self.bus.position
        self.bus._merge_progress = min(1.0, self.bus._merge_progress + delta)
        self.bus.position = self.bus.merge_start_x + (self.bus._merge_progress * self.bus.exit_taper_length)
        self.bus.velocity = max(0.0, (self.bus.position - old_x) / self.dt)

        # --- 3. PHYSICS TICK ---
        mock_leader = {
            'position': self.bus.position, 
            'velocity': self.bus.velocity,
            'length': self.bus.projected_length_along_x(), 
            'presence': self.bus._merge_progress 
        }
        
        for car in self.vehicles:
            # We assume the cars only care about the bus for this isolated training snippet
            car.update_physics(self.dt, leader=mock_leader, bus=self.bus)

        # --- 4. CALCULATE REWARDS ---
        # REWARD STRATEGY v2: Flow-Oriented
        # Goal: Reduce timeouts by incentivizing completion and reducing timidity.
        reward = -1.0 # Time penalty
        done = False
        
        gap, car_speed, car_accel, car_width, rel_speed = self._scan_nearest_car()
        progress = self.bus._merge_progress
        state = self._get_state()

        if intervened:
            # Penalty kept small so training remains stable while still discouraging
            # unnecessary pushes into dangerous gaps. The near-risk shaping term
            # (below) carries the main safety signal — this is just a directional nudge.
            reward -= (4.0 + 20.0 * (pressure ** 2)) * intervention_strength
            
        # --- THE CONDITIONAL CARROT ---
        # Only reward pushing (delta) when the gap is actually safe.
        # Safe = Plenty of room (> 5.0m) OR bus is matching/exceeding car speed.
        is_safe_to_merge = (gap > 5.0) or (gap > 0.5 and rel_speed > -0.5)

        if is_safe_to_merge:
            reward += (delta * 55.0)

        # Unconditional progress bonus: always give a small reward for being further along.
        # Without this, an agent at 80% merged in tight traffic earns NOTHING but takes
        # full near-risk penalties — teaching it that partial merges are bad and to never start.
        reward += progress * 8.0

        # Near-completion push: once 90% merged, give a strong incentive to finish.
        # This breaks the "stuck at 85%" local optimum.
        if progress > 0.9:
            reward += 20.0

        # Completion urgency: when mostly merged AND gap is clear, reward any push heavily.
        # This builds the training signal for the stuck-at-mid-progress + safe-gap state
        # that causes deterministic timeouts in eval.
        if progress > 0.5 and gap > 5.0:
            reward += delta * 40.0

        if progress > 0.1 and car_accel < -2.5 and gap > -10.0:
            reward -= abs(car_accel) * 3.0
        # Continuous near-risk shaping: penalize dangerous proximity REGARDLESS of pressure.
        # Root cause fix: removing '* pressure' so the agent learns WHEN to merge, not to move less.
        # Coefficient reduced from 140->70 to keep the same average magnitude (previously pressure ~0.5 avg).
        if 0.0 < gap < 12.0:
            gap_risk = (12.0 - gap) / 12.0
            speed_risk = np.clip((-rel_speed) / 8.0, 0.0, 1.0)
            merge_risk = np.clip((progress - 0.2) / 0.8, 0.0, 1.0)
            width_factor = 1.0 + (max(0.0, car_width - 1.8) * 0.7)
            reward -= 70.0 * gap_risk * (0.6 + 0.4 * speed_risk) * merge_risk * width_factor

        # Time-to-collision shaping: penalize risky closing maneuvers earlier.
        if gap > 0.0 and rel_speed > 0.1:
            ttc = gap / rel_speed
            if ttc < 4.0:
                ttc_risk = (4.0 - ttc) / 4.0
                reward -= 120.0 * ttc_risk * (0.4 + 0.6 * progress) * pressure

        if progress >= 1.0:
            reward += 2500.0
            # Shield-free bonus: agent found a clean gap without needing safety overrides
            if self.episode_shields == 0:
                reward += 200.0
            done = True
            self.last_success = True
        elif gap <= 0.0 and gap > -5.0 and progress >= 0.75:
            reward -= 4500.0
            done = True
            self.last_crash = True
        elif gap <= 0.0 and gap > -5.0 and progress >= 0.3:
            overlap = min(5.0, -gap) / 5.0
            reward -= 600.0 * ((progress - 0.3) / 0.5) * (0.5 + overlap)

        if self.current_step >= self.max_steps and not done:
            # Scaled up proportionally to the extra time budget (max_steps 300→500).
            # Without this increase the agent would learn it's OK to waste 50s.
            reward -= 2000.0
            done = True
            self.last_timeout = True

        return state, reward, done

    def _get_state(self):
        gap, nearest_speed, nearest_accel, nearest_width, relative_speed = self._scan_nearest_car()

        # PPO requires a clean, 32-bit float numpy array as the observation
        return self._encode_state(gap, nearest_speed, nearest_accel, self.bus._merge_progress, nearest_width, relative_speed)