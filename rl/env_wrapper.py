import numpy as np
import random
import sys
import os

# Ensure we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vehicle import Vehicle, Bus, BusState, CAR_PARAMS

class BusMergeEnv:
    def __init__(self):
        self.dt = 0.1
        self.max_steps = 300
        
        # PyTorch Network Dimensions
        self.state_size = 4  # [Gap, Speed, Accel, Merge_Progress]
        self.action_size = 1 # [Merge_Pressure] -> Continuous float from -1.0 to 1.0
        
        self.bus = None
        self.vehicles = []
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        
        # 1. Spawn Bus directly in the bay (skip the driving phase)
        self.bus = Bus(v_id=999, position=300.0, velocity=0.0, params=CAR_PARAMS)
        self.bus.state = BusState.WAITING_TO_MERGE
        self.bus._merge_progress = 0.0
        
        # 2. Spawn a random stream of 1 to 3 cars approaching
        self.vehicles = []
        num_cars = random.randint(1, 3)
        for i in range(num_cars):
            # Space them out realistically
            dist = random.uniform(10.0, 50.0) + (i * 25.0) 
            speed = random.uniform(5.0, 15.0)
            car = Vehicle(v_id=i, position=300.0 - dist, velocity=speed, params=CAR_PARAMS)
            self.vehicles.append(car)
            
        return self._get_state()

    def step(self, action_value):
        self.current_step += 1
        
        # --- 1. PARSE CONTINUOUS ACTION ---
        # The Neural Network outputs a float between -1.0 and 1.0.
        # -1.0 means "Wait", 1.0 means "Push maximum".
        # We clip it to [0.0, 1.0] and scale it to our delta max of 0.10.
        pressure = np.clip(action_value, 0.0, 1.0)
        delta = pressure * 0.10 
        
        # --- 2. THE SAFETY SHIELD ---
        # We still prevent the AI from forcing physically impossible crashes during training
        nearest_gap = self._get_state()[0]
        nearest_speed = self._get_state()[1]
        intervened = False
        
        if pressure > 0 and nearest_gap > -5.0:
            if nearest_gap < 5.0 and nearest_speed > 2.0:
                delta = 0.0
                intervened = True
            elif pressure > 0.5 and nearest_gap < 15.0 and nearest_speed > 10.0:
                delta = 0.0
                intervened = True

        # Apply movement
        self.bus._merge_progress = min(1.0, self.bus._merge_progress + delta)

        # --- 3. PHYSICS TICK ---
        mock_leader = {
            'position': self.bus.position, 
            'velocity': self.bus.velocity,
            'length': self.bus.params.length, 
            'presence': self.bus._merge_progress 
        }
        
        for car in self.vehicles:
            # We assume the cars only care about the bus for this isolated training snippet
            car.update_physics(self.dt, leader=mock_leader, bus=self.bus)

        # --- 4. CALCULATE REWARDS ---
        reward = -1.0 # Time penalty
        done = False
        
        state = self._get_state()
        gap, car_speed, car_accel, progress = state[0], state[1], state[2], state[3]

        if intervened:
            reward -= 5000.0

        if progress > 0.1 and car_accel < -2.5 and gap > -10.0:
            reward -= abs(car_accel) * 5.0

        if progress >= 1.0:
            reward += 1000.0
            done = True
        elif gap <= 0.0 and gap > -5.0 and progress >= 0.8:
            reward -= 10000.0
            done = True
        elif gap <= 0.0 and gap > -5.0 and progress >= 0.3:
            reward -= 100.0 * ((progress - 0.3) / 0.5)

        if self.current_step >= self.max_steps and not done:
            reward -= 500.0
            done = True

        return state, reward, done

    def _get_state(self):
        # Scan for the most relevant car
        gap = 999.0
        nearest_speed = 0.0
        nearest_accel = 0.0
        
        for v in self.vehicles:
            current_gap = self.bus.position - self.bus.params.length - v.position
            if -10 < current_gap < 60:
                if abs(current_gap) < abs(gap):
                    gap = current_gap
                    nearest_speed = v.velocity
                    nearest_accel = v.acceleration
                    
        # PPO requires a clean, 32-bit float numpy array as the observation
        return np.array([gap, nearest_speed, nearest_accel, self.bus._merge_progress], dtype=np.float32)