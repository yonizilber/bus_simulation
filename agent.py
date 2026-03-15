import numpy as np
import pickle
import os
import random

class QLearningAgent:
    def __init__(self, filename="bus_brain.pkl"):
        self.filename = filename
        self.epsilon = 1.0       
        self.epsilon_min = 0.01  
        self.epsilon_decay = 0.999 
        self.alpha = 0.1         
        self.gamma = 0.95        
        
        self.q_table = {}        
        self.load()

    def get_state(self, gap, car_speed, car_accel, bus_progress):
        # 1. The Ghost-Free Gap
        if gap < -5.0: d_bin = "PASSED"         # Car is safely in front of us
        elif gap < 5.0: d_bin = "CRASH_ZONE"    # Car is physically alongside
        elif gap < 12.0: d_bin = "DANGER"
        elif gap < 25.0: d_bin = "CLOSE"
        elif gap < 45.0: d_bin = "APPROACH"
        else: d_bin = "FAR"

        # 2. Car Speed
        if car_speed < 2.0: s_bin = "STOPPED"
        elif car_speed < 8.0: s_bin = "SLOW"
        elif car_speed < 12.0: s_bin = "MEDIUM" 
        else: s_bin = "FAST"
            
        # 3. Car Acceleration
        if car_accel < -1.5: a_bin = "HARD_BRAKING"
        elif car_accel < -0.2: a_bin = "YIELDING"
        else: a_bin = "CRUISING"

        # 4. Bus Progress
        if bus_progress < 0.1: p_bin = "START"
        elif bus_progress < 0.5: p_bin = "POKING_OUT"
        elif bus_progress < 0.85: p_bin = "BLOCKING"
        else: p_bin = "COMMIT"

        return (d_bin, s_bin, a_bin, p_bin)

    def choose_action(self, state):
        # Initialize new states on the fly
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0] 

        # Epsilon-Greedy: Explore vs Exploit
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2])
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table: self.q_table[state] = [0.0, 0.0, 0.0]
        if next_state not in self.q_table: self.q_table[next_state] = [0.0, 0.0, 0.0]

        # Standard Q-Learning update
        current_q = self.q_table[state][action]
        max_future_q = np.max(self.q_table[next_state])
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][action] = new_q

    def decay_exploration(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self):
        with open(self.filename, 'wb') as f:
            pickle.dump({'q_table': self.q_table, 'epsilon': self.epsilon}, f)

    def load(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data.get('q_table', {})
                self.epsilon = data.get('epsilon', 1.0)