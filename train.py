import random
import os
from vehicle import Vehicle, Bus, BusState, CAR_PARAMS
from agent import QLearningAgent

def run_training(episodes=10000):
    agent = QLearningAgent()
    dt = 0.1
    max_steps = 300 # INCREASED: 30 seconds max
    
    successes, crashes, timeouts, shield_blocks = 0, 0, 0, 0
    recent_scores = []
    
    breakdown = {
        'Success(+)': 0, 'Time(-)': 0, 'Shield(-)': 0, 
        'Brake(-)': 0, 'Danger(-)': 0, 'Crash/TO(-)': 0
    }

    print(f"Starting Clean Training for {episodes} episodes...")

    for episode in range(episodes):
        bus = Bus(v_id=999, position=100.0, velocity=0.0, params=CAR_PARAMS)
        bus.state = BusState.WAITING_TO_MERGE
        bus._merge_progress = 0.0
        
        start_dist = random.uniform(10.0, 70.0)
        start_speed = random.uniform(5.0, 15.0)
        car = Vehicle(v_id=1, position=100.0 - start_dist, velocity=start_speed, params=CAR_PARAMS)

        total_reward = 0
        
        # Initial gap calculation
        gap = bus.position - bus.params.length - car.position
        state = agent.get_state(gap, car.velocity, car.acceleration, bus._merge_progress)

        for step in range(max_steps):
            intended_action = agent.choose_action(state)
            approved_action = intended_action
            
            # Recalculate gap for the current frame
            gap = bus.position - bus.params.length - car.position
            intervened = False
            
            # B. THE SMARTER SAFETY SHIELD
            # We only shield if the car is behind us or alongside us (gap > -5.0)
            if intended_action > 0 and gap > -5.0: 
                if gap < 5.0:
                    approved_action = 0
                    intervened = True
                elif intended_action == 2 and gap < 15.0 and car.velocity > 10.0:
                    approved_action = 0
                    intervened = True

            # C. Apply Action
            if approved_action == 0: delta = 0.0       
            elif approved_action == 1: delta = 0.05    
            elif approved_action == 2: delta = 0.10    
            
            bus._merge_progress = min(1.0, bus._merge_progress + delta)
            
            mock_leader = {
                'position': bus.position, 'velocity': bus.velocity,
                'length': bus.params.length, 'presence': bus._merge_progress 
            }
            car.update_physics(dt, leader=mock_leader, bus=bus)

            # D. Evaluate
            reward = -1  
            breakdown['Time(-)'] -= 1
            done = False
            gap = bus.position - bus.params.length - car.position

            if intervened:
                reward -= 1000
                shield_blocks += 1
                breakdown['Shield(-)'] -= 1000

            # Only penalize for spilled coffee if the car is actually behind the bus
            if bus._merge_progress > 0.1 and car.acceleration < -2.5 and gap > -10.0:
                brake_pen = abs(car.acceleration) * 5.0
                reward -= brake_pen
                breakdown['Brake(-)'] -= brake_pen

            if bus._merge_progress >= 1.0:
                reward += 1000
                successes += 1
                breakdown['Success(+)'] += 1000
                done = True
                
            elif gap <= 0.0 and gap > -5.0 and bus._merge_progress >= 0.8:
                reward -= 2000 
                crashes += 1
                breakdown['Crash/TO(-)'] -= 2000
                done = True
                
            elif gap <= 0.0 and gap > -5.0 and bus._merge_progress >= 0.3:
                danger_pen = 100.0 * ((bus._merge_progress - 0.3) / 0.5)
                reward -= danger_pen
                breakdown['Danger(-)'] -= danger_pen
            
            if step == max_steps - 1 and not done:
                reward -= 500 
                timeouts += 1
                breakdown['Crash/TO(-)'] -= 500
                done = True

            # E. Learn
            next_state = agent.get_state(gap, car.velocity, car.acceleration, bus._merge_progress)
            agent.learn(state, intended_action, reward, next_state)
            
            state = next_state
            total_reward += reward

            if done:
                break

        recent_scores.append(total_reward)
        agent.decay_exploration()

        if (episode + 1) % 500 == 0:
            avg_score = sum(recent_scores) / len(recent_scores)
            
            print(f"\n--- Episode {episode + 1:^5} ---")
            print(f"Stats  | Succ: {successes} | Crash: {crashes} | TO: {timeouts} | Shield: {shield_blocks}")
            print(f"Scores | Avg: {avg_score:.1f} | Eps: {agent.epsilon:.3f}")
            print(f"Points | " + " | ".join([f"{k}: {v/500:.0f}" for k, v in breakdown.items()]))
            
            successes, crashes, timeouts, shield_blocks = 0, 0, 0, 0
            recent_scores = []
            breakdown = {k: 0 for k in breakdown}

    agent.save()
    print("\nTraining complete! Brain saved.")

if __name__ == "__main__":
    run_training(episodes=10000)