import torch
import torch.nn as nn
from env_wrapper import BusMergeEnv
from ppo_agent import PPOAgent

def train():
    env = BusMergeEnv()
    agent = PPOAgent()
    # --- ADD THIS LINE TO START FROM YOUR EXISTING BRAIN ---
    agent.load()
    epochs = 500
    batch_size = 2000
    
    print("Starting PPO PyTorch Training with Analytics...")
    
    for epoch in range(epochs):
        state = env.reset()
        epoch_reward = 0
        steps = 0
        
        # --- NEW: Trackers for this batch ---
        successes, crashes, timeouts, shield_blocks = 0, 0, 0, 0
        episodes_completed = 0
        
        while steps < batch_size:
            action_value, action_logprob = agent.choose_action(state)
            next_state, reward, done = env.step(action_value)
            
            # --- NEW: Track exact events based on the reward logic in env_wrapper ---
            if reward <= -1000 and not done:
                shield_blocks += 1
            if done:
                episodes_completed += 1
                if reward >= 500: # Success (+1000)
                    successes += 1
                elif reward <= -1500: # Crash (-2000)
                    crashes += 1
                elif steps >= env.max_steps: # Timeout (-500)
                    timeouts += 1
            
            agent.store_transition(state, action_value, action_logprob, reward, next_state, done)
            
            state = next_state
            epoch_reward += reward
            steps += 1
            
            if done:
                state = env.reset()
                
        # --- PHASE 2: PPO UPDATE (Unchanged) ---
        states = torch.FloatTensor([m[0] for m in agent.memory])
        actions = torch.FloatTensor([m[1] for m in agent.memory]).unsqueeze(1)
        old_logprobs = torch.FloatTensor([m[2] for m in agent.memory]).unsqueeze(1)
        rewards = [m[3] for m in agent.memory]
        dones = [m[5] for m in agent.memory]
        
        discounted_rewards = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d: R = 0
            R = r + (agent.gamma * R)
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.FloatTensor(discounted_rewards).unsqueeze(1)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)
        
        with torch.no_grad():
            _, _, state_values = agent.policy(states)
            advantages = discounted_rewards - state_values

        for _ in range(4):
            action_mean, action_std, current_state_values = agent.policy(states)
            dist = torch.distributions.Normal(action_mean, action_std)
            new_logprobs = dist.log_prob(actions)
            
            ratios = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - agent.clip_ratio, 1 + agent.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(current_state_values, discounted_rewards)
            
            loss = actor_loss + 0.5 * critic_loss
            
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
            
        agent.memory.clear()
        
        # --- PHASE 3: NEW DETAILED LOGGING ---
        if (epoch + 1) % 10 == 0:
            avg_score = epoch_reward / episodes_completed if episodes_completed > 0 else 0
            
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            print(f"Stats  | Episodes: {episodes_completed} | Succ: {successes} | Crash: {crashes} | TO: {timeouts} | Shield: {shield_blocks}")
            print(f"Scores | Avg Score: {avg_score:.1f} | Actor Loss: {actor_loss.item():.3f} | Critic Loss: {critic_loss.item():.3f}")
            agent.save()

if __name__ == "__main__":
    train()