import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os

# --- 1. THE NEURAL NETWORKS ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # THE ACTOR (The Driver)
        # state_dim Inputs -> 64 Neurons -> 64 Neurons -> 2 Outputs (steer_intent, gas_intent)
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # Tanh maps to [-1, 1] for steer and gas
        )
        
        # The Actor's uncertainty (How wide the Bell Curve is)
        # We start it as a learnable parameter that can shrink over time
        self.actor_log_std = nn.Parameter(torch.full((1, action_dim), -0.7))
        
        # THE CRITIC (The Instructor)
        # state_dim Inputs -> 64 Neurons -> 64 Neurons -> 1 Output (Predicted Score)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        # 1. Ask the Actor for the target gas pressure (mean) and uncertainty (std)
        action_mean = self.actor_mean(state)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        # Clamp std to a minimum of 0.05 to prevent exploration collapse.
        # Without this, the agent can become fully deterministic and get stuck.
        action_std = torch.exp(action_log_std).clamp(min=0.05)
        
        # 2. Ask the Critic for the predicted score
        state_value = self.critic(state)
        
        return action_mean, action_std, state_value


# --- 2. THE PPO MATH ENGINE ---
class PPOAgent:
    def __init__(self, state_dim=11, action_dim=2, filename="checkpoints/ppo_bus_brain.pth"):
        self.filename = filename
        
        # Hyperparameters (The dials of the AI)
        self.gamma = 0.99       # How much it cares about the future vs immediate points
        self.clip_ratio = 0.2   # Forbid learning steps larger than 20%
        self.lr = 5e-5          # Lowered from 1e-4: finer steps for fine-tuning near-optimal policy
        
        # Initialize the Brain
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # Memory storage for a single "batch" of training
        self.memory = []

    def choose_action(self, state):
        # Convert the standard numpy array into a PyTorch Tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # We don't want PyTorch calculating calculus while we are just driving, so we use no_grad()
        with torch.no_grad():
            action_mean, action_std, _ = self.policy(state_tensor)
            
        # Create the Bell Curve (Normal Distribution)
        dist = Normal(action_mean, action_std)
        
        # Sample a point from the Bell Curve
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        # action[0] = steer_intent, action[1] = gas_intent (both in [-1, 1])

        # Get the mathematical probability of that specific choice (needed for PPO update later)
        action_logprob = dist.log_prob(action)

        # Return as numpy array (shape [2]) and summed logprob scalar
        return action.squeeze(0).numpy(), action_logprob.sum().item()

    def store_transition(self, state, action, action_logprob, reward, next_state, done):
        # Save this exact 0.1-second memory so we can learn from it later
        self.memory.append((state, action, action_logprob, reward, next_state, done))

    def update(self):
        # (For the sake of keeping this file clean and readable for your first PyTorch dive, 
        # the complex PPO Advantage calculus goes here. We will call this from our training loop!)
        pass

    def save(self):
        # Ensure the checkpoints folder exists
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        # Save the exact neural weights to the hard drive
        torch.save(self.policy.state_dict(), self.filename)

    def load(self):
        if os.path.exists(self.filename):
            # Load the neural weights back into the network
            try:
                state_dict = torch.load(self.filename, weights_only=True)
            except TypeError:
                state_dict = torch.load(self.filename)
            try:
                self.policy.load_state_dict(state_dict)
                print("Successfully loaded pre-trained PPO brain!")
            except RuntimeError as e:
                print(f"Checkpoint shape mismatch, starting with fresh weights: {e}")