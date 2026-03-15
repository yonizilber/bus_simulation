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
        # 4 Inputs -> 64 Neurons -> 64 Neurons -> 1 Output (Gas Pedal)
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(), # Tanh is a math function that smooths the signal
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Sigmoid() # Sigmoid forces the final output to be strictly between 0.0 and 1.0
        )
        
        # The Actor's uncertainty (How wide the Bell Curve is)
        # We start it as a learnable parameter that can shrink over time
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # THE CRITIC (The Instructor)
        # 4 Inputs -> 64 Neurons -> 64 Neurons -> 1 Output (Predicted Score)
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
        action_std = torch.exp(action_log_std)
        
        # 2. Ask the Critic for the predicted score
        state_value = self.critic(state)
        
        return action_mean, action_std, state_value


# --- 2. THE PPO MATH ENGINE ---
class PPOAgent:
    def __init__(self, state_dim=4, action_dim=1, filename="checkpoints/ppo_bus_brain.pth"):
        self.filename = filename
        
        # Hyperparameters (The dials of the AI)
        self.gamma = 0.99       # How much it cares about the future vs immediate points
        self.clip_ratio = 0.2   # Forbid learning steps larger than 20%
        self.lr = 3e-4          # The learning rate (how fast PyTorch tweaks the neurons)
        
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
        
        # Get the mathematical probability of that specific choice (needed for PPO update later)
        action_logprob = dist.log_prob(action)
        
        return action.item(), action_logprob.item()

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
            self.policy.load_state_dict(torch.load(self.filename))
            print("Successfully loaded pre-trained PPO brain!")