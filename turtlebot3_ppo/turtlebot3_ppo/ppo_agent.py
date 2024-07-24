import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

# build the actor and critic deep networks
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.actor_mean = nn.Linear(128, output_dim)
        self.actor_log_std = nn.Linear(128, output_dim)  # Log std for numerical stability
        self.critic = nn.Linear(128, 1)

    def forward(self, x): # x is state. Cannot input any inf or nan values
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_mean = self.actor_mean(x)
        action_log_std = self.actor_log_std(x)
        action_std = torch.exp(action_log_std)
        state_value = self.critic(x)
        return action_mean, action_std, state_value

class PPOAgent:
    def __init__(self, env, learning_rate=1e-4, gamma=0.99, epsilon=0.2, k_epochs=10): # adjust training hyperparameters here
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.k_epochs = k_epochs
        
        self.actor_critic = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        self.policy_old = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
        self.policy_old.load_state_dict(self.actor_critic.state_dict())
    
    # Select action of robot based on the current policy and state (sensor readings). Initializes as random, but learns to make correct decisions.
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_mean, action_std, _ = self.policy_old(state)
        action_dist = torch.distributions.Normal(action_mean, action_std)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)
        return action.numpy(), action_log_prob.numpy()

    def train(self, memory):
        states, actions, rewards, dones, log_probs_old = zip(*memory)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        log_probs_old = torch.FloatTensor(log_probs_old)
        
        # Calculate discounted rewards
        discounted_rewards = []
        for t in range(len(rewards)):
            G = 0
            pw = 0
            for r in rewards[t:]:
                G = G + self.gamma ** pw * r
                pw = pw + 1
            discounted_rewards.append(G)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        
        for _ in range(self.k_epochs):
            action_means, action_stds, state_values = self.actor_critic(states)
            action_dists = Normal(action_means, action_stds)
            log_probs = action_dists.log_prob(actions).sum(axis=1)
            
            ratios = torch.exp(log_probs - log_probs_old)
            advantages = discounted_rewards - state_values.squeeze()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            loss = -torch.min(surr1, surr2).mean() + 0.5 * (discounted_rewards - state_values.squeeze()).pow(2).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.policy_old.load_state_dict(self.actor_critic.state_dict())
