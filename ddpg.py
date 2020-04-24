import numpy as np
import random
import copy
from collections import namedtuple, deque

import model
import ounoise
import replay_buffer

import torch
import torch.nn.functional as F
import torch.optim as optim

class HyperParameters:
    def __init__(self):
        pass
    
hp = HyperParameters()

hp.state_size = 24
hp.action_size = 2
hp.random_seed = 222
hp.buffer_size = int(1e5)  # replay buffer size
hp.batch_size = 128        # minibatch size
hp.gamma = 0.99            # discount factor
hp.tau = 1e-3              # for soft update of target parameters
hp.lr_actor = 2e-4         # learning rate of the actor # ADDED
hp.lr_critic = 2e-4        # learning rate of the critic # ADDED
hp.weight_decay = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, hp):
        """Initialize an Agent object.
        
        Params
        ======
            hp: hyper parameters
        """
        self.hp = hp

        # Actor Network (w/ Target Network)
        self.actor_local = model.Actor(self.hp.state_size, self.hp.action_size, self.hp.random_seed).to(device)
        self.actor_target = model.Actor(self.hp.state_size, self.hp.action_size, self.hp.random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.hp.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = model.Critic(self.hp.state_size, self.hp.action_size, self.hp.random_seed).to(device)
        self.critic_target = model.Critic(self.hp.state_size, self.hp.action_size, self.hp.random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.hp.lr_critic, weight_decay=self.hp.weight_decay)
        self.soft_update(self.critic_local, self.critic_target, 1)
        self.soft_update(self.actor_local, self.actor_target, 1)   

        # Noise process
        self.noise = ounoise.OUNoise(self.hp.action_size, self.hp.random_seed)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.hp.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) # ADDED
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.hp.tau)
        self.soft_update(self.actor_local, self.actor_target, self.hp.tau)   
        
        return critic_loss.cpu().data.numpy(), actor_loss.cpu().data.numpy()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
