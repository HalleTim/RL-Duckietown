import torch
import torch.nn as nn
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.objectives import SoftUpdate, DDPGLoss
from tensordict import TensorDict
import numpy as np

from ddpg_net import Actor, Critic

class Agent():
    def __init__(self, action_dim, state_dim, hidden_dim, buffer_size, batch_size, lr_actor, lr_critic, gamma, tau):
        self.tau=tau
        self.gamma=gamma    
        self.lr_actor=lr_actor
        self.lr_critic=lr_critic

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim,hidden_dim, action_dim).to(self.device)

        self.target_actor = Actor().to(self.device)
        self.target_critic = Critic().to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.critic_loss = nn.MSELoss()
        self.memory = ReplayBuffer(buffer_size=LazyTensorStorage(buffer_size), batch_size=batch_size)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        return self.actor(state)

    def storeStep(self, state, action, reward, next_state, done):
        data=TensorDict(
            {
                'observation': state,
                "action": torch.tensor(action, dtype=torch.float32),
                "reward": torch.tensor(reward, dtype=torch.float32),
                "next_state": torch.from_numpy(next_state),
            }
        )

        self.memory.extend(data)

    def sample(self, batch_size=None):
        if batch_size is None:
            return self.memory.sample()
        else:
            return self.memory.sample(batch_size)