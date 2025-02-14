import torch
import torch.nn as nn
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.objectives import SoftUpdate, DDPGLoss
from tensordict import TensorDict
import numpy as np

from ddpg_net import Actor, Critic

class Agent():
    def __init__(self, action_dim, c, hidden_dim, buffer_size, batch_size, lr_actor, lr_critic, gamma, tau):
        self.tau=tau
        self.gamma=gamma    
        self.lr_actor=lr_actor
        self.lr_critic=lr_critic
        self.c=c
        self.batch_size=batch_size 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(c, hidden_dim, action_dim).to(self.device)
        self.critic = Critic(c,hidden_dim, action_dim).to(self.device)

        self.target_actor = Actor(c, hidden_dim, action_dim).to(self.device)
        self.target_critic = Critic(c, hidden_dim, action_dim).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.loss_fn = nn.MSELoss()

        self.memory = ReplayBuffer(storage=LazyTensorStorage(buffer_size), batch_size=batch_size)

    def update(self, tau=None):
        if tau ==None:
            tau=self.tau
        
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)
            
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)
        

    def select_action(self, state):
        state=np.array([state])
        state = torch.FloatTensor(state).to(self.device)
        predicted_action = self.actor(state).cpu().detach().numpy().flatten()

        return predicted_action

    def storeStep(self, state, new_state, action, reward, done):
        data=TensorDict(
            {
                'state': torch.FloatTensor(state).to(self.device),
                'new_state': torch.FloatTensor(new_state).to(self.device),
                'action': torch.FloatTensor(action).to(self.device),
                'reward': torch.FloatTensor([reward]).to(self.device),
                'done': torch.FloatTensor([done]).to(self.device)
            },
            batch_size=[]
        )
        self.memory.add(data)

    def sample(self, batch_size=None):
        if batch_size is None:
            return self.memory.sample()
        else:
            return self.memory.sample(batch_size)
    
    def train(self):
        sample = self.memory.sample()
        states = sample['state'].to(self.device)
        new_states = sample['new_state'].to(self.device)
        actions = sample['action'].to(self.device)
        rewards = sample['reward'].to(self.device)
        dones = sample['done'].to(self.device)


        target_actions = self.target_actor(new_states)
        target_critics = self.target_critic(new_states, target_actions)
        critic_v=self.critic(states, actions)
        target_q=rewards + (1-dones)*self.gamma*target_critics
        critic_loss=self.loss_fn(critic_v, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions=self.actor(states)
        actor_loss=-self.critic(states, new_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update()


        return actor_loss, critic_loss
    
    