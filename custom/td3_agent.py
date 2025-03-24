import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
import numpy as np
import os 

from td3_net import Actor, Critic

class Agent_TD3():
    def __init__(self, action_dim, max_action, low_action, c, buffer_size=None, batch_size=None, lr_actor=None, lr_critic=None, tau=None, discount=None, update_interval=2):
       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(c,  action_dim, max_action).to(self.device)
        self.critic_1 = Critic(c, action_dim).to(self.device)
        self.critic_2 = Critic(c, action_dim).to(self.device) 

        self.max_action=max_action
        self.low_action=low_action

        #set parameters for DNN 
        self.tau=tau
        self.epsilon=1.0
        self.epsilon_decay=1e-6
        self.lr_actor=lr_actor
        self.lr_critic=lr_critic
        self.batch_size=batch_size
        self.discount=discount
        self.update_interval=update_interval

        #noise function for exploration
        self.trainMode=True

        #initialize target networks
        self.target_actor = Actor(c, action_dim, max_action).to(self.device)
        self.target_critic_1 = Critic(c, action_dim).to(self.device)
        self.target_critic_2 = Critic(c, action_dim).to(self.device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), lr=lr_critic)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(), lr=lr_critic)
        self.loss_fn = nn.MSELoss()

        self.memory = ReplayBuffer(storage=LazyTensorStorage(buffer_size), batch_size=batch_size)

    def update(self, tau=None):
        if tau ==None:
            tau=self.tau
        
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)
            
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)
        

    def select_action(self, state):
        self.evalMode()
        state=np.array([state])
        state = torch.FloatTensor(state).to(self.device)
        predicted_action = self.actor(state).cpu().detach().numpy().flatten()
        
        #if self.trainMode:
            #noise=self.ou.sample()*self.epsilon
            #noise==[noise [0],noise[0]]
            #predicted_action+=noise
        self.trainM()
        return np.clip(predicted_action,self.low_action,self.max_action)

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
    
    def evalMode(self):
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        self.trainMode=False

    def trainM(self):
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        self.trainMode=True

    def train(self, iteration):
        mean_actorLoss=0
        mean_criticLoss=0
        
        for i in range(iteration):  
            sample = self.memory.sample().to(self.device)
            states = sample['state']
            new_states = sample['new_state']
            actions = sample['action']
            rewards = sample['reward']
            dones = sample['done']


            target_actions = self.target_actor.forward(new_states)
            #add target noise too smooth 
            noise=torch.clamp(torch.normal(0, 0.2, size=(1,2)),-0.5,0.5).to(self.device)

            target_actions = target_actions+ noise


            target_q1 = self.target_critic_1.forward(new_states, target_actions)
            target_q2 = self.target_critic_2.forward(new_states, target_actions)

            q1=self.critic_1.forward(states, actions)
            q2=self.critic_2.forward(states, actions)

            target_critics = torch.min(target_q1, target_q2)
            target_q=rewards +(1-dones)*self.discount*target_critics
                
            self.critic_optimizer_1.zero_grad()
            self.critic_optimizer_2.zero_grad()

            critic_loss_1=F.mse_loss(target_q.detach(), q1)
            critic_loss_2=F.mse_loss(target_q.detach(), q2 )
            critic_loss=critic_loss_1+critic_loss_2

            critic_loss.backward()
            self.critic_optimizer_1.step()
            self.critic_optimizer_2.step()

            if(i%self.update_interval==0):
                self.actor_optimizer.zero_grad()

                new_actions=self.actor(states)
                actor_loss=-self.critic_1.forward(states, new_actions).mean()

                actor_loss.backward()
                self.actor_optimizer.step()

                self.update()

                mean_actorLoss+=actor_loss

                mean_criticLoss+=critic_loss

            return (mean_actorLoss/iteration), (mean_criticLoss/iteration)
        
        return [mean_criticLoss]
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.actor.state_dict(), f"{path}/actor.pth")
        torch.save(self.critic_1.state_dict(), f"{path}/critic_1.pth")
        torch.save(self.critic_2.state_dict(), f"{path}/critic_2.pth")
    
    def load(self,path):
        self.actor.load_state_dict(torch.load(f"{path}/actor.pth", map_location=self.device))
        self.critic_1.load_state_dict(torch.load(f"{path}/critic_1.pth", map_location=self.device))
        self.critic_1.load_state_dict(torch.load(f"{path}/critic_2.pth", map_location=self.device))
    
    
