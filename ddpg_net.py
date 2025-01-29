import torch.nn as nn
import torch

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()

        self.conv1=nn.Conv2d(state_dim,32,8, stride=4)
        self.conv2=nn.Conv2d(32,64,4, stride=2)
        self.conv3=nn.Conv2d(64,64,3, stride=1)

        self.fc1 = nn.Linear(7*7*64, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self,state):
        x=self.conv1(state)
        x=self.conv2(x)
        x=self.conv3(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.fc2(x)
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Critic, self).__init__()

        self.conv1=nn.Conv2d(state_dim,32,8, stride=4)
        self.conv2=nn.Conv2d(32,64,4, stride=2)
        self.conv3=nn.Conv2d(64,64,3, stride=1)

        self.fc1 = nn.Linear(7*7*64, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim+action_dim, 1)
    
    def forward(self,state,action):
        x=self.conv1(state)
        x=self.conv2(x)
        x=self.conv3(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=torch.relu(x)
        x=torch.cat([x,action],1)
        x=self.fc2(x)
        return x