import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=1e-4):
        super(Actor, self).__init__()
        self.conv1=nn.Conv2d(input_dim, hidden_dim,8, stride=4)
        self.conv2=nn.Conv2d(hidden_dim, hidden_dim, kernel_size=8, stride=4)

        self.fc1 = nn.Linear(400*28*38, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.tanh = nn.Tanh()

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x=x.view(-1,400*28*38)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate=1e-4):
        super(Critic, self).__init__()
        self.conv1=nn.Conv2d(state_dim, hidden_dim,8, stride=4)
        self.conv2=nn.Conv2d(hidden_dim, hidden_dim,8, stride=4)
        self.conv3=nn.Conv2d(hidden_dim, hidden_dim,8, stride=4)

        self.fc1 = nn.Linear(400*6*8, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state, action):
        x= torch.relu(self.conv1(state))
        x= torch.relu(self.conv2(x))
        x= torch.relu(self.conv3(x))
        print(x.shape)
        x=x.view(-1,400*6*8)
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(torch.cat([x, action], dim=1)))
        x = self.fc3(x)
        return x