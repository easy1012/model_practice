import numpy as np
import torch
import torch.nn as nn

class Critic_Model(nn.Module):
    def __init__(self,state_dim):
        super(Critic_Model,self).__init__()
        self.state_dim = state_dim
        self.fc1 = nn.Sequential(nn.Linear(state_dim, 64),
                                 nn.LeakyReLU())
        self.fc2 = nn.Sequential(nn.Linear(64, 32),
                                 nn.LeakyReLU())
        self.fc3 = nn.Sequential(nn.Linear(32, 16),
                                 nn.LeakyReLU())
        self.fc4 = nn.Sequential(nn.Linear(16,1))
    def forward(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

class Critic(object):
    def __init__(self, state_dim, action_dim, learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        self.model = Critic_Model(self.state_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)

    def train_on_batch(self,states,td_targets):
        td_targets = torch.from_numpy(td_targets).detach()
        predict = self.model(states)
        loss = torch.mean((predict - td_targets) **2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))
