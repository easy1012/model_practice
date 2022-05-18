
import numpy as np
import torch
import torch.nn as nn

class Critic_Model(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Critic_Model,self).__init__()
        self.state_dim = state_dim
        self.fc1 = nn.Sequential(nn.Linear(state_dim, 64),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(64, 32))
        self.fc3 = nn.Sequential(nn.Linear(64, 16),
                                 nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(16,1))
        self.fa = nn.Sequential(nn.Linear(action_dim,32))
    def forward(self, state,action):
        x1 = self.fc1(state)
        x1 = self.fc2(x1)
        a1 = self.fa(action)
        x = torch.concat([x1,a1],axis = -1)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

class Critic(object):
    def __init__(self, state_dim, action_dim,tau,learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.learning_rate = learning_rate

        self.model = Critic_Model(self.state_dim,self.action_dim)
        self.target_model = Critic_Model(self.state_dim,self.action_dim)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
        self.target_optimizer = torch.optim.Adam(self.target_model.parameters(),lr=self.learning_rate)


    def target_predict(self,state,action):
        return self.target_model(state,action)

    def update_target_network(self):
        for i in self.target_model.state_dict():
            tmp = self.model.get_parameter(i).data
            self.target_model.get_parameter(i).data = self.tau * tmp + (1-self.tau) * self.target_model.get_parameter(i).data

    def dq_da(self, states, s_actions):
        return self.model(states,s_actions)
    
    def train_on_batch(self,states,actions,td_targets):
        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions).float()
        td_targets = torch.from_numpy(td_targets).detach()
        predict = self.model(states,actions)
        loss = torch.mean((predict - td_targets) **2)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))