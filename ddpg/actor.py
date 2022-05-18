import numpy as np
import torch.nn as nn
import torch
from torchvision.transforms import Lambda
import torch.nn.utils as torch_utils

class Actor_Model(nn.Module):
    def __init__(self,state_dim,action_dim,action_bound):
        super(Actor_Model,self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(state_dim, 64),
                                    nn.LeakyReLU())
        self.fc2 = nn.Sequential(nn.Linear(64, 32),
                                    nn.LeakyReLU())
        self.fc3 = nn.Sequential(nn.Linear(32, 16),
                                    nn.LeakyReLU())
        self.fc4 = nn.Sequential(nn.Linear(16, action_dim),
                                    nn.Tanh())
        self.action_bound = action_bound
    def forward(self,state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = torch.clamp(x,-self.action_bound,self.action_bound)
        return x

class Actor(object):
    def __init__(self, state_dim, action_dim, action_bound,tau,learning_rate):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.tau = tau
        self.learning_rate = learning_rate


        self.model = Actor_Model(self.state_dim,self.action_dim,self.action_bound)
        self.target_model = Actor_Model(self.state_dim,self.action_dim,self.action_bound)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = self.learning_rate)


    def get_policy_action(self, state):
        mu_a, std_a = self.model(state)

        mu_a = mu_a.detach().numpy()
        std_a = std_a.detach().numpy()
        std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu_a, std_a, size=(1 ,self.action_dim))
        return mu_a,std_a,action

    def predict(self, state):
        return self.model(state)

    def target_predict(self,state):
        return self.target_model(state)

    def update_target_network(self):
        for i in self.target_model.state_dict():
            tmp = self.model.get_parameter(i).data
            self.target_model.get_parameter(i).data = self.tau * tmp + (1-self.tau) * self.target_model.get_parameter(i).data

    def Learn(self,states):
        states = -states.mean()
        
        self.optimizer.zero_grad()
        states.backward()
        self.optimizer.step()


    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))