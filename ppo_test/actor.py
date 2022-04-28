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
        self.fc5 = nn.Sequential(nn.Linear(16, action_dim),
                                    nn.Softplus())
        self.action_bound = action_bound
    def forward(self,state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        out_mu = self.fc4(x)
        std_out = self.fc5(x)
        mu_out = out_mu * self.action_bound
        return mu_out,std_out

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, learning_rate,ratio_clipping):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = torch.tensor(action_bound)
        self.learning_rate = learning_rate
        self.ratio_clipping = ratio_clipping

        self.std_bound = [1e-2, 1.0]

        self.model = Actor_Model(self.state_dim,self.action_dim,self.action_bound)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = self.learning_rate)

    def log_pdf(self, mu, std, action):
        std = std.clamp(self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = - 0.5 * (((action - mu) ** 2 / var) + (torch.log(var * 2 * np.pi)))
        return torch.sum(log_policy_pdf, dim=1, keepdim=True)

    def get_policy_action(self, state):
        mu_a, std_a = self.model(state)

        mu_a = mu_a.detach().numpy()
        std_a = std_a.detach().numpy()
        std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu_a, std_a, size=(1 ,self.action_dim))
        return mu_a,std_a,action

    def predict(self, state):
        mu_a, std_a = self.model(state)
        return mu_a

    def Learn(self,old_policy_pdf, states, actions, advantages):
        actions = torch.FloatTensor(actions).view(states.shape[0], self.action_dim)
        advantages = torch.FloatTensor(advantages).view(states.shape[0], self.action_dim)
        log_old_policy_pdf = torch.FloatTensor(old_policy_pdf).view(states.shape[0],1)
        
        mu, std = self.model(states)
        log_policy_pdf = self.log_pdf(mu, std, actions)
        ratio = torch.exp(log_policy_pdf - log_old_policy_pdf)
        clipping_ratio = torch.clamp(ratio,1.0-self.ratio_clipping,1.0+self.ratio_clipping)
        surrogate = -torch.minimum(ratio *advantages, clipping_ratio * advantages)
        loss = torch.mean(surrogate)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))
