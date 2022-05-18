import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from actor import Actor
from critic import Critic
from replaybuffer import ReplayBuffer

class DDPGagent(object):
    def __init__(self,env):
        self.GAMMA = 0.95
        self.BATCH_SIZE = 64
        self.BUFFER_SIZE = 20000
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.TAU = 0.001

        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]

        self.actor = Actor(self.state_dim,self.action_dim,self.action_bound,self.TAU,self.ACTOR_LEARNING_RATE)
        self.critic = Critic(self.state_dim,self.action_dim,self.TAU,self.CRITIC_LEARNING_RATE)

        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        self.save_epi_reward = []

    def ou_noise(self,x,rho = 0.15, mu = 0,dt = 1e-1,sigma = 0.2,dim = 1):
        return x + rho*(mu - x)* dt + sigma *np.sqrt(dt) * np.random.normal(size = dim)

    def td_target(self,rewards,q_values,dones):
        y_k = q_values.detach().numpy()
        q_values = q_values.detach().numpy()
        for i in range(q_values.shape[0]):
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * q_values[i]

        return y_k

    def Learn(self,max_episode_num):
        self.actor.update_target_network()
        self.critic.update_target_network()

        for ep in range(int(max_episode_num)):
            pre_noise = np.zeros(self.action_dim)
            time, episode_reward, done = 0, 0, False
            state = self.env.reset()
            while not done:
                # self.env.render()
                action =self.actor.predict(torch.tensor(state))
                action = action.detach().numpy()
                noise = self.ou_noise(pre_noise,dim = self.action_dim)
                
                action = np.clip(action + noise, -self.action_bound,self.action_bound)

                next_state, reward, done, _ = self.env.step(action)
                train_reward = (reward + 8)/8
                self.buffer.add_buffer(state, action, train_reward, next_state, done)
                
                if self.buffer.buffer_size > 1000:
                    states,actions,rewards,next_states,dones = self.buffer.sample_batch(self.BATCH_SIZE)
                    tmp_action = self.actor.target_predict(torch.tensor(next_states))
                    target_qs = self.critic.target_predict(torch.tensor(next_states),tmp_action)

                    y_i = self.td_target(rewards,target_qs,dones)

                    self.critic.train_on_batch(states,actions,y_i)

                    s_actions = self.actor.model(torch.tensor(states))
                    s_grads = self.critic.dq_da(torch.tensor(states),s_actions)

                    self.actor.Learn(s_grads)#여기 수정해야함

                    self.actor.update_target_network()
                    self.critic.update_target_network()

                pre_noise = noise
                state = next_state
                episode_reward += reward
                time += 1

            print('Episode:', ep+1, 'Time:', time, 'Rewrad:', episode_reward)
            self.save_epi_reward.append(episode_reward)

            self.actor.save_weights
            self.actor.save_weights('pendulum_actor.th')
            self.critic.save_weights('pendulum_critic.th')
        np.savetxt('./pendulum_epi_reward.txt',self.save_epi_reward)

    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()

