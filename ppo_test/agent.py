import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from actor import Actor
from critic import Critic

class Agent(object):
    def __init__(self,env):
        self.GAMMA = 0.95
        self.GAE_LAMBDA =0.9
        self.BATCH_SIZE = 64
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.RATIO_CLIPPING = 0.2
        self.EPOCHS = 10

        self.env = env

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]

        self.actor = Actor(self.state_dim,self.action_dim,self.action_bound,self.ACTOR_LEARNING_RATE,self.RATIO_CLIPPING)
        self.critic = Critic(self.state_dim,self.action_dim,self.CRITIC_LEARNING_RATE)

        self.save_epi_reward = []

    def gae_target(self,rewards,v_values,next_v_value,done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0
        v_values = v_values.detach().numpy()
        if not done:
            foward_val = next_v_value
        
        for k in reversed(range(0,len(rewards))):
            delta = rewards[k] + self.GAMMA *forward_val - v_values[k] 
            gae_cumulative = self.GAMMA *self.GAE_LAMBDA * gae_cumulative + delta
            gae[k]  = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae,n_step_targets

    def unpack_batch(self,batch):
        unpack = batch[0]
        for idx in range(len(batch)-1):
            unpack = np.append(unpack,batch[idx +1],axis = 0)
        return unpack

    def Learn(self,max_episode_num):
        batch_state,batch_action, batch_reward = [],[],[]
        batch_log_old_policy_pdf = []

        for ep in range(int(max_episode_num)):
            time,episode_reward,done = 0,0,False
            state = self.env.reset()
            while not done:
                # self.env.render()
                mu_old, std_old, action = self.actor.get_policy_action(torch.tensor(state))
                action = np.clip(action, -self.action_bound,self.action_bound)
                var_old = std_old ** 2
                log_old_policy_pdf = -0.5 * (action - mu_old) ** 2 / var_old - 0.5 * np.log(var_old * 2 * np.pi)
                log_old_policy_pdf = np.sum(log_old_policy_pdf)

                next_state,reward,done,_ = self.env.step(action)

                state = np.reshape(state,[1,self.state_dim])
                action = np.reshape(action,[1,self.action_dim])
                reward = np.reshape(reward, [1,1])
                log_old_policy_pdf = np.reshape(log_old_policy_pdf,[1,1])

                batch_state.append(state)
                batch_action.append(action)
                batch_reward.append((reward + 8)/8)
                batch_log_old_policy_pdf.append(log_old_policy_pdf)

                if len(batch_state) < self.BATCH_SIZE:
                    state = np.reshape(next_state,[1,self.state_dim])
                    episode_reward += reward[0]
                    time += 1
                    continue

                states = self.unpack_batch(batch_state)
                actions = self.unpack_batch(batch_action)
                rewards = self.unpack_batch(batch_reward)
                log_old_policy_pdfs = self.unpack_batch(batch_log_old_policy_pdf)

                batch_state,batch_action,batch_reward = [],[],[]
                batch_log_old_policy_pdf = []

                next_state = np.reshape(next_state,[1,self.state_dim])
                next_state = torch.from_numpy(next_state).detach()
                next_v_value = self.critic.model(next_state)
                states = torch.from_numpy(states).detach()
                v_values = self.critic.model(states)
                gaes,y_i = self.gae_target(rewards,v_values,next_v_value,done)

                for i in range(self.EPOCHS):
                    self.actor.Learn(log_old_policy_pdfs,states,actions,gaes)
                    self.critic.train_on_batch(states,y_i)

                    state = next_state
                    episode_reward += reward[0]
                    time += 1

                    print('Episode:', ep+1, 'Time:', time, 'Rewrad:', episode_reward)
                    self.save_epi_reward.append(episode_reward)

                    if ep % 10 ==0:
                        self.actor.save_weights('pendulum_actor.th')
                        self.critic.save_weights('pendulum_critic.th')

        print(self.save_epi_reward)
            
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()


