import gym
import numpy as np

env = gym.make('Pendulum-v1')
print(env.observation_space.shape[0])
print(env.action_space.shape[0])
print(env.action_space.high[0])
print(env.reset())
action = np.array([0])
# print(env.action_space)
print(env.step(action)[1])
