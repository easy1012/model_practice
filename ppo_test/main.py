from agent import Agent
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import gym

def main():
    max_episode_num = 1000

    env_name = 'Pendulum-v1'
    #env_name = 'Pendulum-v0'

    env = gym.make(env_name)
    agent = Agent(env)

    agent.Learn(max_episode_num)

    agent.plot_result()


if __name__ == "__main__":
    main()