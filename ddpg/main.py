import gym
from agent import DDPGagent

def main():
    max_episode_num = 200
    env_name = 'Pendulum-v1'
    #env_name = 'Pendulum-v0'

    env = gym.make(env_name)

    agent = DDPGagent(env)

    agent.Learn(max_episode_num)

    agent.plot_result()


if __name__ == "__main__":
    main()