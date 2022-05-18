import gym
from agent import DDPGagent as Agent
import torch

def main():
    env_name = 'Pendulum-v1'

    env = gym.make(env_name)
    agent = Agent(env)

    agent.actor.load_weights('./pendulum_actor.th')
    agent.critic.load_weights('./pendulum_critic.th')

    time = 0
    state = env.reset()
    env.render()

    while True:
        env.render()
        action = agent.actor.predict(torch.Tensor(state))
        action = action.detach().numpy()
        state,reward,done,_ = env.step(action)
        time += 1
        print('Time:', time, 'Rewrad:',reward)
        
        if done:
            break
    


if __name__== '__main__':
    main()