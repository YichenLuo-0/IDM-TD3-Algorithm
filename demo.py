import gym
import torch

from actor_net import ActorNet


class Agent(object):
    def __init__(self, env_name):
        self.env = gym.make(env_name)

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        # Load the actor net
        self.actor = ActorNet(state_dim, action_dim)
        self.actor.load_state_dict(torch.load("model/" + env_name + "/actor"))
        # Load the dynamic component
        self.actor.load_dynamic_module(env_name + "\dynamics")

    def action(self, s):
        s = torch.tensor([s], dtype=torch.float)
        a = self.actor(s).detach().numpy()
        return a

    def play(self, episode):
        s = self.env.reset()
        done = False

        for i in range(episode):
            episode_reward = 0
            while not done:
                self.env.render()
                a = self.action(s)[0]
                s, r, done, _ = self.env.step(a)
                episode_reward += r

            s = self.env.reset()
            done = False
            print("Episode: ", i + 1, ", reward: ", episode_reward)


if __name__ == '__main__':
    agent = Agent('HalfCheetah-v3')
    agent.play(50)
