import gym
import numpy as np
import torch
from torch import nn

from actor_net import DynamicModule
from units import ReplayBuffer_D


class DynamicAgentTrainer:
    def __init__(self, env_name, lr, batch_size, update_freq, step_number, capacity):
        self.env_name = env_name
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.step_number = step_number

        self.env = gym.make(env_name)
        self.env.reset()

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.buffer = ReplayBuffer_D(self.state_dim, self.action_dim, int(capacity))

        self.dynamic_module = DynamicModule(self.state_dim, self.action_dim)
        self.optimizer = torch.optim.Adam(self.dynamic_module.parameters(), lr=lr)
        self.loss_fn = nn.HuberLoss()

        self.eval_reward = []
        self.time = []

    def learn(self, episode):
        batch_s, batch_s_n, batch_a = self.buffer.sample(self.batch_size)

        input = torch.cat([torch.FloatTensor(batch_s), torch.FloatTensor(batch_s_n)], dim=1)
        a_pred = self.dynamic_module.forward(input)
        a_true = torch.FloatTensor(batch_a)
        loss = self.loss_fn(a_pred, a_true)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print('Episode - ', episode + 1, ', loss - ', loss.detach().numpy())

    def iteration(self, max_episode, animation=True):
        episode = 0
        total_timesteps = 0

        while True:
            s = self.env.reset()

            done = False
            while not done:
                if animation: self.env.render()

                a = self.env.action_space.sample()
                s_n, _, done, _ = self.env.step(a)

                self.buffer.put(s, s_n, a)
                s = s_n

                if total_timesteps > self.update_freq:
                    self.learn(episode)
                    self.save()

                    episode += 1
                    total_timesteps = 0

                total_timesteps += 1

                if episode >= max_episode:
                    return

    def save(self):
        torch.save(self.dynamic_module.state_dict(), "model/" + self.env_name + "/dynamics")


if __name__ == "__main__":
    module = DynamicAgentTrainer(
        env_name='HalfCheetah-v3',
        lr=1e-3,
        batch_size=256,
        update_freq=128,
        step_number=5000,
        capacity=1e6
    )

    module.iteration(3000, False)
