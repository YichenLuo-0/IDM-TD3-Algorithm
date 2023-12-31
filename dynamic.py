import argparse

import gym
import torch
from torch import nn

from actor_net import DynamicModule
from units import ReplayBuffer_D


class DynamicAgentTrainer:
    def __init__(self, env_name, lr, batch_size, update_freq, step_number, capacity):
        # Initialize the hyperparameters
        self.env_name = env_name
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.step_number = step_number

        # Initialize the Gym event
        self.env = gym.make(env_name)
        self.env.reset()

        # Dimensions of observation space and action space
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        # Replay buffer
        self.buffer = ReplayBuffer_D(self.state_dim, self.action_dim, int(capacity))

        # Dynamic component
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

                # Using a random policy
                a = self.env.action_space.sample()
                s_n, _, done, _ = self.env.step(a)

                self.buffer.put(s, s_n, a)
                s = s_n

                # The IDM is updated at a fixed frequency
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
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_name", default="HalfCheetah-v3")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--update_freq", default=32, type=int)
    parser.add_argument("--step_number", default=5000, type=int)
    parser.add_argument("--capacity", default=1e6, type=int)

    parser.add_argument("--max_episode", default=5000, type=int)
    parser.add_argument("--animation", default=False)
    args = parser.parse_args()

    module = DynamicAgentTrainer(
        env_name=args.env_name,
        lr=args.lr,
        batch_size=args.batch_size,
        update_freq=args.update_freq,
        step_number=args.step_number,
        capacity=args.capacity
    )
    module.iteration(args.max_episode, args.animation)
