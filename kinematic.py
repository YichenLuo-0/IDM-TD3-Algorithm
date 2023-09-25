import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from actor_net import ActorNet
from critic_net import CriticNet
from units import ReplayBuffer_K


class TD3AgentTrainer(object):
    def __init__(self, env_name, actor_lr, critic_lr, gamma, tau, capacity, start_timesteps, batch_size, expl_noise,
                 policy_noise, policy_noise_clip, policy_freq):
        # Initialize the hyperparameters
        self.env_name = env_name
        self.gamma = gamma
        self.tau = tau
        self.start_timesteps = start_timesteps
        self.batch_size = batch_size
        self.expl_noise = expl_noise
        self.policy_noise = policy_noise
        self.policy_noise_clip = policy_noise_clip
        self.policy_freq = policy_freq

        # Initialize the Gym event
        self.env = gym.make(env_name)
        self.env.reset()

        # Dimensions of observation space and action space
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = self.env.action_space.high[0]
        # Replay buffer
        self.buffer = ReplayBuffer_K(self.state_dim, self.action_dim, int(capacity))

        # Actor network
        self.actor = ActorNet(self.state_dim, self.action_dim)
        self.actor_target = ActorNet(self.state_dim, self.action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic network
        self.critic = CriticNet(self.state_dim, self.action_dim)
        self.critic_target = CriticNet(self.state_dim, self.action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizer
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Loading the pre-trained IDM
        self.actor.load_dynamic_module(env_name)
        self.actor_target.load_dynamic_module(env_name)

        self.total_it = 0

    def action(self, s):
        s = torch.tensor([s], dtype=torch.float)
        a = self.actor(s).detach().numpy().flatten()
        return a

    def learn(self):
        s, a, r, s_n, not_done = self.buffer.sample(self.batch_size)

        # Gradient descent for critic networks
        def critic_learn():
            with torch.no_grad():
                # Get next action and add target policy smoothing noise
                a_n = self.actor_target(s_n)
                noise = torch.clamp(torch.randn_like(a_n) * self.policy_noise, self.policy_noise_clip,
                                    -self.policy_noise_clip)
                a_n = torch.clamp(a_n + noise, -self.max_action, self.max_action)

                target_q1, target_q2 = self.critic_target(s_n, a_n)
                target_q = r + self.gamma * not_done * torch.min(target_q1, target_q2)

            # The smallest q-value is chosen as the gradient
            q1, q2 = self.critic(s, a)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

        # Gradient descent for actor networks
        def actor_learn():
            actor_loss = -self.critic.Q1(s, self.actor(s)).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

        # Soft update target networls
        def soft_update(net_target, net, tau):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        critic_learn()
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            actor_learn()
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)

        self.total_it += 1

    def iteration(self, max_episode, animation=True):
        s = self.env.reset()
        total_timesteps = 0

        for i in range(max_episode):
            episode_reward = 0
            done = False

            while not done:
                if total_timesteps >= self.start_timesteps:
                    # Uniformly distributed random noise is used in policy
                    a = (
                            self.action(s)
                            + np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
                    ).clip(-self.max_action, self.max_action)
                else:
                    # Random policy
                    a = self.env.action_space.sample()

                # Take a step
                s_n, r, done, _ = self.env.step(a)
                self.buffer.put(s, a, r, s_n, done)
                s = s_n

                # Perform the learning process
                if total_timesteps >= self.start_timesteps:
                    self.learn()
                    if animation: self.env.render()
                    # Perform the evaluation every 2000 steps
                    if total_timesteps % 2000 == 0: self.eval()

                episode_reward += r
                total_timesteps += 1

            s = self.env.reset()
            print('Episode - ', i + 1, ', total reward - ', episode_reward, ', timestep - ', total_timesteps)

    def eval(self):
        # The target network was used for testing
        actor = self.actor_target
        eval_env = gym.make(self.env_name)
        s = eval_env.reset()

        r_buffer = []

        for i in range(10):
            total_r = 0
            done = False

            while not done:
                s = torch.tensor([s], dtype=torch.float)
                # No random noise is added to the test
                a = actor(s).detach().numpy().flatten()
                s_n, r, done, _ = eval_env.step(a)

                total_r += r
                s = s_n

            r_buffer.append(total_r)
            s = eval_env.reset()

        print('-----------------------------------')
        print('eval reward:', np.mean(np.array(r_buffer)))
        print('-----------------------------------')

    def save(self):
        torch.save(self.actor_target.state_dict(), "model/" + self.env_name + "/actor")
        torch.save(self.critic_target.state_dict(), "model/" + self.env_name + "/critic")


if __name__ == '__main__':
    trainer = TD3AgentTrainer(
        env_name='HalfCheetah-v3',
        actor_lr=5e-4,
        critic_lr=5e-4,
        gamma=0.99,
        tau=0.005,
        capacity=1e6,
        start_timesteps=25e3,
        batch_size=256,
        expl_noise=0.2,
        policy_noise=0.2,
        policy_noise_clip=0.5,
        policy_freq=6
    )

    trainer.iteration(100000, False)
