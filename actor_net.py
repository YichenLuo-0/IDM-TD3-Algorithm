import torch
import torch.nn.functional as F
from torch import nn


class KinematicModule(nn.Module):
    def __init__(self, state_size):
        super(KinematicModule, self).__init__()

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, state_size)

    def forward(self, s):
        x = F.leaky_relu(self.fc1(s))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DynamicModule(nn.Module):
    def __init__(self, state_size, actor_size):
        super(DynamicModule, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=20, out_channels=40, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(state_size * 40 * 2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, actor_size)

    def forward(self, x):
        input_size = x.size()
        x = x.reshape([input_size[0], 1, input_size[1]])

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.reshape([input_size[0], input_size[1] * 40])

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x


class ActorNet(nn.Module):
    def __init__(self, state_size, actor_size):
        super(ActorNet, self).__init__()

        self.kinematic_module = KinematicModule(state_size)

        self.dynamic_module = DynamicModule(state_size, actor_size)
        self.dynamic_module.requires_grad_(False)

    def load_dynamic_module(self, mode_name):
        state_dict = torch.load("model/" + mode_name + "/dynamics")
        self.dynamic_module.load_state_dict(state_dict)

    def load_kinematic_module(self, mode_name):
        state_dict = torch.load("model/" + mode_name + "/kinematics")
        self.kinematic_module.load_state_dict(state_dict)

    def forward(self, s):
        x = self.kinematic_module(s)
        x = torch.cat([s, x], dim=1)
        x = self.dynamic_module(x)
        return x
