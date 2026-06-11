import torch
import torch.nn as nn

class FrameStack_Actor(nn.Module):
    def __init__(self,lidarRays,FrameLen):
        super().__init__()
        self.lidar_rays = lidarRays
        self.frLen = FrameLen
        self.net = nn.Sequential(
            nn.Linear(12, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.lidar_net = nn.Sequential(
            nn.Linear(self.frLen * self.lidar_rays, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        self.fc_final = nn.Linear(128 + 128, 128)
        self.fc2a = nn.Linear(128, 1)
        self.fc2b = nn.Linear(128, 1)
        self.tanh = nn.Tanh()

    def forward(self, state):
        robot_data = state[:, :(-self.lidar_rays * self.frLen)]
        lidar_flat = state[:, (-self.lidar_rays * self.frLen):]
        stack_features=self.lidar_net(lidar_flat)
        x = self.net(robot_data)
        combined = torch.cat([x, stack_features], dim=1)
        x1 = self.fc_final(combined)
        a1=self.tanh(self.fc2a(x1))
        a2 = self.tanh(self.fc2b(x1))
        ret=torch.cat([a1, a2], dim=1)
        return ret

class FrameStack_Critic(nn.Module):
    def __init__(self, lidarRays, FrameLen, action_dim):
        super().__init__()
        self.lidar_rays = lidarRays
        self.frLen = FrameLen

        self.lidar_net = nn.Sequential(
            nn.Linear(self.frLen * self.lidar_rays, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        combined_dim = 128 + 12 + action_dim
        self.net = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)            # Q value
        )

    def forward(self, state, action):
        robot_data = state[:, :(-self.lidar_rays*self.frLen)]
        lidar_flat = state[:, (-self.lidar_rays * self.frLen):]
        stack_out=self.lidar_net(lidar_flat)
        combined = torch.cat([stack_out, robot_data, action], dim=1)
        return self.net(combined)

