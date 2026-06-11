import torch
import torch.nn as nn

class ConvLSTM_Actor(nn.Module):
    def __init__(self,lidarRays,FrameLen):
        super().__init__()
        self.lidar_rays=lidarRays
        self.frLen=FrameLen
        self.net = nn.Sequential(
            nn.Linear(12, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.lidar_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.lstm = nn.LSTM(input_size=448, hidden_size=128, batch_first=True)

        self.fc_final = nn.Linear(128 + 128, 128)
        self.fc2a = nn.Linear(128, 1)
        self.fc2b = nn.Linear(128, 1)
        self.tanh = nn.Tanh()

    def forward(self, state):
        batch_size = state.size(0)
        robot_data = state[:, :(-self.lidar_rays * self.frLen)]
        lidar_flat = state[:, (-self.lidar_rays * self.frLen):].contiguous().view(batch_size * self.frLen, 1, self.lidar_rays)
        conv_features = self.lidar_conv(lidar_flat)
        lstm_input = conv_features.view(batch_size, self.frLen, -1)
        x=self.net(robot_data)
        lstm_out, _ = self.lstm(lstm_input)
        y = lstm_out[:, -1, :]
        combined = torch.cat([x, y], dim=1)
        x = torch.relu(self.fc_final(combined))
        a1=self.tanh(self.fc2a(x))
        a2 = self.tanh(self.fc2b(x))
        ret=torch.cat([a1, a2], dim=1)
        return ret

class ConvLSTM_Critic(nn.Module):
    def __init__(self,lidarRays,FrameLen, action_dim):
        super().__init__()
        self.lidar_rays = lidarRays
        self.frLen = FrameLen

        self.lidar_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.lstm = nn.LSTM(input_size=448, hidden_size=128, batch_first=True)

        combined_dim = 128 + 12 + action_dim

        self.net = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        batch_size = state.size(0)
        robot_data = state[:, :(-self.lidar_rays*self.frLen)]
        lidar_flat = state[:, (-self.lidar_rays * self.frLen):].contiguous().view(batch_size * self.frLen, 1, self.lidar_rays)
        conv_out = self.lidar_conv(lidar_flat)
        lstm_input = conv_out.view(batch_size, self.frLen, -1)
        lstm_out, _ = self.lstm(lstm_input)
        lidar_features = lstm_out[:, -1, :]
        combined = torch.cat([lidar_features, robot_data, action], dim=1)
        return self.net(combined)