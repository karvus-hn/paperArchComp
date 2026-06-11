import torch
import torch.nn as nn


class Att_Actor(nn.Module):
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

        self.embedding = nn.Linear(64, 128)
        self.pos_embedding = nn.Parameter(torch.randn(1, 3, 128))
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc_final = nn.Linear(128 + 128, 128)
        self.fc2a = nn.Linear(128, 1)
        self.fc2b = nn.Linear(128, 1)
        self.tanh = nn.Tanh()

    def forward(self, state):
        batch_size = state.size(0)
        robot_data = state[:, :(-self.lidar_rays * self.frLen)]
        lidar_seq = state[:, (-self.lidar_rays*self.frLen):].view(batch_size, self.frLen, self.lidar_rays)
        x_emb = self.embedding(lidar_seq)
        x_emb = x_emb + self.pos_embedding
        attn_out = self.transformer(x_emb)
        context_vector = torch.mean(attn_out, dim=1)
        x = self.net(robot_data)
        combined = torch.cat([x,context_vector], dim=1)
        x1 = self.fc_final(combined)
        a1 = self.tanh(self.fc2a(x1))
        a2 = self.tanh(self.fc2b(x1))
        ret=torch.cat([a1, a2], dim=1)
        return ret

class Att_Critic(nn.Module):
    def __init__(self,lidarRays,FrameLen, action_dim):
        super().__init__()
        self.lidar_rays = lidarRays
        self.frLen = FrameLen
        self.embedding = nn.Linear(64, 128)
        self.pos_embedding = nn.Parameter(torch.randn(1, 3, 128))
        encoder_layer = nn.TransformerEncoderLayer(d_model=128,nhead=4,dim_feedforward=128,batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

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
        lidar_seq = state[:, (-self.lidar_rays*self.frLen):].view(batch_size, self.frLen, self.lidar_rays)
        x_emb = self.embedding(lidar_seq)
        x_emb = x_emb + self.pos_embedding
        attn_out = self.transformer(x_emb)
        context_vector = torch.mean(attn_out, dim=1)
        combined = torch.cat([context_vector, robot_data, action], dim=1)
        return self.net(combined)
