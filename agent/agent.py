import torch
import torch.nn.functional as F
from collections import deque
import random
import numpy as np

from models import ACTOR_REGISTRY, CRITIC_REGISTRY

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, nxt, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(nxt),
            torch.FloatTensor(done).unsqueeze(1),
        )

    def serialize(self):
        return {
            "capacity": self.capacity,
            "buffer": self.buffer
        }

    @staticmethod
    def deserialize(data):
        buf = ReplayBuffer(data["capacity"])
        buf.buffer = data["buffer"]
        return buf

    def __len__(self):
        return len(self.buffer)

class OUNoise:
    def __init__(self, dim, mu=0.00, theta=0.15, sigma=0.2,end_sigma=0.01, decay_rate=0.992):
        self.dim = dim
        self.mu = mu * np.ones(dim)
        self.theta = theta
        self.sigma = sigma
        self.end_sigma = end_sigma
        self.decay_rate = decay_rate
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def update_sigma(self):
        if self.sigma > self.end_sigma:
            self.sigma *= self.decay_rate
            self.sigma = max(self.sigma, self.end_sigma)

    def __call__(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.dim)
        self.state += dx
        return self.state

class DDPGAgent:
    def __init__(
        self,
        lidarRays,
        FrameLen,
        action_dim,
        arch_type="fStack",
        device: str = "cpu",
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
    ):
        self.device = torch.device(device)
        self.arch_type = arch_type
        ActorClass = ACTOR_REGISTRY[self.arch_type]
        CriticClass = CRITIC_REGISTRY[self.arch_type]

        # Инициализация актора
        self.actor = ActorClass(lidarRays, FrameLen).to(self.device)
        self.actor_target = ActorClass(lidarRays, FrameLen).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Инициализация критика
        self.critic = CriticClass(lidarRays, FrameLen, action_dim).to(self.device)
        self.critic_target = CriticClass(lidarRays, FrameLen, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # оптимизаторы
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # гиперпараметры
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # буфер и шум
        self.replay = ReplayBuffer(buffer_capacity)
        self.noise = OUNoise(action_dim)

    def select_action(self, state, explore=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        self.actor.train()
        if explore:
            action += self.noise()
        return np.clip(action, -1.0, 1.0)

    def train(self):
        if len(self.replay) < self.batch_size:
            return
        s, a, r, s2, d = self.replay.sample(self.batch_size)

        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s2 = s2.to(self.device)
        d = d.to(self.device)

        with torch.no_grad():
            a2 = self.actor_target(s2)
            q_target = self.critic_target(s2, a2)
            q_target = r + (1 - d) * self.gamma * q_target

        q_current = self.critic(s, a)
        critic_loss = F.mse_loss(q_current, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)

    def soft_update(self, target, source, tau):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(tau * s_param.data + (1.0 - tau) * t_param.data)

    def save(self, filename):
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_opt_state_dict": self.actor_opt.state_dict(),
            "critic_opt_state_dict": self.critic_opt.state_dict(),
            "replay_buffer": self.replay.serialize(),
            "hyperparams": {
                "gamma": self.gamma,
                "tau": self.tau,
                "batch_size": self.batch_size,
                "actor_lr": self.actor_opt.param_groups[0]["lr"],
                "critic_lr": self.critic_opt.param_groups[0]["lr"],
            },
        }, filename)
        print(f" Checkpoint saved to {filename}")

    def load(self, filename, map_location=None):
        ckpt = torch.load(filename,weights_only=False, map_location=map_location or self.device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.actor_target.load_state_dict(ckpt["actor_target_state_dict"])
        self.critic_target.load_state_dict(ckpt["critic_target_state_dict"])
        self.actor_opt.load_state_dict(ckpt["actor_opt_state_dict"])
        self.critic_opt.load_state_dict(ckpt["critic_opt_state_dict"])
        self.replay_buffer = ReplayBuffer.deserialize(ckpt["replay_buffer"])
        hp = ckpt["hyperparams"]
        self.gamma = hp["gamma"]
        self.tau = hp["tau"]
        self.batch_size = hp["batch_size"]
        self.actor_opt.param_groups[0]["lr"]=hp["actor_lr"]
        self.critic_opt.param_groups[0]["lr"]=hp["critic_lr"]
        print(f" Checkpoint loaded from {filename}")
