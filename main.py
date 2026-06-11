import statistics
import torch
import matplotlib.pyplot as plt
import random
import timeit
import csv
import numpy as np
import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from env.env2D import Simple2DEnv
from agent.agent import DDPGAgent

RANDOM_SEED=155
frameLen=3

def save_all_rewards(filename, history_list):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward'])
        writer.writerows(history_list)
    print(f"Награды успешно сохранены в файл: {filename}")

def plot_single_variant(rewards_list, label_name="test", window_size=40):
    data = np.array(rewards_list)
    episodes = np.arange(1, len(data) + 1)
    mean = np.array([np.mean(data[max(0, i - window_size + 1): i + 1]) for i in range(len(data))])
    std = np.array([np.std(data[max(0, i - window_size + 1): i + 1]) for i in range(len(data))])
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.plot(episodes, mean, label=label_name, color='#000000', linewidth=2.0,
            linestyle='-', marker='o', markersize=5, markevery=40)
    ax.fill_between(episodes, mean - std, mean + std, color='#000000', alpha=0.08)
    ax.set_xlabel('Эпизод', fontsize=12)
    ax.set_ylabel('Награда', fontsize=12)
    ax.set_xlim(0, len(data))
    ax.grid(True, linestyle=':', alpha=0.6, color='#aaaaaa')
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#cccccc')
    plt.tight_layout()
    plt.savefig(f'training_plot_bw_{label_name}.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def train_ddpg(env,
               agent,
               episodes=400,
               max_steps=1000,label='test'):
    rewards = []
    rewards_history = []
    steps=[]
    for ep in range(episodes):
        state = env.reset()
        agent.noise.reset()
        ep_reward = 0.0
        for step in range(max_steps):
            raw_a = agent.select_action(state, explore=True)
            nxt, r, done, trunc,info = env.step(raw_a)
            agent.replay.push(state, raw_a, r, nxt, done)
            agent.train()
            state = nxt
            ep_reward += r
            if done:
                rewards_history.append([ep, ep_reward])
                break

        steps.append(env.steps)
        rewards.append(ep_reward)
        agent.noise.update_sigma()
        # визуализация
        # if (ep % 50 == 0):
        #     env.render()
        print(f"Эпизод {ep+1:3d} | награда = {ep_reward:6.2f} | {info}")
    print(statistics.mean(steps))
    plot_single_variant(rewards,label_name=label)
    save_all_rewards(f'rewards_{label}.csv', rewards_history)

ARCH_TYPES = ["fStack", "conv-lstm", "att"]

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    dif='train'
    env = Simple2DEnv(world_size=10, seed=RANDOM_SEED, max_lin_vel=0.25, max_lin_acc=0.5, max_ang_vel=1.0, max_ang_acc=0.15,
                      difficulty=dif, use_lidar=True )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # "fStack" | "conv-lstm" | "att"
    parser = argparse.ArgumentParser(description="Запуск обучения.")
    parser.add_argument("--arch", type=str, required=True,choices=ARCH_TYPES, help="Архитектура обучения")
    arch = parser.parse_args().arch #"att"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = DDPGAgent(lidarRays=env.lidar_num_rays,FrameLen=frameLen,
                      action_dim=action_dim,
                      arch_type=arch,
                      device=device,
                      gamma=0.99, tau=0.01,
                      actor_lr=1e-5, critic_lr=1e-4,
                      batch_size=128,
                      buffer_capacity=1000_000)

    steps=500 if dif=='train' else 100
    try:
        start = timeit.default_timer()
        train_ddpg(env,
                   agent,
                   episodes=steps,
                   max_steps=env.max_steps,label=arch)
        agent.save(f"ddpg_{frameLen}fr_paper{steps}_{arch}t.pth")
        stop = timeit.default_timer()
        print('Time: ', stop - start)
    finally:
        env.close()