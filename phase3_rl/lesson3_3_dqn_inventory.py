"""Lesson 3.3 — Minimal DQN on the inventory problem (PyTorch, CPU).

Same env as lessons 3.1/3.2 (state 0..10, actions 0..10). Deliberately
tabular-sized so DQN can be graded against the exact Value Iteration
policy. Components demonstrated: replay buffer, target network,
Huber loss, eps-greedy decay, one-hot state encoding.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from lesson3_1_qlearning_inventory import (
    InventoryEnv, exact_solution, valid_actions, greedy_policy,
    MAX_INV, N_STATES, N_ACTIONS, EP_LEN, evaluate, policy_exact,
)

GAMMA = 0.95
EPISODES = 600
BUFFER = 20_000
BATCH = 64
LR = 5e-4
TARGET_SYNC = 500          # env steps between target-network syncs
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 300   # episodes


def one_hot(s):
    v = np.zeros(N_STATES, dtype=np.float32)
    v[s] = 1.0
    return v


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_STATES, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, N_ACTIONS),   # one Q value per action
        )

    def forward(self, x):
        return self.net(x)


def train_dqn(seed=7):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = InventoryEnv(seed)
    online, target = QNet(), QNet()
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=LR)
    buf = deque(maxlen=BUFFER)
    losses, eps_hist = [], []
    steps = 0

    for ep in range(EPISODES):
        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * ep / EPS_DECAY)
        eps_hist.append(eps)
        s = env.reset()
        for _ in range(EP_LEN):
            steps += 1
            if random.random() < eps:
                a = random.choice(valid_actions(s))
            else:
                with torch.no_grad():
                    qs = online(torch.tensor(one_hot(s)))
                    a = max(valid_actions(s), key=lambda x: qs[x].item())
            s2, r = env.step(s, a)
            buf.append((s, a, r / 10.0, s2))  # reward scaling /10
            s = s2

            if len(buf) < BATCH:
                continue
            batch = random.sample(buf, BATCH)
            S = torch.tensor(np.stack([one_hot(b[0]) for b in batch]))
            A = torch.tensor([b[1] for b in batch])
            R = torch.tensor([b[2] for b in batch])
            S2 = torch.tensor(np.stack([one_hot(b[3]) for b in batch]))
            with torch.no_grad():
                # target network computes the bootstrap term (deadly-triad patch)
                max_next = torch.tensor([
                    max(target(S2[i])[x].item() for x in valid_actions(b[3]))
                    for i, b in enumerate(batch)
                ])
                y = R + GAMMA * max_next
            pred = online(S).gather(1, A.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(pred, y)   # Huber
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())

            if steps % TARGET_SYNC == 0:
                target.load_state_dict(online.state_dict())

    return online, losses, eps_hist


if __name__ == "__main__":
    V_exact, policy_exact = exact_solution()
    net, losses, eps_hist = train_dqn()

    def dqn_policy(s):
        with torch.no_grad():
            qs = net(torch.tensor(one_hot(s)))
        return max(valid_actions(s), key=lambda x: qs[x].item())

    dqn_pol = np.array([dqn_policy(s) for s in range(N_STATES)])

    print(f"final loss (last 100 avg): {np.mean(losses[-100:]):.4f}")
    print(f"eps: start {eps_hist[0]:.2f} -> end {eps_hist[-1]:.2f}")
    print("\nfinal greedy policy (order per inventory level 0..10):")
    print(f"  exact DP : {policy_exact}")
    print(f"  DQN      : {dqn_pol}")

    perf_exact = evaluate(policy_exact)
    perf_dqn = evaluate(dqn_pol)
    print(f"\n2000-episode evaluation:")
    print(f"  exact DP: {perf_exact:8.1f} (100%)")
    print(f"  DQN     : {perf_dqn:8.1f} ({100*perf_dqn/perf_exact:.1f}% of exact)")
