"""Lesson 3.6 figure: training curves of the triad ablation variants.

Re-runs THREE key variants (full DQN / no-replay / divergent config) at a
reduced budget (400 episodes) with the same dynamics as
phase3_rl/lesson3_6_triad_ablation.py, plotting smoothed episode returns +
max|Q| growth. The full-budget table lives in the lesson's evidence file.
Runtime ~4 min on 4 cores.
Run:  python phase2_mdp/make_figures_36.py
"""
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "phase3")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "phase3_rl"))

from lesson3_1_qlearning_inventory import (  # noqa: E402
    InventoryEnv, MAX_INV, N_STATES, N_ACTIONS)

GAMMA = 0.95
EPISODES = 400
BATCH = 32
LR = 8e-4
BUFFER = 5_000
SYNC = 150


def one_hot(s):
    v = np.zeros(N_STATES, dtype=np.float32)
    v[s] = 1.0
    return v


def qnet(hidden=64):
    return nn.Sequential(
        nn.Linear(N_STATES, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, N_ACTIONS))


def train_variant(use_replay=True, use_target=True, seed=7,
                  lr=LR, clip_grads=True):
    torch.manual_seed(seed); random.seed(seed)
    env = InventoryEnv(seed)
    pyrng = random.Random(seed)
    net = qnet()
    net2 = qnet()
    net2.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    buf, curve, qmax_hist = [], [], []
    steps = 0
    for ep in range(EPISODES):
        s = env.reset()
        eps = max(0.05, 1.0 - ep / (0.7 * EPISODES))
        ep_r = 0.0
        for _ in range(50):
            with torch.no_grad():
                q = net(torch.tensor(one_hot(s)))
            a = (random.random() < eps and random.choice(
                range(MAX_INV - s + 1)) or int(q.argmax()))
            a = int(min(max(int(a), 0), MAX_INV - s))
            s2, r = env.step(s, a)
            ep_r += r
            steps += 1
            buf.append((s, a, r, s2))
            if len(buf) > BUFFER:
                buf.pop(0)
            batch = (pyrng.sample(buf, min(BATCH, len(buf)))
                     if use_replay else [buf[-1]])
            obs = torch.tensor(np.stack([one_hot(b[0]) for b in batch]))
            act = torch.tensor([b[1] for b in batch])
            rew = torch.tensor([float(b[2]) for b in batch])
            nxt = torch.tensor(np.stack([one_hot(b[3]) for b in batch]))
            with torch.no_grad():
                src = net2 if use_target else net
                target = rew + GAMMA * src(nxt).max(dim=1).values
            q_sa = net(obs).gather(1, act.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(q_sa, target)
            opt.zero_grad(); loss.backward()
            if clip_grads:
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            if use_target and steps % SYNC == 0:
                net2.load_state_dict(net.state_dict())
            s = s2
        curve.append(ep_r)
        with torch.no_grad():
            all_q = net(torch.tensor(np.stack(
                [one_hot(x) for x in range(N_STATES)]))).numpy()
        qmax_hist.append(float(np.abs(all_q).max()))
    return np.array(curve), np.array(qmax_hist)


def smooth(x, k=25):
    return np.convolve(x, np.ones(k) / k, mode="valid")


def main():
    variants = [
        ("full DQN (replay+target+clip)", dict(), "#2ca02c"),
        ("NO replay", dict(use_replay=False), "#ff7f0e"),
        ("DIVERGENT (no clip, lr 3e-3)", dict(lr=3e-3, clip_grads=False,
                                              use_replay=False,
                                              use_target=False), "#d62728"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    ax = axes[0]
    axq = axes[1]
    for name, kw, color in variants:
        curve, qmax = train_variant(**kw)
        sm = smooth(curve)
        ax.plot(sm, lw=1.5, color=color,
                label=f"{name} — final {sm[-1]:.0f}")
        axq.plot(qmax, lw=1.5, color=color)
        print(f"{name}: final smoothed {sm[-1]:.0f}, max|Q| end {qmax[-1]:.1f}")
    ax.set_xlabel("episode")
    ax.set_ylabel("episode return (smoothed, k=25)")
    ax.set_title("training curves — the policy-level symptom")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    axq.set_xlabel("episode")
    axq.set_ylabel("max |Q| over all states")
    axq.set_title("the value-level symptom: Q blow-up")
    axq.set_yscale("log")
    axq.grid(alpha=0.3, which="both")

    fig.tight_layout()
    path = os.path.join(OUT, "lesson3_6_triad_curves.png")
    fig.savefig(path, bbox_inches="tight")
    print("saved:", os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()
