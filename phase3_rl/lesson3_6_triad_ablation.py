"""Lesson 3.6 — The Deadly Triad, experimentally: DQN ablations.

Same inventory env as 3.1/3.3 (11 states — small enough to grade against
exact VI, big enough for a 2-layer net to matter). Five variants, identical
budget, identical seeds:
  0. linear-FA TD on coarse features (the provably-safe regime, from 2.4)
  1. full DQN (replay + target net + Huber)  — the reference
  2. NO target network  (bootstrap on the moving net)
  3. NO replay buffer   (train on the raw trajectory)
  4. NEITHER            (pure online regression to moving targets)
Reports: % of exact policy performance + training-curve volatility.
Runtime: ~90 s total on 4 CPU cores.
Run:  python lesson3_6_triad_ablation.py
"""
import random

import numpy as np
import torch
import torch.nn as nn

from lesson3_1_qlearning_inventory import (
    InventoryEnv, exact_solution, valid_actions,
    MAX_INV, N_STATES, N_ACTIONS, EP_LEN, evaluate, policy_exact,
)
GAMMA = 0.95
EPISODES = 300
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


def dqn_variant(use_replay=True, use_target=True, seed=7, linear=False):
    """Train one variant; return greedy policy table + volatility metric."""
    torch.manual_seed(seed); random.seed(seed)
    env = InventoryEnv(seed)
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)

    if linear:
        # linear-FA: Q(s,a) = theta . phi(s) * onehot(a) — provably stable
        # in the on-policy setting; we keep eps-greedy + online updates.
        theta = np.zeros((N_STATES, N_ACTIONS))
        net = None
    else:
        net = qnet()
        net2 = qnet()
        net2.load_state_dict(net.state_dict())
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        buf = []

    policy_perf_curve = []
    steps = 0
    for ep in range(EPISODES):
        s = env.reset()
        eps = max(0.05, 1.0 - ep / (0.7 * EPISODES))
        ep_r = 0.0
        for _ in range(EP_LEN):
            if linear:
                a = (rng.random() < eps and rng.choice(valid_actions(s))
                     or int(theta[s].argmax()))
            else:
                with torch.no_grad():
                    q = net(torch.tensor(one_hot(s)))
                a = (rng.random() < eps and rng.choice(valid_actions(s))
                     or int(q.argmax()))
            a = int(a) if not isinstance(a, int) else a
            a = int(min(max(a, 0), MAX_INV - s))   # clip to feasible range
            s2, r = env.step(s, a)
            ep_r += r
            steps += 1

            if linear:
                td = r + GAMMA * theta[s2].max() - theta[s, a]
                theta[s, a] += 0.02 * td
            else:
                buf.append((s, a, r, s2))
                if len(buf) > BUFFER:
                    buf.pop(0)
                if use_replay:
                    batch = pyrng.sample(buf, min(BATCH, len(buf)))
                else:
                    batch = [(s, a, r, s2)]
                obs = torch.tensor(np.stack([one_hot(b[0]) for b in batch]))
                act = torch.tensor([b[1] for b in batch])
                rew = torch.tensor([float(b[2]) for b in batch])
                nxt = torch.tensor(np.stack([one_hot(b[3]) for b in batch]))
                with torch.no_grad():
                    src = net2 if use_target else net
                    q_next = src(nxt).max(dim=1).values
                    target = rew + GAMMA * q_next
                q_sa = net(obs).gather(1, act.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(q_sa, target)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                if use_target and steps % SYNC == 0:
                    net2.load_state_dict(net.state_dict())
            s = s2
        policy_perf_curve.append(ep_r)

    if linear:
        pol = theta.argmax(axis=1)
    else:
        with torch.no_grad():
            obs_all = torch.tensor(np.stack([one_hot(x) for x in range(N_STATES)]))
            pol = net(obs_all).argmax(dim=1).numpy()
    curve = np.array(policy_perf_curve)
    volatility = float(np.std(np.diff(np.convolve(curve, np.ones(51) / 51, mode="valid"))))
    return pol, volatility, curve


if __name__ == "__main__":
    V_exact, policy_exact = exact_solution()
    exact_perf = evaluate(policy_exact)
    print(f"exact (s,S) policy performance: {exact_perf:.1f}\n")

    variants = [
        ("linear-FA TD (on-policy, provably stable)", dict(linear=True)),
        ("full DQN (replay + target + Huber)", {}),
        ("NO target net", dict(use_target=False)),
        ("NO replay", dict(use_replay=False)),
        ("NEITHER (online moving targets)", dict(use_replay=False, use_target=False)),
    ]
    print(f"{'variant':44s} {'%exact':>7s} {'volatility':>11s}")
    for name, kw in variants:
        pol, vol, curve = dqn_variant(**kw)
        pol = np.array([min(int(x), MAX_INV - s) for s, x in enumerate(pol)])
        perf = evaluate(pol)
        print(f"{name:44s} {100*perf/exact_perf:6.1f}% {vol:11.2f}")
        print(f"    policy: {list(pol)}")
