"""Lesson 4.6 add-on A — mini QR-DQN: learn the RETURN DISTRIBUTION,
then read CVaR off the quantile heads (no re-training per risk level).

Same inventory env as lessons 3.1/3.3. QR-DQN with N=9 quantile heads:
targets are the pinball-loss-quantiles of the sampled return distribution
(one-step distributional Bellman). After training:
  mean, VaR, CVaR read directly from the heads (§3.5 of the note)
  compared against the EMPIRICAL distribution from many rollouts of the
  greedy policy — the honesty check.
Runtime ~90 s on CPU.
Run:  python phase4_hybrid/lesson4_6a_qrdqn.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "phase3_rl"))

import random

import numpy as np
import torch
import torch.nn as nn

from lesson3_1_qlearning_inventory import (
    InventoryEnv, MAX_INV, N_STATES, EP_LEN)

GAMMA = 0.95
N_QUANTILES = 9
N_ACTIONS = MAX_INV + 1
TAUS = (2 * np.arange(1, N_QUANTILES + 1) - 1) / (2 * N_QUANTILES)
EPISODES = 400


def one_hot(s):
    v = np.zeros(N_STATES, dtype=np.float32)
    v[s] = 1.0
    return v


def qnet():
    return nn.Sequential(nn.Linear(N_STATES, 64), nn.ReLU(),
                         nn.Linear(64, 64), nn.ReLU(),
                         nn.Linear(64, N_ACTIONS * N_QUANTILES))


def pinball(pred_q, target, taus):
    """pred_q: (batch, N) quantile predictions for one (s,a);
    target: scalar sampled return. Mean pinball loss over heads."""
    u = target - pred_q
    return torch.mean(torch.where(u >= 0, taus * u, (taus - 1) * u))


def train_qrdqn(seed=7, episodes=EPISODES):
    torch.manual_seed(seed); random.seed(seed)
    env, pyrng = InventoryEnv(seed), random.Random(seed)
    net, tgt = qnet(), qnet()
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    buf, steps = [], 0
    for ep in range(episodes):
        s = env.reset()
        eps = max(0.05, 1.0 - ep / (0.7 * episodes))
        for _ in range(50):
            with torch.no_grad():
                q = net(torch.tensor(one_hot(s)).unsqueeze(0))
            a = (random.random() < eps and
                 random.randrange(MAX_INV - s + 1) or
                 int(q.reshape(N_ACTIONS, N_QUANTILES).mean(1).argmax()))
            a = int(min(max(a, 0), MAX_INV - s))
            s2, r = env.step(s, a)
            steps += 1
            buf.append((s, a, r, s2))
            if len(buf) > 5000:
                buf.pop(0)
            batch = pyrng.sample(buf, min(64, len(buf)))
            # distributional target: for each sample, sample ONE quantile
            # head of the next state's max and treat it as the return draw
            obs = torch.tensor(np.stack([one_hot(b[0]) for b in batch]))
            act = torch.tensor([b[1] for b in batch])
            rew = torch.tensor([float(b[2]) for b in batch])
            nxt = torch.tensor(np.stack([one_hot(b[3]) for b in batch]))
            with torch.no_grad():
                q_next = tgt(nxt).reshape(len(batch), N_ACTIONS,
                                          N_QUANTILES)
                next_a = q_next.mean(dim=2).argmax(dim=1)
                z_next = q_next[torch.arange(len(batch)), next_a]
                # sample a quantile head per sample (random-tau Bellman)
                k = torch.randint(0, N_QUANTILES, (len(batch),))
                target = rew + GAMMA * z_next[torch.arange(len(batch)), k]
            q_sa = net(obs).reshape(len(batch), N_ACTIONS, N_QUANTILES)[
                torch.arange(len(batch)), act]
            taus_t = torch.tensor(TAUS, dtype=torch.float32)
            losses = torch.stack([
                pinball(q_sa[i], target[i], taus_t)
                for i in range(len(batch))])
            loss = losses.mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            if steps % 300 == 0:
                tgt.load_state_dict(net.state_dict())
            s = s2
    return net


def cvar_from_quantiles(thetas, taus, alpha=0.05):
    """CVaR_alpha of the return distribution represented by quantile
    heads: tail mean over heads with tau < alpha (lower tail = risk)."""
    mask = taus < alpha
    if not mask.any():
        mask = taus <= taus.min()
    return float(np.mean(thetas[mask]))


def main():
    t0 = time.time()
    net = train_qrdqn()
    print(f"QR-DQN trained in {time.time()-t0:.0f}s "
          f"({EPISODES} episodes, N={N_QUANTILES} heads)\n")

    print("per-state quantile heads (state s=0, greedy action):")
    with torch.no_grad():
        z = net(torch.tensor(one_hot(0)).unsqueeze(0)).reshape(
            N_ACTIONS, N_QUANTILES).numpy()
    best_a = int(z.mean(axis=1).argmax())
    heads = z[best_a]
    print(f"  greedy action = {best_a}")
    for tau, th in zip(TAUS, heads):
        ptp = heads.max() - heads.min()
        bar = "#" * int((th - heads.min()) / max(1e-9, ptp) * 30)
        print(f"  tau={tau:.2f}: {th:8.2f}  {bar}")

    # honesty check: empirical distribution from greedy rollouts
    env = InventoryEnv(99)
    rets = []
    for _ in range(3000):
        s, G, g, guard = env.reset(), 0.0, 1.0, 0
        while guard < 50:
            with torch.no_grad():
                a = int(net(torch.tensor(
                    one_hot(s)).unsqueeze(0)).reshape(
                        N_ACTIONS, N_QUANTILES).mean(1).argmax())
            a = min(max(a, 0), MAX_INV - s)
            s, r = env.step(s, a)
            G += g * r
            g *= GAMMA
            guard += 1
        rets.append(G)
    rets = np.array(rets)
    emp_mean = rets.mean()
    emp_cvar = rets[rets <= np.quantile(rets, 0.05)].mean()

    print("\nread-outs (state 0, greedy policy) vs empirical rollouts:")
    var5 = heads[TAUS <= 0.06].max()   # lowest head = tau 0.056 (N=9)
    print(f"  {'metric':8s} {'QR-DQN heads':>13s} {'empirical':>10s}")
    print(f"  {'mean':8s} {heads.mean():13.2f} {emp_mean:10.2f}")
    print(f"  {'VaR 5%':8s} {var5:13.2f} "
          f"{np.quantile(rets, 0.05):10.2f}")
    print(f"  {'CVaR 5%':8s} {cvar_from_quantiles(heads, TAUS):13.2f} "
          f"{emp_cvar:10.2f}")
    print("\n(no re-training per risk level — the distribution IS the model,"
          "\n and every risk measure is a lookup on it, as promised in §3.5)")


import time  # noqa: E402

if __name__ == "__main__":
    main()
