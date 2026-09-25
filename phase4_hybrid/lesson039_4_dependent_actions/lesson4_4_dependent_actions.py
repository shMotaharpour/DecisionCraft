"""Lesson 4.4 — Dependent actions: masking in DQN, domains in CP-SAT,
masked softmax in policy gradients.

Demo env: a tiny shop. State = (level 0..3, wallet 0..3 items affordable).
Actions: 0=WORK (earn 1 coin), 1..4=BUY item i (legal iff wallet >= price
of item i AND wallet stays >= 0; item prices [1,2,3,4]).
Reward: WORK = +0.2; BUY i = value_i if affordable (values [3,5,8,10] -
price) else illegal.
Goal: learn to work, then buy the best affordable item — with and without
masking, comparing sample efficiency and Q distortion.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

PRICES = np.array([1, 2, 3, 4])
VALUES = np.array([3.0, 5.0, 8.0, 10.0])
N_ACTIONS = 5  # 0 = work, 1..4 = buy item i
MAX_WALLET = 6


def legal_mask(wallet):
    m = np.zeros(N_ACTIONS, dtype=bool)
    m[0] = True
    affordable = wallet - PRICES >= 0
    m[1:] = affordable
    return m


def step(wallet, a):
    """Returns (next_wallet, reward, legal_was)."""
    if a == 0:
        return min(MAX_WALLET, wallet + 1), 0.2, True
    if a - 1 < len(PRICES) and wallet >= PRICES[a - 1]:
        profit = VALUES[a - 1] - PRICES[a - 1]
        return wallet - PRICES[a - 1], profit, True
    return wallet, -1.0, False  # illegal: error penalty path


def train_dqn(mask: bool, seed=7, episodes=300):
    torch.manual_seed(seed); random.seed(seed)
    net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, N_ACTIONS))
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    buf = deque(maxlen=20_000)
    rng = np.random.default_rng(seed)
    illegal_hits, eps_start = 0, 1.0
    GAMMA = 0.9

    def one_hot(w):
        return np.array([w / MAX_WALLET, 1.0 if mask else 0.0], dtype=np.float32)

    for ep in range(episodes):
        eps = max(0.05, eps_start * (1 - ep / 200))
        wallet = 0
        for t in range(30):
            mask_t = legal_mask(wallet)
            if rng.random() < eps:
                valid = np.where(mask_t)[0] if mask else np.arange(N_ACTIONS)
                a = int(rng.choice(valid))
            else:
                with torch.no_grad():
                    qs = net(torch.tensor(one_hot(wallet))).numpy()
                    if mask:
                        qs[~mask_t] = -1e9
                    a = int(qs.argmax())
            w2, r, legal = step(wallet, a)
            if not legal:
                illegal_hits += 1
            buf.append((one_hot(wallet), a, r, one_hot(w2), mask_t if mask else np.ones(N_ACTIONS, bool)))
            wallet = w2
            if len(buf) >= 64:
                B = random.sample(buf, 64)
                S = torch.tensor(np.stack([b[0] for b in B]))
                A = torch.tensor([b[1] for b in B])
                R = torch.tensor([b[2] for b in B])
                S2 = torch.tensor(np.stack([b[3] for b in B]))
                M2 = torch.tensor(np.stack([b[4] for b in B]))
                with torch.no_grad():
                    q2 = net(S2).clone()
                    q2[~M2] = -1e9  # mask in the TARGET as well (the classic bug spot)
                    y = R + GAMMA * q2.max(dim=1).values
                pred = net(S).gather(1, A.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)
                opt.zero_grad(); loss.backward(); opt.step()
    return net, illegal_hits


def policy_of(net, mask):
    pol = {}
    for w in range(MAX_WALLET + 1):
        with torch.no_grad():
            qs = net(torch.tensor([w / MAX_WALLET, 1.0 if mask else 0.0])).numpy()
        m = legal_mask(w)
        if mask:
            qs[~m] = -1e9
        pol[w] = int(qs.argmax())
    return pol


def demo_dqn():
    print("=== Demo 1: DQN masking vs error-penalty (300 episodes each) ===")
    net_m, hits_m = train_dqn(mask=True)
    net_u, hits_u = train_dqn(mask=False)
    print(f"masked   : illegal actions taken during training = {hits_m:5d}")
    print(f"unmasked : illegal actions taken during training = {hits_u:5d}")
    print(f"masked   policy (wallet 0..6): {[policy_of(net_m, True)[w] for w in range(MAX_WALLET + 1)]}")
    print(f"unmasked policy (wallet 0..6): {[policy_of(net_u, False)[w] for w in range(MAX_WALLET + 1)]}")
    print("  (0=work, 1..4=buy item; sensible: work until rich enough, then buy best item)")


def demo_cpsat():
    print("\n=== Demo 2: CP-SAT — action domains from state ===")
    from ortools.sat.python import cp_model
    wallet, level = 3, 2
    m = cp_model.CpModel()
    # which item to buy this step: domain excludes unaffordable items
    allowed = [i + 1 for i in range(4) if PRICES[i] <= wallet] + [0]  # 0 = work
    a = m.NewIntVarFromDomain(cp_model.Domain.FromValues(allowed), "action")
    # prerequisite logic as constraints: buy i requires wallet >= price_i
    for i in range(4):
        b = m.NewBoolVar(f"buy_{i}")
        m.Add(a == i + 1).OnlyEnforceIf(b)
        m.Add(wallet >= PRICES[i]).OnlyEnforceIf(b)
        m.Add(a != i + 1).OnlyEnforceIf(b.Not())
    print(f"  state: wallet={wallet}, level={level}")
    print(f"  legal action domain: {allowed}  (0=work; item actions filtered by wallet)")
    print("  -> per-step domains make illegal actions structurally impossible")


def demo_ppo_mask():
    print("\n=== Demo 3: masked softmax (policy gradient style) ===")
    logits = torch.tensor([1.0, 2.0, 0.5, 3.0, 2.5])  # raw network output
    wallet = 2
    m = legal_mask(wallet)
    masked_logits = logits.clone()
    masked_logits[~torch.tensor(m)] = -1e9
    dist_raw = torch.distributions.Categorical(logits=logits)
    dist_mask = torch.distributions.Categorical(logits=masked_logits)
    print(f"  wallet={wallet}, legal mask = {m.tolist()}")
    print(f"  raw   probs : {np.round(dist_raw.probs.numpy(), 3)}")
    print(f"  masked probs: {np.round(dist_mask.probs.numpy(), 3)}  (illegal -> 0)")
    print("  log_prob of sampled (feasible) actions stays exact; entropy fine")


if __name__ == "__main__":
    demo_dqn()
    demo_cpsat()
    demo_ppo_mask()
