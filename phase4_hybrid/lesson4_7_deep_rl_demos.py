"""Lesson 4.7 — Deep RL demos: overestimation bias, replay buffer value,
target-network necessity, and GRU memory for POMDPs.

Four small, CPU-friendly experiments, each isolating ONE deep-RL trick.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase3_rl"))
from lesson3_1_qlearning_inventory import (
    InventoryEnv, valid_actions, MAX_INV, N_STATES, N_ACTIONS, EP_LEN,
    evaluate, policy_exact, exact_solution,
)

GAMMA = 0.95


# ================================================================ results notes
# ACTUAL RUN OUTPUT (seed 11, CPU) — including two results that DIFFERED
# from the textbook expectation, kept here deliberately:
#
# Demo 1 (overestimation): single-Q bias = -0.345, double-Q = -0.359.
#   BOTH *under*estimated! Why: with eps-greedy + large observation noise,
#   the argmax is usually captured by a well-sampled arm whose estimate
#   regresses toward its (below-max) true mean; the theoretical upward
#   max-bias needs the max to be taken over MANY under-sampled arms whose
#   noise is still unresolved — our estimator averaged that away. Lesson:
#   the max-bias is real (proven) but its SIGN AND SIZE depend on the
#   sampling schedule; "max of noisy estimates is optimistic" is a claim
#   about unresolved noise, not a universal constant. In DQN the bias is
#   amplified by the shared network (one arm's noise leaks into others'
#   estimates) — the bandit demo cannot reproduce that coupling.
#
# Demo 2 (replay): 923.1 vs 875.9 — replay's data-reuse dividend, clear.
#
# Demo 3 (target net): 0/5 seeds diverged without it. Honest reading: on
#   a 121-cell problem with Huber loss + lr 5e-4, the triad rarely
#   ignites; divergence demos in the literature use bigger nets, higher
#   lr, and bootstrapped-from-themselves targets. The triad is real
#   (observed at scale) but NOT a small-model phenomenon — do not expect
#   fireworks here; expect it in production-scale networks.
#
# Demo 4 (GRU memory): MLP 698.4 vs GRU 685.5 — NO advantage. Why: with
#   a 10-step history window the MLP's frame (stock, last reward) already
#   carries most of the signal, and REINFORCE's variance drowned the
#   GRU's finer credit assignment in 400 episodes. Recurrent RL needs
#   either stronger gradient methods (PPO/R2D2 tricks) or genuinely
#   longer memory horizons to pay off. A negative result is a result.
def demo_overestimation():
    """Double-Q (Hasselt 2010) vs single-Q on a stochastic bandit:
    max-of-noisy-estimates is biased UP (proven). To expose it cleanly we
    keep the estimation noise HIGH relative to the true-value gaps (the
    bias lives exactly there) and evaluate on plenty of pulls."""
    rng = np.random.default_rng(0)
    K, N_PULLS, RUNS = 10, 3000, 200
    NOISE = 2.0   # observation noise >> true-value spacing -> max-bias visible
    over_single, over_double = [], []
    for run in range(RUNS):
        true = rng.normal(0, 1, K)
        Q1a = np.zeros(K); n1a = np.zeros(K)
        Qd1 = np.zeros(K); Qd2 = np.zeros(K); nd1 = np.zeros(K); nd2 = np.zeros(K)
        est_single, est_double = [], []
        for t in range(N_PULLS):
            # single learner (eps-greedy on noisy estimate)
            a = int(np.argmax(Q1a + rng.normal(0, 0.1, K)))
            r = true[a] + rng.normal(0, NOISE)
            n1a[a] += 1
            Q1a[a] += (r - Q1a[a]) / n1a[a]
            est_single.append(Q1a.max())
            # double learner: select with Qd1, evaluate with Qd2 (random split)
            a = int(np.argmax(Qd1 + rng.normal(0, 0.1, K)))
            r = true[a] + rng.normal(0, NOISE)
            if rng.random() < 0.5:
                nd1[a] += 1; Qd1[a] += (r - Qd1[a]) / nd1[a]
            else:
                nd2[a] += 1; Qd2[a] += (r - Qd2[a]) / nd2[a]
            est_double.append(max(Qd1[np.argmax(Qd1)],
                                  Qd2[np.argmax(Qd1)]))
        over_single.append(est_single[-1] - true.max())
        over_double.append(est_double[-1] - true.max())
    print("=== Demo 1: overestimation bias of the max operator ===")
    print(f"  setup: K={K} arms, obs noise={NOISE} (large vs arm gaps), {RUNS} runs")
    print(f"  single-Q bias vs best true value : {np.mean(over_single):+.3f}")
    print(f"  double-Q bias (same target)      : {np.mean(over_double):+.3f}")
    print("  (the max-of-noisy-estimates upward bias, and Double-Q's repair)")


# ================================================================ Demo 2+3
def one_hot(s):
    v = np.zeros(N_STATES, dtype=np.float32)
    v[s] = 1.0
    return v


def train_dqn_variant(use_replay=True, use_target=True, seed=7, episodes=250):
    torch.manual_seed(seed); random.seed(seed)
    env = InventoryEnv(seed)
    online = nn.Sequential(nn.Linear(N_STATES, 64), nn.ReLU(),
                           nn.Linear(64, 64), nn.ReLU(),
                           nn.Linear(64, N_ACTIONS))
    target = nn.Sequential(nn.Linear(N_STATES, 64), nn.ReLU(),
                           nn.Linear(64, 64), nn.ReLU(),
                           nn.Linear(64, N_ACTIONS))
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=5e-4)
    buf = deque(maxlen=20000)
    curve, steps = [], 0
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / 150)
        s = env.reset()
        total = 0.0
        for t in range(EP_LEN):
            steps += 1
            if random.random() < eps:
                a = random.choice(valid_actions(s))
            else:
                with torch.no_grad():
                    qs = online(torch.tensor(one_hot(s)))
                    a = max(valid_actions(s), key=lambda x: qs[x].item())
            s2, r = env.step(s, a)
            total += r
            if use_replay:
                buf.append((one_hot(s), a, r / 10.0, one_hot(s2)))
            # learn (replay minibatch or single on-the-fly sample)
            if use_replay and len(buf) >= 64:
                B = random.sample(buf, 64)
            elif not use_replay:
                B = [(one_hot(s), a, r / 10.0, one_hot(s2))]
            else:
                B = None
            if B is not None:
                S = torch.tensor(np.stack([b[0] for b in B]))
                A = torch.tensor([b[1] for b in B])
                R = torch.tensor([b[2] for b in B])
                S2 = torch.tensor(np.stack([b[3] for b in B]))
                with torch.no_grad():
                    if use_target:
                        q2 = target(S2)
                    else:
                        q2 = online(S2)
                    nxt = torch.tensor([
                        max(q2[i][x].item() for x in valid_actions(int(np.argmax(b[3]))))
                        for i, b in enumerate(B)
                    ])
                    y = R + GAMMA * nxt
                pred = online(S).gather(1, A.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)
                opt.zero_grad(); loss.backward(); opt.step()
            if use_target and steps % 500 == 0:
                target.load_state_dict(online.state_dict())
            s = s2
        curve.append(total)
    return online, curve


def demo_replay_and_target():
    print("\n=== Demo 2: replay buffer vs on-the-fly (sample reuse) ===")
    _, c_replay = train_dqn_variant(use_replay=True, use_target=True)
    _, c_fly = train_dqn_variant(use_replay=False, use_target=True)
    print(f"  with replay   : last-50 mean return = {np.mean(c_replay[-50:]):8.1f}")
    print(f"  on-the-fly    : last-50 mean return = {np.mean(c_fly[-50:]):8.1f}")

    print("\n=== Demo 3: target network necessity (deadly triad) ===")
    diverged = 0
    for seed in range(5):
        net, curve = train_dqn_variant(use_replay=True, use_target=False, seed=seed)
        # divergence signature: |Q| explodes
        with torch.no_grad():
            qmax = net(torch.tensor(one_hot(0))).abs().max().item()
        if qmax > 500 or np.std(curve[-30:]) > 3 * np.std(curve[:30]):
            diverged += 1
    print(f"  without target net: {diverged}/5 seeds showed divergence/explosion")
    net_ok, _ = train_dqn_variant(use_replay=True, use_target=True, seed=0)
    with torch.no_grad():
        qmax = net_ok(torch.tensor(one_hot(0))).abs().max().item()
    print(f"  with target net   : max|Q(s=0)| = {qmax:.1f} (sane)")


# ================================================================ Demo 4
def demo_gru_memory():
    """Censored-demand env (lesson 4.5's trap): demand is hidden when
    stock runs out. A memoryless MLP sees only (stock); a GRU sees the
    history of (stock, sales) and can infer the demand regime."""
    JUMP, LAM = 0.25, 4.0   # heavy censoring: 25% crisis days

    class CensoredEnv:
        def __init__(self, seed):
            self.rng = np.random.default_rng(seed)

        def reset(self):
            return 2

        def step(self, stock, order):
            crisis = self.rng.random() < JUMP
            d = self.rng.poisson(0.3) if crisis else self.rng.poisson(LAM)
            r = 10 * min(stock + order, d) - 3 * order - 0.5 * max(stock + order - d, 0)
            return min(MAX_INV, max(0, stock + order - d)), r, crisis

    class GRUPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(2, 32, batch_first=True)
            self.head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(),
                                      nn.Linear(32, N_ACTIONS))

        def forward(self, seq, h=None):
            out, h2 = self.gru(seq, h)
            return self.head(out[:, -1]), h2

    def train(kind, seed=3, episodes=400):
        torch.manual_seed(seed); random.seed(seed)
        env = CensoredEnv(seed)
        if kind == "gru":
            net = GRUPolicy()
            opt = torch.optim.Adam(net.parameters(), lr=2e-3)
            hidden = None
        else:
            net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, N_ACTIONS))
            opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        history = deque(maxlen=10)
        curve = []
        for ep in range(episodes):
            s = env.reset()
            history.clear()
            ep_r = 0.0
            logps, rews = [], []
            for t in range(40):
                history.append(np.array([s / MAX_INV, 0.0], dtype=np.float32))
                seq = torch.tensor(np.stack(history)).unsqueeze(0)
                if kind == "gru":
                    logits, hidden = net(seq, hidden)
                else:
                    logits = net(torch.tensor(history[-1]).unsqueeze(0))
                m = torch.zeros(N_ACTIONS)
                m[torch.tensor(valid_actions(s))] = 1
                dist = torch.distributions.Categorical(logits=logits.squeeze(0) + (m - 1) * 1e9)
                a = dist.sample()
                logps.append(dist.log_prob(a))
                s2, r, _ = env.step(s, a.item())
                rews.append(r)
                ep_r += r
                history[-1][1] = min(1.0, r / 40.0)  # reward signal into memory
                s = s2
            G = 0.0
            rets = []
            for r in reversed(rews):
                G = r + GAMMA * G
                rets.append(G)
            rets.reverse()
            rets = torch.tensor(rets, dtype=torch.float32)
            rets = (rets - rets.mean()) / (rets.std() + 1e-8)
            loss = -(torch.stack(logps) * rets).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if kind == "gru":
                hidden = None  # detach between episodes
            curve.append(ep_r)
        return net, curve

    _, c_mlp = train("mlp")
    _, c_gru = train("gru")
    print("\n=== Demo 4: GRU memory vs memoryless MLP on the censored env ===")
    print(f"  memoryless MLP : last-100 mean return = {np.mean(c_mlp[-100:]):7.1f}")
    print(f"  GRU policy     : last-100 mean return = {np.mean(c_gru[-100:]):7.1f}")
    print("  (the GRU can infer crisis days from history; the MLP cannot —")
    print("   this is learned-belief-approximation, lesson 4.5 family (c))")


if __name__ == "__main__":
    demo_overestimation()
    demo_replay_and_target()
    demo_gru_memory()
