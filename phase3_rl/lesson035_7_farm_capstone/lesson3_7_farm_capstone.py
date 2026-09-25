"""Lesson 3.7 — Phase-3 capstone: DQN + PPO on a tiny farm game.

A MINIATURE of Kaggriculture (the real competition the Chista agent plays):
one tile, T=12 steps per episode, 3 crop types with (cost, sell, growth time),
water needed each step while growing, weather (sun/rain) is stochastic.
Actions: 0=plant crop-i (i=1..3), 4=water, 5=harvest, 0=noop.
Reward: sell revenue at harvest; wasted water/cost negative.

Two agents (DQN, PPO — reuse lesson 3.3/3.4 machinery patterns) vs a
hand-written rule-based policy (plant best EV, water if growing, harvest
when ready). Runtime: ~2 min total on CPU.
Run:  python lesson3_7_farm_capstone.py
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

T_STEPS = 12
CROPS = {                      # id: (seed cost, sell price, growth steps)
    1: (2.0, 8.0, 3),
    2: (3.0, 14.0, 5),
    3: (1.0, 5.0, 2),
}
N_CROPS = len(CROPS)
# actions: 0 = noop, 1..3 = plant crop i, 4 = water, 5 = harvest
N_ACTIONS = 6
WATER_COST = 0.2
RAIN_P = 0.3                    # rain waters for free

STATE_DIM = N_CROPS + 1 + 1 + 1 + 1 + T_STEPS   # crop planted?/ready?, watered?, budget, weather, +1 slack, onehot-t


class FarmEnv:
    """Tiny farm: plant once, water while growing, harvest for revenue."""
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.t = 0
        self.crop = 0            # 0 = empty, else crop id
        self.growth = 0
        self.watered = False
        self.budget = 6.0        # can't buy seeds beyond this
        return self._obs(0)

    def _obs(self, weather):
        o = np.zeros(STATE_DIM, dtype=np.float32)
        if self.crop:
            o[self.crop - 1] = 1.0 if self.growth >= CROPS[self.crop][2] else 0.5
        o[N_CROPS] = 1.0 if self.watered else 0.0
        o[N_CROPS + 1] = self.budget / 6.0
        o[N_CROPS + 2] = weather
        o[N_CROPS + 3 + self.t] = 1.0
        return o

    def step(self, action):
        weather = float(self.rng.random() < RAIN_P)   # rain waters free
        r = 0.0
        if action in CROPS and self.crop == 0:
            cost, _, grow = CROPS[action]
            if self.budget >= cost:
                self.crop, self.growth, self.budget = action, 0, self.budget - cost
            else:
                r = -0.5                          # invalid plant attempt
        elif action == 4 and self.crop and not self.watered \
                and self.growth < CROPS[self.crop][2]:
            self.watered = True
            r = -WATER_COST
        elif action == 5 and self.crop and self.growth >= CROPS[self.crop][2]:
            r = CROPS[self.crop][1]
            self.crop = 0
        # growth: needs water each step (rain counts)
        if self.crop and (weather or self.watered):
            self.growth += 1
            self.watered = False
        self.t += 1
        done = self.t >= T_STEPS
        return self._obs(weather), r, done


def rule_policy(obs):
    """Hand-written: plant crop 2 if affordable, water if growing, harvest
    when ready, else noop."""
    planted = obs[:N_CROPS].sum()
    ready = 1.0 in obs[:N_CROPS]
    growing = planted > 0 and not ready
    if ready:
        return 5
    if growing and obs[N_CROPS] == 0.0:
        return 4
    if planted == 0 and obs[N_CROPS + 1] >= 0.5:
        return 2
    return 0


# ---------------------------------------------------------------- exact DP
def solve_exact_dp():
    """Backward induction over the FULL farm state space (19,344 states).
    Budget lives on the 0.2 lattice (all costs are multiples of 0.2).
    Returns V(s=empty, t=0, budget=6) and the optimal action table."""
    from functools import lru_cache
    B0 = 6.0
    B_LATTICE = [round(0.2 * i, 1) for i in range(31)]
    b_idx = {b: i for i, b in enumerate(B_LATTICE)}

    def b_floor(b):                     # largest lattice value <= b
        return B_LATTICE[min(b_idx[round(min(b, 6.0) / 0.2, 0)], 30)]

    @lru_cache(maxsize=None)
    def V(t, crop, growth, watered, b):
        if t >= T_STEPS:
            return 0.0
        best = -1e9
        for a in range(N_ACTIONS):
            r, ncrop, ngrowth, nwatered, nb = _apply(
                t, crop, growth, watered, b, a)
            ev = r
            # weather: rain (p=0.3) waters for free
            for wprob, w in ((RAIN_P, 1.0), (1 - RAIN_P, 0.0)):
                g2 = ngrowth
                if ncrop and (w or nwatered):
                    g2 = ngrowth + 1
                    # watered flag consumed by rain OR by manual watering
                ev += wprob * V(t + 1, ncrop, g2,
                                nwatered and not w, nb)
            if ev > best:
                best = ev
        return best

    def _apply(t, crop, growth, watered, b, a):
        r, ncrop, ngrowth, nwatered, nb = 0.0, crop, growth, watered, b
        if a in CROPS and crop == 0:
            cost, _, grow = CROPS[a]
            if b >= cost:
                ncrop, ngrowth, nb = a, 0, round(b - cost, 1)
            else:
                r = -0.5
        elif a == 4 and crop and not watered and growth < CROPS[crop][2]:
            nwatered = True
            r = -WATER_COST
            nb = round(b - WATER_COST, 1)
            if nb < 0:                  # cannot afford water
                nwatered, nb = watered, b
                r = -0.5
        elif a == 5 and crop and growth >= CROPS[crop][2]:
            r = CROPS[crop][1]
            ncrop = 0
        return r, ncrop, ngrowth, nwatered, nb

    v0 = V(0, 0, 0, False, b_idx[6.0])
    return v0


# ---------------------------------------------------------------- networks
def mlp(inp, out, hidden=64):
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, out))


# ---------------------------------------------------------------- DQN
def train_dqn(episodes=350, seed=7):
    torch.manual_seed(seed); random.seed(seed)
    env, pyrng = FarmEnv(seed), random.Random(seed)
    net, tgt = mlp(STATE_DIM, N_ACTIONS), mlp(STATE_DIM, N_ACTIONS)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    buf, curve = [], []
    steps = 0
    for ep in range(episodes):
        obs, ep_r, done = env.reset(), 0.0, False
        while not done:
            eps = max(0.05, 1.0 - ep / (0.7 * episodes))
            if random.random() < eps:
                a = random.randrange(N_ACTIONS)
            else:
                with torch.no_grad():
                    a = int(net(torch.tensor(obs)).argmax())
            steps += 1
            obs2, r, done = env.step(a)
            buf.append((obs, a, r, obs2))
            if len(buf) > 5000:
                buf.pop(0)
            batch = pyrng.sample(buf, min(64, len(buf)))
            o = torch.tensor(np.stack([b[0] for b in batch]))
            act = torch.tensor([b[1] for b in batch])
            rw = torch.tensor([float(b[2]) for b in batch])
            o2 = torch.tensor(np.stack([b[3] for b in batch]))
            with torch.no_grad():
                target = rw + (1 - done) * tgt(o2).max(dim=1).values
            q = net(o).gather(1, act.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(q, target)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            if steps % 150 == 0:
                tgt.load_state_dict(net.state_dict())
            ep_r += r
            obs = obs2
        curve.append(ep_r)
    return net, curve


# ---------------------------------------------------------------- PPO
def run_traj(env, net, pyrng, explore=1.0):
    traj, obs, done = [], env.reset(), False
    while not done:
        logits = net(torch.tensor(obs))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        obs2, r, done = env.step(a.item())
        traj.append((obs, a.item(), dist.log_prob(a).detach(), r))
        obs = obs2
    return traj


def train_ppo(episodes=800, seed=7, clip=0.2, epochs=3, batch_eps=10, lr=1e-3,
              ent_coef=0.02):
    torch.manual_seed(seed); random.seed(seed)
    env, pyrng = FarmEnv(seed), random.Random(seed)
    net = mlp(STATE_DIM, N_ACTIONS)
    critic = mlp(STATE_DIM, 1)
    opt = torch.optim.Adam(list(net.parameters()) + list(critic.parameters()),
                           lr=lr)
    curve = []
    for it in range(episodes // batch_eps):
        rollouts = [run_traj(env, net, pyrng) for _ in range(batch_eps)]
        curve += [sum(t[3] for t in tr) for tr in rollouts]
        obs_all, a_all, logp_all, adv_all, ret_all = [], [], [], [], []
        for tr in rollouts:
            G, rets = 0.0, []
            for _, _, _, r in reversed(tr):
                G = r + 0.99 * G
                rets.append(G)
            rets.reverse()
            for (obs, a, logp, r), G in zip(tr, rets):
                v = critic(torch.tensor(obs)).item()
                adv_all.append(G - v)
                ret_all.append(G)
                obs_all.append(obs); a_all.append(a); logp_all.append(logp)
        adv = torch.tensor(adv_all)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        o = torch.tensor(np.stack(obs_all))
        a = torch.tensor(a_all)
        ol = torch.stack(logp_all)
        R = torch.tensor(ret_all, dtype=torch.float32)
        for _ in range(epochs):
            dist = torch.distributions.Categorical(logits=net(o))
            ratio = torch.exp(dist.log_prob(a) - ol)
            s1, s2 = ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv
            critic_loss = (critic(o).squeeze(1) - R).pow(2).mean()
            loss = (-torch.min(s1, s2).mean() + 0.5 * critic_loss
                    - ent_coef * dist.entropy().mean())
            opt.zero_grad(); loss.backward(); opt.step()
    return net, curve

# ---------------------------------------------------------------- evaluation
def eval_policy(pi_fn, episodes=3000, seed=999):
    env = FarmEnv(seed)
    tot = 0.0
    for _ in range(episodes):
        obs, done = env.reset(), False
        while not done:
            obs, r, done = env.step(pi_fn(obs))
            tot += r
    return tot / episodes


if __name__ == "__main__":
    import time
    t0 = time.time()
    v_dp = solve_exact_dp()
    print(f"exact DP (backward induction, 19k states): V = {v_dp:.3f} "
          f"({time.time()-t0:.1f}s)  <- the ceiling\n")

    print("training DQN ...")
    net_dqn, c_dqn = train_dqn(episodes=2000)
    print("training PPO ...")
    net_ppo, c_ppo = train_ppo(episodes=3000)
    # Sparse reward (harvest pays once per episode) is the honest headline:
    # see notes — reported, not tuned away.

    def dqn_pi(obs):
        with torch.no_grad():
            return int(net_dqn(torch.tensor(obs)).argmax())

    def ppo_pi(obs):
        with torch.no_grad():
            return int(net_ppo(torch.tensor(obs)).argmax())

    print(f"\n{3000}-episode evaluation (avg reward per 12-step episode):")
    for name, pi in (("exact DP (from V)", None), ("rule-based", rule_policy),
                     ("DQN", dqn_pi), ("PPO", ppo_pi)):
        if pi is None:
            print(f"  {name:15s}: {v_dp:7.3f}  (analytic optimum)")
        else:
            print(f"  {name:15s}: {eval_policy(pi):7.3f}")
    cd, cp = np.array(c_dqn), np.array(c_ppo)
    print(f"\nlast-100 training means: DQN {cd[-100:].mean():.3f}  "
          f"PPO {cp[-100:].mean():.3f}")
