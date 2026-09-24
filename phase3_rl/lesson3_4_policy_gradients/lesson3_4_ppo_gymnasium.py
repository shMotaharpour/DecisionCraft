"""Lesson 3.4 — REINFORCE vs Actor-Critic vs PPO on a Gymnasium inventory env.

A Gymnasium-native environment wrapping the familiar inventory problem,
then three policy-gradient agents trained from scratch (no SB3), all
graded against the exact (s,S) policy from Value Iteration.
"""
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from lesson3_1_qlearning_inventory import (
    InventoryEnv, exact_solution, MAX_INV, N_STATES, N_ACTIONS,
    EP_LEN, evaluate, policy_exact,
)

GAMMA = 0.95


# ---------------------------------------------------------------- Gym env
class InventoryGymEnv(gym.Env):
    """Gymnasium protocol over the same inventory dynamics."""
    metadata = {"render_modes": []}

    def __init__(self, seed=0):
        super().__init__()
        self.inner = InventoryEnv(seed)
        self.action_space = spaces.Discrete(N_STATES)      # order 0..10
        self.observation_space = spaces.Box(0.0, 1.0, shape=(N_STATES,), dtype=np.float32)

    def _obs(self, s):
        v = np.zeros(N_STATES, dtype=np.float32)
        v[int(s)] = 1.0
        return v

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs(self.inner.reset()), {}

    def step(self, action):
        s2, r = self.inner.step(int(np.argmax(self._obs_last)), action)
        return self._obs(s2), r / 10.0, False, False, {}   # never terminal in training

    # small stateful helper so step() knows the current one-hot
    _obs_last = np.zeros(N_STATES, dtype=np.float32)

    def step_fixed(self, s, action):
        """Cleaner API used below: returns the state INDEX (not one-hot).
        Action is clipped to the feasible order range (0..MAX_INV-s) —
        policy-gradient methods sample from the full softmax, so the env
        must keep invalid actions from corrupting the state."""
        action = int(np.clip(action, 0, MAX_INV - s))
        s2, r = self.inner.step(s, action)
        return s2, float(r) / 10.0


# ---------------------------------------------------------------- networks
def mlp(inp, out, hidden=64):
    return nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, out))


def run_episode(env, policy_fn, ep_len=EP_LEN):
    """Collect a trajectory with the state-index (not one-hot) for logic."""
    s = env.inner.reset()
    traj = []
    for t in range(ep_len):
        obs = env._obs(s)
        a, logp = policy_fn(obs)
        s2, r = env.step_fixed(s, a)
        traj.append((obs, a, logp, r, s))
        s = s2
    return traj


def discounted_returns(rewards, gamma=GAMMA):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return list(reversed(out))


# ---------------------------------------------------------------- agents
def train_reinforce(seed=7, episodes=1500, lr=3e-3):
    torch.manual_seed(seed); random.seed(seed)
    env = InventoryGymEnv(seed)
    net = mlp(N_STATES, N_ACTIONS)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    curve = []
    for ep in range(episodes):
        traj = run_episode(env, lambda obs: _categorical(net, obs))
        returns = discounted_returns([t[3] for t in traj])
        loss = 0.0
        for (obs, a, logp, r, _), G in zip(traj, returns):
            loss = loss - logp * G
        loss = loss / len(traj)
        opt.zero_grad(); loss.backward(); opt.step()
        curve.append(sum(t[3] for t in traj))
    return net, curve


def _categorical(net, obs):
    logits = net(torch.tensor(obs))
    dist = torch.distributions.Categorical(logits=logits)
    a = dist.sample()
    return a.item(), dist.log_prob(a)


def train_actor_critic(seed=7, episodes=1500, lr=3e-3):
    torch.manual_seed(seed); random.seed(seed)
    env = InventoryGymEnv(seed)
    actor, critic = mlp(N_STATES, N_ACTIONS), mlp(N_STATES, 1)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    curve = []
    for ep in range(episodes):
        s = env.inner.reset()
        ep_r = 0.0
        for t in range(EP_LEN):
            obs = env._obs(s)
            logits = actor(torch.tensor(obs))
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            s2, r = env.step_fixed(s, a.item())
            ep_r += r
            # bootstrapped advantage: r + gamma V(s') - V(s)
            v_s = critic(torch.tensor(obs))
            v_s2 = critic(torch.tensor(env._obs(s2)))
            advantage = r + GAMMA * v_s2 - v_s
            logp = dist.log_prob(a)
            actor_loss = -logp * advantage.detach()
            critic_loss = advantage.pow(2)
            loss = actor_loss + 0.5 * critic_loss
            opt.zero_grad(); loss.backward(); opt.step()
            s = s2
        curve.append(ep_r)
    return actor, critic, curve


def train_ppo(seed=7, episodes=800, lr=3e-3, clip=0.2, epochs=4, batch_eps=8):
    torch.manual_seed(seed); random.seed(seed)
    env = InventoryGymEnv(seed)
    actor, critic = mlp(N_STATES, N_ACTIONS), mlp(N_STATES, 1)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    curve = []
    for it in range(episodes // batch_eps):
        # ---- collect a batch of rollouts with the CURRENT policy
        rollouts = []
        for _ in range(batch_eps):
            traj = run_episode(env, lambda obs: _categorical(actor, obs))
            returns = discounted_returns([t[3] for t in traj])
            rollouts.append((traj, returns))
            curve.append(sum(t[3] for t in traj))
        # ---- advantages (returns - critic value), normalized
        obs_all, a_all, old_logp_all, adv_all = [], [], [], []
        for traj, returns in rollouts:
            for (obs, a, logp, r, _), G in zip(traj, returns):
                v = critic(torch.tensor(obs)).item()
                adv_all.append(G - v)
                obs_all.append(obs); a_all.append(a); old_logp_all.append(logp)
        adv = torch.tensor(adv_all)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # normalization
        obs_T = torch.tensor(np.stack(obs_all))
        a_T = torch.tensor(a_all)
        old_logp_T = torch.stack(old_logp_all).detach()

        # ---- PPO epochs: multiple passes with the clip trust region
        for _ in range(epochs):
            logits = actor(obs_T)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(a_T)
            ratio = torch.exp(logp - old_logp_T)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
            actor_loss = -torch.min(surr1, surr2).mean()
            values = critic(obs_T).squeeze(1)
            targets = torch.tensor([G for traj, rets in rollouts
                                    for G in rets])
            critic_loss = (values - targets).pow(2).mean()
            ent = dist.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * ent
            opt.zero_grad(); loss.backward(); opt.step()
    return actor, critic, curve


def policy_from_net(net):
    def pi(s):
        with torch.no_grad():
            logits = net(torch.tensor(InventoryGymEnv(0)._obs(s)))
        return int(logits.argmax())
    return np.array([pi(s) for s in range(N_STATES)])


if __name__ == "__main__":
    V_exact, policy_exact = exact_solution()
    exact_perf = evaluate(policy_exact)

    print("training REINFORCE ...")
    net_r, c_r = train_reinforce()
    print("training Actor-Critic ...")
    net_ac, net_c, c_ac = train_actor_critic()
    print("training PPO ...")
    net_p, net_pc, c_p = train_ppo()

    print("\nfinal greedy policy:")
    print(f"  exact DP : {policy_exact}")
    print(f"  REINFORCE: {policy_from_net(net_r)}")
    print(f"  A2C      : {policy_from_net(net_ac)}")
    print(f"  PPO      : {policy_from_net(net_p)}")

    print("\n2000-episode evaluation (feasible-action clipped, like training env):")
    class ClippedPolicy:
        def __init__(self, table):
            self.table = table
        def __getitem__(self, s):
            return int(min(int(self.table[s]), MAX_INV - int(s)))
    for name, net in [("REINFORCE", net_r), ("A2C", net_ac), ("PPO", net_p)]:
        p = evaluate(ClippedPolicy(policy_from_net(net)))
        print(f"  {name:10s}: {p:8.1f} ({100*p/exact_perf:6.1f}% of exact)")

    print("\nlast-500-episode training means (variance visible):")
    for name, c in [("REINFORCE", c_r), ("A2C", c_ac), ("PPO", c_p)]:
        c = np.array(c)
        print(f"  {name:10s}: {c[-500:].mean():8.2f}  (std {c[-500:].std():6.2f})")
