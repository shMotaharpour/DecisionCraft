"""Lesson 4.9 — Monte Carlo Tree Search from scratch (the AlphaGo engine).

Game: mini-farm duel — a shared pool of W=6 water; on your turn either
IRRIGATE (+2 points, takes 1 water) or REST (+0). When the pool empties
(or 12 turns pass), the higher score wins +1 (loser −1) and the player
who made the LAST irrigation gets a +3 finishing bonus. Deterministic,
perfect-information, zero-sum — small enough for exact minimax ground
truth, rich enough that the value is non-trivial.

(a) minimax ground truth (the exact answer MCTS must chase)
(b) pure MCTS (random rollouts): win rate vs the minimax player as
    simulation budget grows — the anytime-ness of MCTS
(c) MCTS with a value prior replacing random rollouts (the AlphaGo
    value-network slot, filled by a 3-line heuristic)
(d) root value-estimate quality vs simulations
Runtime < 15 s.
Run:  python phase4_hybrid/lesson4_9_mcts.py
"""
import math
import time
from functools import lru_cache

import numpy as np

W_TOTAL = 6
POINTS = 2
LAST_BONUS = 3
HORIZON_TURNS = 2 * W_TOTAL


class FarmDuel:
    """Deterministic 2-player zero-sum game. player 0 moves first."""

    def __init__(self, water=W_TOTAL, mine=0, other=0, player=0, turn=0,
                 last=-1):
        self.water, self.mine, self.other = water, mine, other
        self.player, self.turn, self.last = player, turn, last

    def key(self):
        return (self.water, self.mine, self.other, self.player,
                self.turn, self.last)

    def moves(self):
        return (["IRRIGATE"] if self.water > 0 else []) + ["REST"]

    def step(self, mv):
        if mv == "IRRIGATE":
            return FarmDuel(self.water - 1, self.mine + POINTS,
                            self.other, 1 - self.player, self.turn + 1,
                            self.player)
        return FarmDuel(self.water, self.other, self.mine,
                        1 - self.player, self.turn + 1, self.last)

    def terminal(self):
        return self.water == 0 or self.turn >= HORIZON_TURNS

    def payoff_from_player0(self):
        d = (self.mine - self.other) if self.player == 0 \
            else (self.other - self.mine)
        base = 1 if d > 0 else (-1 if d < 0 else 0)
        bonus = LAST_BONUS if self.last == 0 else -LAST_BONUS
        return base + bonus


@lru_cache(maxsize=None)
def minimax(k):
    s = FarmDuel(*k)
    if s.terminal():
        return s.payoff_from_player0()
    vals = [minimax(s.step(m).key()) for m in s.moves()]
    return max(vals) if s.player == 0 else min(vals)


# ---------------------------------------------------------------- MCTS
class Node:
    __slots__ = ("state", "parent", "move", "children", "untried",
                 "wins", "visits", "player_just_moved")

    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent, self.move = parent, move
        self.children = []
        self.untried = list(state.moves())
        self.wins = 0.0                       # from mover's perspective
        self.visits = 0
        self.player_just_moved = 1 - state.player

    def ucb(self, c):
        if self.visits == 0:
            return math.inf
        return self.wins / self.visits + c * math.sqrt(
            math.log(self.parent.visits) / self.visits)


def mcts(state, simulations=500, c=1.4, value_prior=None, seed=7):
    """One MCTS call = one move decision. Returns (best_move, root).
    value_prior(state) -> [0,1] estimate for the player to move; replaces
    the random rollout (the AlphaGo value-network slot)."""
    r = np.random.default_rng(seed)
    root = Node(state)
    for _ in range(simulations):
        node = root
        # 1) selection: walk down fully-expanded nodes via UCB1
        while not node.untried and node.children:
            node = max(node.children, key=lambda ch: ch.ucb(c))
        # 2) expansion: add one untried child
        if node.untried:
            mv = node.untried.pop(int(r.integers(len(node.untried))))
            child = Node(node.state.step(mv), parent=node, move=mv)
            node.children.append(child)
            node = child
        # 3) simulation: random rollout OR value prior
        cur = node.state
        if value_prior is not None:
            outcome = 1.0 if value_prior(cur) > 0.5 else \
                (-1.0 if value_prior(cur) < 0.5 else 0.0)
        else:
            guard = 0
            while not cur.terminal() and guard < 40:
                cur = cur.step(cur.moves()[int(
                    r.integers(len(cur.moves())))])
                guard += 1
            raw = cur.payoff_from_player0()
            outcome = float(np.sign(raw))
        # 4) backprop: sign flips per level (zero-sum alternating)
        while node is not None:
            node.visits += 1
            if node.player_just_moved == 0:
                node.wins += outcome
            else:
                node.wins -= outcome
            node = node.parent
    best = max(root.children, key=lambda ch: ch.visits)
    return best.move, root


def play_match(agent0, agent1, seed=3):
    r = np.random.default_rng(seed)
    s = FarmDuel()
    while not s.terminal():
        mv = agent0(s, r) if s.player == 0 else agent1(s, r)
        s = s.step(mv)
    return s.payoff_from_player0()


def minimax_agent(s, r):
    vals = {m: minimax(s.step(m).key()) for m in s.moves()}
    best = max(vals.values()) if s.player == 0 else min(vals.values())
    return [m for m, v in vals.items() if v == best][0]


def main():
    t0 = time.time()
    print("mini-farm duel: W=6, IRRIGATE=+2, last irrigator +3 bonus,")
    print("higher score wins (+1/-1). Deterministic, perfect info.\n")

    truth = minimax(FarmDuel().key())
    print(f"(a) minimax ground truth: game value {truth:+.0f} "
          f"[{time.time()-t0:.1f}s]")
    print("    (positive = player 0 can force a win with perfect play)")

    def mcts_agent(sims):
        def agent(s, r):
            mv, _ = mcts(s, simulations=sims, seed=int(r.integers(1e9)))
            return mv
        return agent

    print("\n(b) pure MCTS (random rollouts) vs the minimax player")
    print("    The game value is -2 for player 0: the SECOND mover wins")
    print("    under perfect play (last-irrigator bonus + tempo). So the")
    print("    honest test: can MCTS (as player 1, the winning side) hold")
    print("    the win? 40 games per budget:")
    for sims in (100, 500, 2000):
        wins = sum(play_match(minimax_agent, mcts_agent(sims),
                              seed=100 + g) < 0 for g in range(40))
        print(f"    {sims:5d} sims/move: MCTS holds the win {wins}/40")

    print("\n(c) guided MCTS — value prior instead of random rollout")
    def value_prior(s):
        """3-line stand-in for AlphaGo's value network. MUST report from
        the perspective of the player who just moved (the node credited in
        backprop is player_just_moved = 1 - s.player)."""
        diff = (s.other - s.mine) if s.player == 0 else (s.mine - s.other)
        return 0.5 + 0.18 * diff + (0.06 if s.water <= 2 else 0.0)

    wins = 0
    for g in range(40):
        def agent(s, r):
            mv, _ = mcts(s, simulations=200, value_prior=value_prior,
                         seed=int(r.integers(1e9)))
            return mv
        wins += play_match(minimax_agent, agent, seed=500 + g) < 0
    print(f"    guided 200 sims/move: MCTS holds the win {wins}/40")

    print("\n(d) root value-estimate quality (truth = %+d)" % truth)
    for sims in (100, 500, 2000):
        mv, root = mcts(FarmDuel(), simulations=sims, seed=11)
        child = max(root.children, key=lambda ch: ch.visits)
        est = child.wins / max(1, child.visits)
        print(f"    {sims:5d} sims: est {est:+.3f} on move '{mv}' "
              f"({child.visits} visits)")

    print(f"\ntotal runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
