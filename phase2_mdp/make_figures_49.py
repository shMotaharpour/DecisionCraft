"""Lesson 4.9 figure: MCTS budget curve + value-guidance effect.

Re-runs the lesson's match protocol with fine simulation budgets:
  left  — MCTS win rate (holding the win as player 1) vs simulation
          budget, pure vs guided (value prior): the guidance gap
  right — root value-estimate (win probability) vs simulations, against
          the true ±1 outcome scale; shows MCTS estimates PROBABILITIES
Run:  python phase2_mdp/make_figures_49.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "assets", "phase4")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "phase4_hybrid"))

from lesson4_9_mcts import (  # noqa: E402
    FarmDuel, mcts, minimax, minimax_agent, play_match)


def value_prior(s):
    """Perspective-corrected prior (the lesson's fixed version): from the
    player who just moved."""
    diff = (s.other - s.mine) if s.player == 0 else (s.mine - s.other)
    return 0.5 + 0.18 * diff + (0.06 if s.water <= 2 else 0.0)


def main():
    truth = minimax(FarmDuel().key())          # -2: player 1 wins
    sims_list = [50, 100, 200, 500, 1000, 2000]
    n_games = 20

    def pure_agent(sims):
        def agent(s, r):
            mv, _ = mcts(s, simulations=sims,
                         seed=int(r.integers(1e9)))
            return mv
        return agent

    def guided_agent(sims):
        def agent(s, r):
            mv, _ = mcts(s, simulations=sims, value_prior=value_prior,
                         seed=int(r.integers(1e9)))
            return mv
        return agent

    pure_wr, guided_wr = [], []
    for sims in sims_list:
        w_pure = sum(play_match(minimax_agent, pure_agent(sims),
                                seed=1000 + g) < 0
                     for g in range(n_games))
        w_guid = sum(play_match(minimax_agent, guided_agent(sims),
                                seed=2000 + g) < 0
                     for g in range(n_games))
        pure_wr.append(w_pure / n_games)
        guided_wr.append(w_guid / n_games)
        print(f"sims {sims:5d}: pure {w_pure}/{n_games}  "
              f"guided {w_guid}/{n_games}")

    # root value estimate vs sims (win-probability reading)
    ests = []
    for sims in sims_list:
        _, root = mcts(FarmDuel(), simulations=sims, seed=42)
        child = max(root.children, key=lambda ch: ch.visits)
        ests.append(child.wins / max(1, child.visits))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    ax = axes[0]
    ax.plot(sims_list, [w * 100 for w in pure_wr], "o-", color="#ff7f0e",
            lw=1.6, label="pure MCTS (random rollouts)")
    ax.plot(sims_list, [w * 100 for w in guided_wr], "s-",
            color="#2ca02c", lw=1.6,
            label="guided MCTS (value prior)")
    ax.set_xscale("log")
    ax.axhline(100, color="gray", ls=":", lw=1,
               label="perfect play (minimax)")
    ax.set_xlabel("simulations per move (log)")
    ax.set_ylabel("MCTS holds the win (%)")
    ax.set_title("anytime strength: 20 games per point, MCTS as player 1")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(sims_list, ests, "o-", color="#08519c", lw=1.6,
            label="MCTS root estimate (visit-weighted)")
    ax.axhline(0, color="black", lw=1)
    ax.text(sims_list[0], 0.03,
            "MCTS estimates a WIN PROBABILITY (±1 outcomes),\n"
            "not the value magnitude (true game value = −2) —\n"
            "the reason AlphaGo regresses a continuous value head",
            fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("simulations per move (log)")
    ax.set_ylabel("root value estimate")
    ax.set_title("what MCTS actually estimates")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    path = os.path.join(OUT, "lesson4_9_mcts_budget.png")
    fig.savefig(path, bbox_inches="tight")
    print("saved:", os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()
