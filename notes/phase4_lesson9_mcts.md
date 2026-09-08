# Phase 4 — Lesson 4.9: Monte Carlo Tree Search — Planning With a Simulator

Evidence: `docs/research/phase4_lesson9_mcts_evidence.txt` (live run, ~86 s).
Demo: `phase4_hybrid/lesson4_9_mcts.py`.

Named four times in this course (2.3, 3.5, 4.2, 4.7) — never built. This
lesson builds MCTS from scratch (~80 lines) on a tiny two-player farm duel
and then fills in, piece by piece, exactly where AlphaGo's networks plug in.

## 1. The algorithm (Browne et al. 2012, four phases)

MCTS is **planning by repeated simulation**, not learned decision trees:
each move decision builds a fresh search tree from the current state:

1. **Selection** — descend from the root through fully-expanded nodes,
   picking children by **UCB1** (`w_i/n_i + c·√(ln N / n_i)`) — lesson
   3.5's bandit formula, one bandit per node.
2. **Expansion** — add one untried child.
3. **Simulation** — finish the game from the new node: either a random
   rollout (pure MCTS) or a **value estimate** (the AlphaGo value-network
   slot).
4. **Backpropagation** — update `wins/visits` up the path, flipping sign
   each level (zero-sum: my win is your loss).

The move played = the child with the most visits (robust child, not the
best-mean child). The tree is built, used, discarded — MCTS re-plans from
scratch every turn (receding-horizon planning, lesson 2.4 route 3).

## 2. The miniature — farm duel (deterministic, perfect info, zero-sum)

Shared pool W=6; each turn IRRIGATE (+2 pts, takes water) or REST. Higher
score wins ±1; the **last irrigator** collects a +3 finishing bonus.
Game value via exact minimax: **−2 for player 0** — the second mover wins
under perfect play. Two design lessons from building it:

1. **Degenerate first draft**: without the turn horizon, REST cycles
   forever → `RecursionError`. Any game with a "pass" action needs a turn
   cap; state design (lesson 2.3 Axis 1) applies to game trees too.
2. **The honest benchmark**: since player 0 *loses* with perfect play,
   the fair test is "can MCTS, as the winning side, hold the win?" —
   testing MCTS from the losing side just measures the opponent.

## 3. Live results (40 games per row)

| agent | result |
|---|---|
| pure MCTS, 100 sims/move (as player 1) | holds the win 31/40 |
| pure MCTS, 500 sims/move | 40/40 |
| pure MCTS, 2000 sims/move | 40/40 |
| **guided MCTS (value prior), 200 sims/move** | **40/40** |

Reading: MCTS's strength grows with simulation budget (anytime), and
replacing random rollouts with even a crude value estimate lets 200
simulations do the work of 2000 — the precise slot AlphaGo's value
network occupies. Root value estimates hovered near −0.26…−0.28 against
the true −2: with only ±1 outcomes, MCTS's estimate is a *win
probability*, not a value magnitude — the reason AlphaGo regressed to a
continuous value head.

## 4. The verified perspective bug (the pedagogical gold)

Guided MCTS v1 scored **0/40** — worse than random rollouts. Cause: the
value prior reported "who is ahead" from the *player-to-move*
perspective, while backpropagation credits `player_just_moved` — one ply
mismatch, so every simulated outcome fed the tree **inverted values**.
An AlphaGo with this bug would play to lose. Fix: the value head must
evaluate from the perspective of the player who just moved
(`1 − s.player`). Meta-lesson: **a value-network sign/perspective error
is invisible in the code and obvious only in head-to-head play** —
always smoke-test the guidance against unguided MCTS.

## 5. Key takeaways

- MCTS = UCB1 (3.5) + simulator (2.3's "P unknown, sim available") +
  receding-horizon planning (2.4). Nothing new mathematically — a new
  *composition*.
- Anytime: 100 sims = 31/40, 500 sims = 40/40. Budget is a dial.
- Networks don't replace MCTS in AlphaGo — they *guide* it (policy =
  which branches to try, value = what rollouts become). The tree stays.
- Chista connection: the MILP tactical layer could sit inside an MCTS as
  the simulation policy — the same hybrid slot lesson 4.3 describes.

## Exercises

1. Sweep UCB `c ∈ {0.2, 0.5, 1.4, 3}` at 200 sims; find the exploration
   sweet spot and tie it to the bandit constant lesson of 3.5.
2. Play guided-vs-guided (both 200 sims): does the first/second mover
   advantage survive? Report the score.
3. Increase W to 10 and watch the minimax cache blow past memory —
   MCTS doesn't care. That asymmetry IS MCTS's reason to exist.
4. Use visits-per-child at the root as a learned policy target (the
   AlphaZero training signal); train a tiny network on 500 self-play
   positions and use it as the policy prior.
