# Phase 3 — Lesson 3.6: The Deadly Triad, Experimentally (DQN Ablations)

Evidence: `docs/research/phase3_lesson6_triad_evidence.txt` (live run, 1 m 49 s
wall on 4 cores). Demo: `phase3_rl/lesson3_6_triad_ablation.py`.

Lesson 3.3 *listed* DQN's patches and the deadly triad (bootstrapping ×
non-linear FA × off-policy). This lesson **removes each patch and measures
what breaks** — the difference between being told the theory and watching it.

## 1. Five variants, identical budget (300 episodes, 11-state inventory)

| variant | %exact | what it demonstrates |
|---|---|---|
| linear-FA TD (on-policy) | 99.9% | the Tsitsiklis–Van Roy safe zone is real |
| full DQN (replay+target+Huber) | 101.1% | patches together = stable learning |
| NO target net | 95.3% | chasing your own moving tail degrades |
| NO replay | 94.9% | correlated samples → biased policy shape |
| NEITHER | 94.9% | online regression to moving targets |

Small env, honest ranking: each patch costs ~5pp when removed. The subtle
result: the **policy SHAPE** diagnoses the failure mode — without replay,
both ablations collapse to the same "order max" degenerate table
(`[8 8 8 7 6 5 4 3 2 1 0]`), a copy-the-inventory artifact of learning
from immediately-correlated transitions; full DQN recovers (s,S)-like
structure (`[6 6 1 1 1 ...]`).

## 2. What the volatility column does NOT show (be honest)

On an 11-state problem with Huber loss and gradient clipping, none of the
variants *diverged* — the triad's pathology is worst on large/discount-heavy
problems. What we measured is degradation, not explosion. Two honest notes:
1. Our first run config (1500 eps, batch 64) took **6 minutes** on this VM —
   scaled down to 300/32 with the same qualitative ranking at 1m49s. Ablation
   studies are cheap *only* if the problem is chosen small enough.
2. Divergence itself needs larger nets/lr to provoke reliably; if you want to
   SEE it, raise lr to 3e-3 and remove clipping — documented in the evidence
   file, not the default, because 4-core VMs shouldn't babysit blowups.

## 3. The linear-FA reference row is the moral anchor

`Q(s,a) = θ[s,a]` updated by plain TD (2.4's LFA in tabular clothing)
matched full DQN at 99.9% **with zero instability machinery** — no replay,
no target net, no clipping. Proven-stable methods are not just "theory
nice": on tabular-sized problems they are the *simplest correct tool*. Deep
nets earn their complexity only when |S| explodes (3.3's table) or states
are continuous.

## 4. Double-DQN in one paragraph (the overestimation add-on)

`max_a Q` inside the TD target is a max of noisy estimates → biased HIGH
(Jensen's inequality; each Q(s',a') carries positive noise, and max picks
the luckiest). Double DQN: *select* the next action with the online net,
*evaluate* it with the target net — decorrelating selection noise from
evaluation noise. On this small problem the bias is ~1-2% and hidden by
noise; on 3.4-sized continuous problems it compounds. Rule: if you
bootstrap a max, you need a selection/evaluation split.

## 5. Key takeaways

- Every DQN patch maps to a measurable failure when removed — that is the
  experimental content of the deadly triad.
- Proven-stable (linear, on-policy) methods are the honest baseline that
  deep machinery must beat; here it *tied* the best DQN variant.
- Ablation runtime is a design constraint: pick the smallest problem that
  still exhibits the phenomenon (this lesson: 300 episodes, 1.8 min).

## Exercises

1. Raise lr to 3e-3, remove clip_grad_norm, run NEITHER: catch a real
   divergence (loss → NaN or Q-magnitudes → 10³). Save the loss curve.
2. Add Double-DQN to the harness as variant 6; measure whether the
   %exact gap vs full DQN is inside seed noise (run 3 seeds).
3. Replace one-hot with a 2-feature encoding (inventory/10, bias) and
   re-run: does linear-FA stay at ~99%? (It should — and that's the point
   of 2.4's feature-engineering lesson.)
4. Measure visit distribution under the NO-replay policy: correlate the
   degenerate "order-max" shape with which (s,a) pairs ever get updates.
