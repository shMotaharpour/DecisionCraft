"""Lesson 4.5 figure: demand censoring — what out-of-stock hides.

Poisson(4) demand, stock capped at 10. Observed sales = min(D, stock):
every demand draw above stock is recorded as 'sold everything' and the
excess vanishes. Left: true vs observed demand histograms. Right: the
empirical mean of observed sales converges to E[min(D, stock)] ≈ 3.66,
silently below the true λ = 4 — the bias that makes censored policies
chronically under-order.
Run:  python phase2_mdp/make_figures_45.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "phase4")
os.makedirs(OUT, exist_ok=True)

LAM = 4.0
STOCK = 10
rng = np.random.default_rng(13)
N = 200_000
true_d = rng.poisson(LAM, N)
observed = np.minimum(true_d, STOCK)

vals = np.arange(0, 16)
true_pmf = poisson.pmf(vals, LAM)
obs_counts = np.array([(observed == v).mean() for v in vals])
obs_pmf = obs_counts
true_counts = np.array([(true_d == v).mean() for v in vals])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)

ax = axes[0]
w = 0.4
ax.bar(vals - w / 2, true_counts, width=w, color="#08519c", alpha=0.85,
       label=f"true demand, Poisson({LAM:.0f})")
ax.bar(vals + w / 2, obs_pmf, width=w, color="#d94801", alpha=0.85,
       label=f"observed sales = min(D, {STOCK})")
ax.axvline(STOCK - 0.02, color="red", ls="--", lw=1.2)
ax.text(STOCK + 0.15, max(obs_pmf) * 0.9, "stock ceiling:\n"
        "mass piles here, excess hidden", fontsize=8, color="red")
ax.set_xlabel("units")
ax.set_ylabel("frequency")
ax.set_title("censoring: demand beyond stock is invisible")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
steps = np.unique(np.geomspace(100, N, 60).astype(int))
means_true = [true_d[:t].mean() for t in steps]
means_obs = [observed[:t].mean() for t in steps]
ax.plot(steps, means_true, lw=1.5, color="#08519c",
        label=f"true demand mean → {np.mean(means_true[-5:]):.3f}")
ax.plot(steps, means_obs, lw=1.5, color="#d94801",
        label=f"censored sales mean → {np.mean(means_obs[-5:]):.3f}")
ax.axhline(LAM, color="#08519c", ls=":", lw=1)
ax.axhline(float(np.sum(vals[:11] * true_pmf[:11])), color="gray",
           ls=":", lw=1, label="E[min(D,10)] — analytic")
ax.set_xscale("log")
ax.set_xlabel("samples (log)")
ax.set_ylabel("running mean")
ax.set_title(f"the bias never heals: {LAM:.0f} vs "
             f"{np.mean(means_obs[-5:]):.2f}")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

fig.tight_layout()
path = os.path.join(OUT, "lesson4_5_censoring.png")
fig.savefig(path, bbox_inches="tight")
print("saved:", os.path.relpath(path, HERE))
print("true mean:", round(float(np.mean(true_d[-50000:])), 3),
      "| censored mean:", round(float(np.mean(observed[-50000:])), 3))
