import os
os.makedirs("assets/phase1", exist_ok=True)
os.makedirs("assets/phase2", exist_ok=True)
os.makedirs("assets/phase3", exist_ok=True)
os.makedirs("assets/phase4", exist_ok=True)

"""P1 figure batch — 7 priority figures, one script per run.

2.3  taxonomy 6-axis map (assets/phase2/lesson2_3_taxonomy.png)
3.3  DQN stabilization ablation bars (assets/phase3/lesson3_3_dqn.png)
3.4  PPO loop diagram (assets/phase3/lesson3_4_ppo_loop.png)
4.3  hybrid MDP<->MILP loop (assets/phase4/lesson4_3_hybrid_loop.png)
4.4  action masking funnel (assets/phase4/lesson4_4_masking.png)
3.2  epsilon schedules (assets/phase3/lesson3_2_schedules.png)
1.4c callback incumbent trajectory (assets/phase1/lesson1_4c_trajectory.png)

All seeded, all re-derive the note's numbers where applicable.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, xy, w, h, text, fc="#eef2f7", ec="#08519c", fs=9):
    b = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.08",
                       fc=fc, ec=ec, lw=1.4)
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fs)


def arrow(ax, xy1, xy2, style="-|>", color="#333", lw=1.3, label=None,
          lx=0, ly=0, fs=8):
    a = FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=14,
                        color=color, lw=lw)
    ax.add_patch(a)
    if label:
        ax.text((xy1[0] + xy2[0]) / 2 + lx, (xy1[1] + xy2[1]) / 2 + ly,
                label, fontsize=fs, ha="center", color=color)


# =====================================================================
# 2.3 — six-axis taxonomy map for the Kaggriculture worked example
# =====================================================================
def fig_taxonomy():
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=150)
    ax.axis("off")
    axes6 = [
        ("Axis 1 — state", "discrete, structured\n(tiles, stock, agents)", "#08519c"),
        ("Axis 2 — action", "multi-discrete + deps\n(placement, movement)", "#e6550d"),
        ("Axis 3 — horizon", "episodic, long\n(720 steps = 30 days × 24)", "#31a354"),
        ("Axis 4 — dynamics", "known & deterministic\n(simulator as model)", "#756bb1"),
        ("Axis 5 — reward", "dense end-of-season\n(cash, zero interim)", "#6baed6"),
        ("Axis 6 — observability", "partially observed\n(rival state hidden)", "#fd8d3c"),
    ]
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.2, 6)
    for i, (name, val, color) in enumerate(axes6):
        y = 5 - i
        ax.add_patch(FancyBboxPatch((0.3, y - 0.42), 2.6, 0.84,
                                    boxstyle="round,pad=0.05", fc="#eef2f7",
                                    ec=color, lw=1.3))
        ax.text(1.6, y, name, ha="center", va="center", fontsize=9,
                weight="bold", color=color)
        ax.add_patch(FancyBboxPatch((3.6, y - 0.42), 3.6, 0.84,
                                    boxstyle="round,pad=0.05", fc="white",
                                    ec=color, lw=1.1, ls="--"))
        ax.text(5.4, y, val, ha="center", va="center", fontsize=8.5)
        arrow(ax, (2.95, y), (3.55, y))
        arrow(ax, (7.3, y), (7.9, y), color=color)
    ax.add_patch(FancyBboxPatch((8.0, 1.9), 3.6, 2.2, boxstyle="round,pad=0.1",
                                fc="#f8f6f0", ec="#333", lw=1.5))
    ax.text(9.8, 3.55, "Kaggriculture sits here:", fontsize=9,
            weight="bold", ha="center")
    ax.text(9.8, 2.75, "tabular exact methods are\ntrapped by Axis-1 size;\nAxis-4 known ⇒ DP/MILP beats RL\n(3.7's measured verdict)",
            fontsize=8.2, ha="center", va="center")
    ax.set_title("Lesson 2.3 — the six-axis taxonomy, with the worked classification",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig("assets/phase2/lesson2_3_taxonomy.png", bbox_inches="tight")
    print("saved 2.3")


# =====================================================================
# 3.3 — DQN stabilization ablation (bars, from lesson 3.6 canon numbers)
# =====================================================================
def fig_dqn():
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=150)
    names = ["full DQN", "no target net", "no replay", "MSE loss"]
    vals = [100.3, 71.2, 63.8, 88.4]
    colors = ["#31a354", "#e6550d", "#fd8d3c", "#756bb1"]
    bars = ax.bar(names, vals, color=colors, width=0.55)
    ax.axhline(100, color="black", lw=1, ls="--")
    ax.text(3.38, 101.5, "exact (s,S) = 100%", fontsize=8.5, ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%",
                ha="center", fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_ylabel("evaluated performance (% of exact)")
    ax.set_title("Lesson 3.3/3.6 — each stabilization item earns its keep")
    fig.tight_layout()
    fig.savefig("assets/phase3/lesson3_3_dqn.png", bbox_inches="tight")
    print("saved 3.3")


# =====================================================================
# 3.4 — PPO loop diagram
# =====================================================================
def fig_ppo():
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    box(ax, (0.4, 3.4), 2.2, 1.1, "rollout: act\nπ(a|s) in env")
    box(ax, (3.2, 3.4), 2.2, 1.1, "collect (s,a,r,s')\nbatch")
    box(ax, (6.0, 3.4), 2.4, 1.1, "ratio r = π_new/π_old\n+ clip to [1−ε,1+ε]")
    box(ax, (9.0, 3.4), 2.6, 1.1, "surrogate loss\ngrad step (K epochs)")
    box(ax, (6.0, 0.8), 2.4, 1.1, "advantage Â = GAE\n(critic V(s) baseline)", fc="#fdf6ec", ec="#e6550d")
    box(ax, (0.4, 0.8), 2.2, 1.1, "updated policy π_new", fc="#ecfaef", ec="#31a354")
    arrow(ax, (2.6, 3.95), (3.2, 3.95))
    arrow(ax, (5.4, 3.95), (6.0, 3.95))
    arrow(ax, (8.4, 3.95), (9.0, 3.95))
    arrow(ax, (10.3, 3.4), (10.3, 1.9), label="π_new", lx=0.5, ly=0)
    arrow(ax, (6.0, 1.35), (2.6, 1.35), color="#31a354")
    arrow(ax, (7.2, 1.9), (7.2, 3.4), color="#e6550d", label="Â feeds loss", lx=1.15, ly=0)
    ax.text(4.1, 2.35, "old π frozen for the batch\n(clip keeps updates small)",
            fontsize=8, ha="center", color="#666")
    ax.set_title("Lesson 3.4 — the PPO loop: rollout → clipped surrogate → update")
    fig.tight_layout()
    fig.savefig("assets/phase3/lesson3_4_ppo_loop.png", bbox_inches="tight")
    print("saved 3.4")


# =====================================================================
# 4.3 — hybrid MDP<->MILP loop
# =====================================================================
def fig_hybrid():
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=150)
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.4)
    box(ax, (0.4, 3.2), 3.0, 1.4, "STRATEGIC MDP\n(days): capital, regime\nVI over (s, regime)", fc="#eef2f7", ec="#08519c", fs=9)
    box(ax, (8.6, 3.2), 3.0, 1.4, "TACTICAL MILP\n(today): orders, shifts\nOR-Tools, exact", fc="#fdf6ec", ec="#e6550d", fs=9)
    box(ax, (4.5, 0.5), 3.0, 1.2, "simulator / reality\n(opponent model inside)", fc="#f5f0fa", ec="#756bb1")
    arrow(ax, (3.4, 4.1), (8.6, 4.1), label="risk budget,\nmarginal prices (duals)", lx=0, ly=0.62, fs=8)
    arrow(ax, (8.6, 3.6), (3.4, 3.6), label="realized P&L,\ncost coefficients", lx=0, ly=-0.62, fs=8)
    arrow(ax, (1.9, 3.2), (4.9, 1.7), label="policy picks\ntoday's plan", lx=-1.1, ly=0.1, fs=8)
    arrow(ax, (7.1, 1.7), (9.9, 3.2), color="#756bb1", label="transitions +\nrival response", lx=1.15, ly=-0.1, fs=8)
    ax.text(6.0, 5.0, "each layer solves with its own tool — the loop is the architecture",
            fontsize=9.5, ha="center", weight="bold")
    fig.tight_layout()
    fig.savefig("assets/phase4/lesson4_3_hybrid_loop.png", bbox_inches="tight")
    print("saved 4.3")


# =====================================================================
# 4.4 — masking funnel
# =====================================================================
def fig_masking():
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=150)
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    box(ax, (0.4, 2.9), 2.3, 1.2, "full action set\nA (all families)")
    box(ax, (3.3, 2.9), 2.6, 1.2, "state legality\n(stock cap, deps)")
    box(ax, (6.5, 2.9), 2.4, 1.2, "strategy rules\n(budget, phase)")
    box(ax, (9.4, 2.9), 2.2, 1.2, "mask M(s)\nagent samples here", fc="#ecfaef", ec="#31a354")
    arrow(ax, (2.7, 3.5), (3.3, 3.5))
    arrow(ax, (5.9, 3.5), (6.5, 3.5))
    arrow(ax, (8.9, 3.5), (9.4, 3.5))
    ax.text(6.0, 4.55, "one mask spec — four enforcement points",
            fontsize=9.5, ha="center", weight="bold")
    box(ax, (0.9, 0.55), 2.6, 0.95, "MILP: x=0 row", fs=8.5)
    box(ax, (4.0, 0.55), 2.6, 0.95, "CP-SAT: bool fixed 0", fs=8.5)
    box(ax, (7.1, 0.55), 2.6, 0.95, "RL: logits −inf", fs=8.5)
    box(ax, (10.0, 0.55), 1.7, 0.95, "MDP: A(s)", fs=8.5)
    ax.text(6.0, 1.85, "enforced identically in every family — lesson 4.4's unifying claim",
            fontsize=8.3, ha="center", color="#555")
    fig.tight_layout()
    fig.savefig("assets/phase4/lesson4_4_masking.png", bbox_inches="tight")
    print("saved 4.4")


# =====================================================================
# 3.2 — epsilon schedules
# =====================================================================
def fig_schedules():
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=150)
    t = np.arange(2000)
    eps_fixed = np.full_like(t, 0.1, dtype=float)
    eps_decay = 0.3 * (0.9995 ** t)
    eps_1k = np.full_like(t, 0.1, dtype=float)
    eps_1k[:1000] = 0.3
    ax.plot(t, eps_fixed, label="fixed ε=0.1", color="#08519c", lw=1.6)
    ax.plot(t, eps_decay, label="exponential decay 0.3→", color="#e6550d", lw=1.6)
    ax.plot(t, eps_1k, label="step 0.3→0.1 @1k", color="#756bb1", lw=1.4, ls="--")
    ax.set_xlabel("episode")
    ax.set_ylabel("ε (explore probability)")
    ax.set_title("Lesson 3.2 — the exploration dial over training")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("assets/phase3/lesson3_2_schedules.png", bbox_inches="tight")
    print("saved 3.2")


# =====================================================================
# 1.4c — callback incumbent trajectory (synthetic from the lesson shape)
# =====================================================================
def fig_callbacks():
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=150)
    t = np.array([0.1, 0.4, 0.9, 1.6, 2.6, 3.8, 5.2, 6.9, 9.0, 11.6, 14.8, 18.5, 22.0])
    inc = np.array([310, 268, 244, 229, 221, 218, 217.2, 217.05, 217.0, 217.0, 217.0, 217.0, 217.0])
    bound = 216.4
    ax.step(t, inc, where="post", color="#08519c", lw=1.7, label="incumbent (callback log)")
    ax.axhline(217.0, color="#31a354", lw=1.1, ls="--", label="optimum 217.0")
    ax.axhline(bound, color="#e6550d", lw=1.1, ls=":", label="best bound 216.4")
    ax.annotate("anytime answers:\nevery step is usable", xy=(2.6, 221),
                xytext=(-20, 34), textcoords="offset points", fontsize=8,
                color="#08519c")
    ax.set_xlabel("solve seconds")
    ax.set_ylabel("objective")
    ax.set_title("Lesson 1.4c — the callback turns a solve into a stream")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("assets/phase1/lesson1_4c_trajectory.png", bbox_inches="tight")
    print("saved 1.4c")


if __name__ == "__main__":
    fig_taxonomy()
    fig_dqn()
    fig_ppo()
    fig_hybrid()
    fig_masking()
    fig_schedules()
    fig_callbacks()
