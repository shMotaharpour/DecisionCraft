"""Lesson 1.8 figure: the scheduling constraint ladder as Gantt charts.

Re-solves the lesson 1.8a instance (4 jobs × 3 machines) in its three
variants and draws one Gantt panel per variant: makespan 5 → 14 → 18.
The picture IS the lesson's argument: each added constraint buys
structure at the price of makespan.
Run:  python phase1_milp/make_figures_18.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "assets", "phase1")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "phase1_milp"))

import lesson1_8a_scheduling as L  # noqa: E402

JOB_COLORS = {"J1": "#08519c", "J2": "#31a354", "J3": "#d94801",
              "J4": "#756bb1"}
JOB_DURS = {job: [d for _, d in seq] for job, seq in L.JOBS.items()}


def main():
    variants = [
        ("A. makespan only — operations float freely", False, False),
        ("B. + job precedence chains", True, False),
        ("C. + shared tooling, AddCumulative cap 2", True, True),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.4), dpi=150,
                             sharex=True)
    for ax, (label, prec, cum) in zip(axes, variants):
        solver, st, starts, ops = L.build_and_solve(prec, cum)
        makespan = int(round(solver.ObjectiveValue()))

        y = 0
        for job in L.JOBS:
            for k, (mach, dur) in enumerate(L.JOBS[job]):
                s = solver.Value(starts[(job, k)])
                mi = L.MACHINES.index(mach)
                y_pos = y * 3 + mi
                ax.barh(y_pos, dur, left=s, height=0.7,
                        color=JOB_COLORS[job], alpha=0.9,
                        edgecolor="white")
                ax.text(s + dur / 2, y_pos, f"{job}.{k+1}",
                        ha="center", va="center", fontsize=7,
                        color="white")
            y += 1

        # row labels for machines of the last job block
        ax.set_yticks([])
        for mi, mach in enumerate(L.MACHINES):
            ax.text(-0.35, mi, mach, fontsize=7, ha="right",
                    va="center", color="dimgray")
        ax.axvline(makespan, color="#a50f15", ls="--", lw=1.3)
        ax.text(makespan + 0.15, 8.0, f"makespan = {makespan}",
                fontsize=9, color="#a50f15")
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_xlim(-0.5, 20)
        ax.set_ylim(-0.5, 8.5)
        ax.grid(alpha=0.25, axis="x")

    axes[-1].set_xlabel("time")
    fig.tight_layout()
    path = os.path.join(OUT, "lesson1_8_gantt.png")
    fig.savefig(path, bbox_inches="tight")
    print("saved:", os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()
