"""Lesson 1.5 figure: the CVRP solution drawn as a map.

Re-solves the lesson's exact instance (data/cvrp_15node.json, K=2, cap=100,
PATH_CHEAPEST_ARC + GLS 3s — same config as lesson1_5a part B) and plots
each vehicle's route over the depot/customer coordinates, with demand
proportional to marker size.
Run:  python phase2_mdp/make_figures_15.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "assets", "phase1")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "phase1_milp"))

from lesson1_5a_routing_tour import dist_matrix, extract, solve  # noqa: E402


def main():
    with open(os.path.join(ROOT, "data", "cvrp_15node.json")) as f:
        d = json.load(f)
    coords, demands = d["coords"], d["demands"]
    M = dist_matrix(coords)
    n, K = len(M), 2

    manager = pywrapcp.RoutingIndexManager(n, K, 0)
    routing = pywrapcp.RoutingModel(manager)
    transit = routing.RegisterTransitCallback(
        lambda fi, ti: M[manager.IndexToNode(fi)][manager.IndexToNode(ti)])
    routing.SetArcCostEvaluatorOfAllVehicles(transit)
    dem_idx = routing.RegisterUnaryTransitCallback(
        lambda fi: demands[manager.IndexToNode(fi)])
    routing.AddDimensionWithVehicleCapacity(dem_idx, 0, [100] * K, True,
                                            "Capacity")
    sol = solve(manager, routing)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)
    colors = ["#08519c", "#d94801"]
    total = 0
    for v in range(K):
        route = extract(manager, routing, sol, v)
        pts = np.array([coords[i] for i in route])
        ax.plot(pts[:, 0], pts[:, 1], "-o", color=colors[v], lw=1.6,
                ms=6, label=f"vehicle {v+1}")
        load = sum(demands[x] for x in route if x)
        ax.annotate(f"load {load}", pts[1], fontsize=8,
                    color=colors[v], xytext=(8, 6),
                    textcoords="offset points")
        total += sol.ObjectiveValue()
    # depot marker + all customers sized by demand
    ax.plot(*coords[0], "s", color="black", ms=11, zorder=5)
    ax.annotate("depot", coords[0], fontsize=9, xytext=(10, -4),
                textcoords="offset points")
    cust = np.array(coords[1:])
    ax.scatter(cust[:, 0], cust[:, 1], s=[8 * dd for dd in demands[1:]],
               color="gray", alpha=0.35, zorder=2)
    for i, (x, y) in enumerate(coords[1:], start=1):
        ax.annotate(str(demands[i]), (x, y), fontsize=7, ha="center",
                    va="center", zorder=4)
    ax.set_title(f"CVRP solution — 2 vehicles, cap 100, "
                 f"objective {sol.ObjectiveValue():.0f} "
                 f"(marker size = demand)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_aspect("equal")

    fig.tight_layout()
    path = os.path.join(OUT, "lesson1_5_cvrp_routes.png")
    fig.savefig(path, bbox_inches="tight")
    print("saved:", os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()
