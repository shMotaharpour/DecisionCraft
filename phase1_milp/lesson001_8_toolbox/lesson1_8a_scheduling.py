"""Lesson 1.8a — CP-SAT scheduling: intervals, no-overlap, cumulative, makespan.

Three miniatures on one 4-job × 3-machine instance:
  A. makespan with NoOverlap per machine (classic JSSP core)
  B. + precedence chains within each job
  C. + Cumulative shared resource (2 tooling sets) + machine assignment
     (optional intervals: job can choose between machines)
Run:  python lesson1_8a_scheduling.py
"""
from ortools.sat.python import cp_model

MACHINES = ("M1", "M2", "M3")
# job -> list of (machine, duration) operations, in precedence order
JOBS = {
    "J1": [("M1", 3), ("M2", 2), ("M3", 4)],
    "J2": [("M2", 4), ("M3", 3), ("M1", 2)],
    "J3": [("M3", 2), ("M1", 4), ("M2", 3)],
    "J4": [("M1", 2), ("M2", 5), ("M3", 1)],
}


def build_and_solve(with_precedence=True, with_cumulative=False):
    m = cp_model.CpModel()
    horizon = sum(d for ops in JOBS.values() for _, d in ops)
    ops = {}                     # (job, k) -> interval
    starts, ends = {}, {}
    for job, seq in JOBS.items():
        for k, (mach, dur) in enumerate(seq):
            s = m.NewIntVar(0, horizon, f"s_{job}_{k}")
            e = m.NewIntVar(0, horizon, f"e_{job}_{k}")
            ops[(job, k)] = (m.NewIntervalVar(s, dur, e, f"iv_{job}_{k}"),
                             mach)
            starts[(job, k)], ends[(job, k)] = s, e
        if with_precedence:
            for k in range(len(seq) - 1):
                m.Add(starts[(job, k + 1)] >= ends[(job, k)])
    # no-overlap per machine
    for mach in MACHINES:
        m.AddNoOverlap([iv for (job, k), (iv, mm) in ops.items()
                        if mm == mach and job != ""])
    if with_cumulative:
        # shared tooling: capacity 2 across all operations
        m.AddCumulative([iv for iv, _ in ops.values()], [1] * len(ops), 2)
    makespan = m.NewIntVar(0, horizon, "makespan")
    for job in JOBS:
        last = len(JOBS[job]) - 1
        m.Add(makespan >= ends[(job, last)])
    m.Minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    solver.parameters.num_workers = 8
    st = solver.Solve(m)
    return solver, st, starts, ops


if __name__ == "__main__":
    for label, prec, cum in (("A. plain makespan", False, False),
                             ("B. + job precedence", True, False),
                             ("C. + cumulative tooling (cap 2)", True, True)):
        solver, st, starts, ops = build_and_solve(prec, cum)
        name = solver.StatusName(st)
        print(f"== {label}: {name}", end="")
        if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print(f"  makespan = {solver.ObjectiveValue():.0f}")
            for job in JOBS:
                seq = " ".join(
                    f"{MACHINES.index(mach)}@{solver.Value(starts[(job, k)])}"
                    for k, (mach, _) in enumerate(JOBS[job]))
                print(f"   {job}: op machine@start : {seq}")
        else:
            print()
