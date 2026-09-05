"""Lesson 1.4c — Callbacks (CpSolverSolutionCallback) + constraint cost.

(a) Collector: subclassing the callback, aggregating all solutions.
(b) EarlyStop: self.StopSearch() from inside the callback.
(c) Cost experiment: pure-linear vs var*var product objective on the same
    counting model — compare wall time, branches and conflicts.
"""
import time

from ortools.sat.python import cp_model


# ------------------------------------------------- (a) Collector
class Collector(cp_model.CpSolverSolutionCallback):
    """Aggregate every solution of a small model."""

    def __init__(self, variables):
        super().__init__()  # required: wire the C++ proxy
        self._variables = variables
        self.seen = []

    def on_solution_callback(self):
        # values must be read via self.* helpers, never solver.Value here
        self.seen.append(tuple(self.Value(v) for v in self._variables))


# ------------------------------------------------- (b) Early stop
class EarlyStop(cp_model.CpSolverSolutionCallback):
    """Record the first N improving solutions then abort the search."""

    def __init__(self, variables, n):
        super().__init__()
        self._variables = variables
        self._n = n
        self.count = 0

    def on_solution_callback(self):
        self.count += 1
        print(f"    improving solution #{self.count}: "
              + ", ".join(f"{v.name}={self.Value(v)}" for v in self._variables)
              + f" obj={self.ObjectiveValue():.0f}")
        if self.count >= self._n:
            print("    -> StopSearch() from inside the callback")
            self.StopSearch()


def count_solutions(with_product: bool):
    """Count solutions of sum(x_i) == 5, i in 0..4, x_i binary.
    Objective differs: pure linear vs containing an x_i * x_j product
    (via AddMultiplicationEquality) — same answers, very different cost."""
    m = cp_model.CpModel()
    xs = [m.NewBoolVar(f"x_{i}") for i in range(5)]
    m.Add(sum(xs) == 3)  # exactly three of five: C(5,3) = 10 solutions
    if with_product:
        # artificial non-linear term: z = x_0 * x_1, maximize z
        z = m.NewBoolVar("z")
        m.AddMultiplicationEquality(z, [xs[0], xs[1]])
        m.Maximize(z)  # forces real search, not just enumeration
    else:
        m.Maximize(sum(i * x for i, x in enumerate(xs)))  # pure linear

    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 4
    t0 = time.perf_counter()
    status = solver.Solve(m)
    dt = time.perf_counter() - t0
    tag = "linear" if not with_product else "with x*y product"
    print(f"(c) {tag:18s}: status={solver.StatusName(status):8s} "
          f"obj={solver.ObjectiveValue():.0f} branches={solver.NumBranches():5d} "
          f"conflicts={solver.NumConflicts():4d} wall={dt*1000:6.1f} ms")


if __name__ == "__main__":
    print("(a) all solutions of exactly-3-of-5 binaries:")
    m = cp_model.CpModel()
    xs = [m.NewBoolVar(f"x_{i}") for i in range(5)]
    m.Add(sum(xs) == 3)
    collector = Collector(xs)
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.num_workers = 1  # IMPORTANT: multithreaded search does
    # not enumerate every solution exactly (workers split the space); single
    # worker guarantees complete enumeration.
    solver.SearchForAllSolutions(m, collector)
    print(f"    collected {len(collector.seen)} solutions (expect 10), e.g. {collector.seen[:3]}")

    print("\n(b) early stopping after 2 improving solutions:")
    m = cp_model.CpModel()
    # a knapsack-style objective forces several improving incumbents before
    # the optimum, so the callback fires more than once
    xs = [m.NewIntVar(0, 20, f"y_{i}") for i in range(4)]
    w = [7, 11, 13, 17]  # pseudo-random weights -> search can't jump to optimum
    m.Add(sum(xs) <= 50)
    m.Maximize(sum(w[i] * xs[i] for i in range(4)))
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 4
    cb = EarlyStop(xs, 2)
    status_after_stop = solver.Solve(m, cb)
    print(f"    Solve() returned {solver.StatusName(status_after_stop)} "
          f"(FEASIBLE = stopped early, best kept; OPTIMAL = found optimum first)")

    print()
    count_solutions(with_product=False)
    count_solutions(with_product=True)