"""Lesson 1.1 — Toy portfolio MILP, solved with HiGHS via highspy (high-level API).

Formulation (see notes/phase1_lesson1_milp_alphabet.md):
    maximize   0.08*x_A + 0.12*x_B + 1000*x_C          (0.20 * 5000)
    subject to x_A + x_B + 5000*x_C <= 10000           (budget)
               x_B - 10000*y_B <= 0                    (big-M switch)
               x_B - 2000*y_B  >= 0                    (minimum if active)
               x_A, x_B >= 0 ;  y_B, x_C in {0, 1}
"""
import highspy

INF = highspy.kHighsInf
h = highspy.Highs()
h.setOptionValue("output_flag", False)

x_A = h.addVariable(0, INF, name="x_A")
x_B = h.addVariable(0, INF, name="x_B")
y_B = h.addVariable(0, 1, name="y_B")
x_C = h.addVariable(0, 1, name="x_C")

for v in (y_B, x_C):
    h.changeColIntegrality(v.index, highspy.HighsVarType.kInteger)

h.addConstr(x_A + x_B + 5000.0 * x_C <= 10000, name="budget")
h.addConstr(x_B <= 10000.0 * y_B, name="bigM_switch")
h.addConstr(x_B >= 2000.0 * y_B, name="min_if_active")

h.maximize(0.08 * x_A + 0.12 * x_B + 1000.0 * x_C)

status = h.getModelStatus()
print("Status:", h.modelStatusToString(status))
print(f"Objective = {h.getObjectiveValue():,.2f}")
for v in (x_A, x_B, y_B, x_C):
    print(f"  {v.name:16s} = {h.val(v):,.4f}")
