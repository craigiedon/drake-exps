from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve 
import numpy as np

# Creat an empty MathematicalProgram named prog (with no decision variables, constraints or costs)
prog = MathematicalProgram()
# Add two decision variables x[0], x[1]
x = prog.NewContinuousVariables(2, "x")

# Add a symbolic linear expression as the cost
cost_1 = prog.AddLinearCost(x[0] + 3 * x[1] + 2)

# Print the newly added cost
print(cost_1)

# The newly added cost is stored in prog.linear_costs()
print(prog.linear_costs()[0])

cost_2 = prog.AddLinearCost(2 * x[1] + 3)
print(f"number of linear cost objects: {len(prog.linear_costs())}")

# Can also add costs in a vector coefficient form
cost_3 = prog.AddLinearCost([3., 4.], 5.0, x)
print(cost_3)

# Can also just call the generic "AddCost".
# If drake thinks its linear, gets added to linear costs list
cost_4 = prog.AddCost(x[0] + 3 * x[1] + 5)
print(f"Number of linear cost objects after calling AddCost: {len(prog.linear_costs())}")

# New program, now with constraints
prog = MathematicalProgram()
x = prog.NewContinuousVariables(2, "x")
y = prog.NewContinuousVariables(3, "y")

bounding_box = prog.AddLinearConstraint(x[1] <= 2)
linear_eq = prog.AddLinearConstraint(x[1] + 2 * y[2] == 1)
linear_ineq = prog.AddLinearConstraint(x[1] + 4 * y[1] >= 2)

# New program, all the way to solving...
prog = MathematicalProgram()
# Declare x as decision variables.
x = prog.NewContinuousVariables(4)
# Add linear costs. To show that calling AddLinearCosts results in the sum of each individual
# cost, we add two costs -3x[0] - x[1] and -5x[2]-x[3]+2
prog.AddLinearCost(-3*x[0] -x[1])
prog.AddLinearCost(-5*x[2] - x[3] + 2)
# Add linear equality constraint 3x[0] + x[1] + 2x[2] == 30
prog.AddLinearConstraint(3*x[0] + x[1] + 2*x[2] == 30)
# Add Linear inequality constraints
prog.AddLinearConstraint(2*x[0] + x[1] + 3*x[2] + x[3] >= 15)
prog.AddLinearConstraint(2*x[1] + 3*x[3] <= 25)
# Add linear inequality constraint -100 <= x[0] + 2x[2] <= 40
prog.AddLinearConstraint(A=[[1., 2.]], lb=[-100], ub=[40], vars=[x[0], x[2]])
prog.AddBoundingBoxConstraint(0, np.inf, x)
prog.AddLinearConstraint(x[1] <= 10)

# Now solve the program.
result = Solve(prog)
print(f"Is solved successfully: {result.is_success()}")
print(f"x optimal value: {result.GetSolution(x)}")
print(f"optimal cost: {result.get_optimal_cost()}")