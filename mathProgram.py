from pydrake.solvers.mathematicalprogram import MathematicalProgram
import numpy as np
import matplotlib.pyplot as plt
from pydrake.solvers.mathematicalprogram import Solve

prog = MathematicalProgram()

x = prog.NewContinuousVariables(2)

print(x)
print(1 + 2*x[0] + 3*x[1] + 4*x[1])

y = prog.NewContinuousVariables(2, "dog")
print(y)
print(y[0] + y[0] + y[1] * y[1] * y[1])

var_matrix = prog.NewContinuousVariables(3, 2, "A")
print(var_matrix)

# Add the constraint x(0) * x(1) = 1 to prog
prog.AddConstraint(x[0] * x[1] == 1)
prog.AddConstraint(x[0] >= 0)
prog.AddConstraint(x[0] - x[1] <= 0)

prog.AddCost(x[0] ** 2 + 3)
prog.AddCost(x[0] + x[1])

# New optimization program, all the way through to solving
prog = MathematicalProgram()
x = prog.NewContinuousVariables(2)
prog.AddConstraint(x[0] + x[1] == 1)
prog.AddConstraint(x[0] <= x[1])
prog.AddCost(x[0] ** 2 + x[1] ** 2)

# Now solve the optimization problem
result = Solve(prog)

print("Success? ", result.is_success())
print('x* = ', result.GetSolution(x))
print('optimal cost = ', result.get_optimal_cost())
print('solver is: ', result.get_solver_id().name())

# New Program with Callback
fig = plt.figure()
curve_x = np.linspace(1, 10, 100)
ax = plt.gca()
ax.plot(curve_x, 9.0 / curve_x)
ax.plot(-curve_x, -9.0 / curve_x)
ax.plot(0, 0, 'o')
x_init = [4., 5.]
point_x, = ax.plot(x_init[0], x_init[1], 'x')
ax.axis('equal')

def update(x):
    global iter_count
    point_x.set_xdata(x[0])
    point_x.set_ydata(x[1])
    ax.set_title(f"iteration {iter_count}")
    fig.canvas.draw()
    fig.canvas.flush_events()
    # Also update hte iter_count variable in the callback
    # This shows we can do more than just visualization in
    # callback
    iter_count += 1
    plt.pause(0.1)

iter_count = 0
prog = MathematicalProgram()
x = prog.NewContinuousVariables(2)
prog.AddConstraint(x[0] * x[1] == 9)
prog.AddCost(x[0] ** 2 + x[1] ** 2)
prog.AddVisualizationCallback(update, x)
result = Solve(prog, x_init)
plt.show()