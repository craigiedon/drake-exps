from pydrake.symbolic import Variable
from pydrake.systems.primitives import SymbolicVectorSystem
from pydrake.systems.framework import BasicVector, LeafSystem
import matplotlib.pyplot as plt
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogOutput

# Define a new symbolic Variable
x = Variable("x")

# Define the system
continuous_vector_system = SymbolicVectorSystem(state=[x], dynamics=[-x + x ** 3], output=[x])

# Discrete time system. Note the additional argument specifying the time period
discrete_vector_system = SymbolicVectorSystem(state=[x], dynamics=[x**3], output=[x], time_period=1.0)


# More complex system? Derive from LeafSystem
# Not that in this simple form, this current system does not support autodiff

class SimpleContinuousTimeSystem(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)

        self.DeclareContinuousState(1) # One state variable
        self.DeclareVectorOutputPort("y", BasicVector(1), self.CopyStateOut) # One output

    # xdot(t) = -x(t) + x^3(t)
    def DoCalcTimeDerivatives(self, context, derivatives):
        x = context.get_continuous_state_vector().GetAtIndex(0)
        xdot = -x + x**3
        derivatives.get_mutable_vector().SetAtIndex(0, xdot)

    # y = x
    def CopyStateOut(self, context, output):
        x = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(x)

continuous_system = SimpleContinuousTimeSystem()

# Simulation

# Create a simple block diagram containig our system
builder = DiagramBuilder()
# system = builder.AddSystem(SimpleContinuousTimeSystem())
system = builder.AddSystem(continuous_vector_system)
logger = LogOutput(system.get_output_port(0), builder)
diagram = builder.Build()

# Set the initial conditions, x(0)
context = diagram.CreateDefaultContext()
context.SetContinuousState([0.9])

# Create the simulator, and simulate for 10 seconds
simulator = Simulator(diagram, context)
simulator.AdvanceTo(10)

# Plot the results
plt.figure()
plt.plot(logger.sample_times(), logger.data().transpose())
plt.xlabel('t')
plt.ylabel('y(t)')
plt.show()