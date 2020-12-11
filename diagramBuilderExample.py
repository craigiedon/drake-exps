import matplotlib.pyplot as plt
import numpy as np

from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.examples.pendulum import PendulumPlant
from pydrake.systems.controllers import PidController
from pydrake.systems.drawing import plot_system_graphviz
from pydrake.systems.primitives import LogOutput

builder = DiagramBuilder()

# First add the pendulum
pendulum = builder.AddSystem(PendulumPlant())
pendulum.set_name("pendulum")

controller = builder.AddSystem(PidController(kp=[10.0], ki=[1.0], kd=[1.]))
controller.set_name("controller")

# Now "wire up" the controller to the plant
builder.Connect(pendulum.get_state_output_port(), controller.get_input_port_estimated_state())
builder.Connect(controller.get_output_port_control(), pendulum.get_input_port())

# Make the desired_state input of the controller an input to the diagram
builder.ExportInput(controller.get_input_port_desired_state())

# Log the state of the pendulum
logger = LogOutput(pendulum.get_state_output_port(), builder)
logger.set_name("logger")

diagram = builder.Build()
diagram.set_name("diagram")

# plt.figure()
# plot_system_graphviz(diagram, max_depth=2)
# plt.show()

# Set up a simulator to run this diagram
simulator = Simulator(diagram)
context = simulator.get_mutable_context()

# We'll try to regular the pendulum to a particular angle
desired_angle = np.pi / 2.0

# First we extract the subsystem context for the pendulum
pendulum_context = diagram.GetMutableSubsystemContext(pendulum, context)

# Then we can set the pendulum state, which is (theta, thetadot)
pendulum_context.get_mutable_continuous_state_vector().SetFromVector([desired_angle + 0.1, 0.2])

# the diagram has a single input port (port index 0), which is the desired_state
diagram.get_input_port(0).FixValue(context, [desired_angle, 0.0])

# Simulate for number of seconds
simulator.AdvanceTo(20)

# Plot the results
t = logger.sample_times()
plt.figure()

# Plot the theta
plt.plot(t, logger.data()[0,:], '.-')
# Draw a line for the diesred angle
plt.plot([t[0], t[-1]], [desired_angle, desired_angle], 'g' )
plt.xlabel('time (seconds)')
plt.ylabel('theta (rad)')
plt.title('PID Control of the Pendulum');
plt.show()
