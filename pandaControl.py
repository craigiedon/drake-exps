# Set up the drake boilerplate system and vis (maybe just multibody plant without "manipulation station"?)
import numpy as np
from pydrake.math import RigidTransform
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.systems.analysis import Simulator
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.systems.framework import DiagramBuilder, LeafSystem, BasicVector
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.geometry import DrakeVisualizer
from pydrake.common import FindResourceOrThrow
from pydrake.geometry.render import (MakeRenderEngineVtk, RenderEngineVtkParams)
from graphviz import Source
from pydrake.systems.primitives import Integrator


def kinematic_chain_diagram(p):
    g = Source(p.GetTopologyGraphvizString())
    print(g.format)
    g.view()


class PseudoInverseController(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._panda = plant.GetModelInstanceByName("panda")
        self._G = plant.GetBodyByName("panda_hand").body_frame()
        self._W = plant.world_frame()

        self.DeclareVectorInputPort("panda_position", BasicVector(7))
        self.DeclareVectorOutputPort("panda_velocity", BasicVector(7), self.CalcOutput)

    def CalcOutput(self, context, output):
        q = self.get_input_port().Eval(context)
        self._plant.SetPositions(self._plant_context, self._panda, q)
        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context, JacobianWrtVariable.kQDot, self._G, [0, 0, 0], self._W, self._W
        )
        J_G = J_G[:, 0:7] # ignore gripper terms
        V_G_desired = np.array([0, -0.1, 0, 0, -0.05, -0.1]) # rot-x, rot-y, rot-z, x, y, z
        v = np.linalg.pinv(J_G).dot(V_G_desired)
        output.SetFromVector(v)



builder = DiagramBuilder()

# A multibody plant is a *system* (in the drake sense) holding all the info for multi-body rigid bodies, providing ports for forces and continuous state, geometry etc.
# The "builder" as an argument is a kind of a "side-effectey" thing, we want to add our created multibodyplant *to* this builder
# The 0.0 is the "discrete time step". A value of 0.0 means that we have made this a continuous system
# Note also, this constructor is overloaded (but this is not a thing you can do in python naturally. It is an artefact of the C++ port)
time_step = 1e-4
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step)

# Load in the panda
# The parser takes a multibody plant mutably, so that anything parsed with it gets automatically added to this multibody system
parser = Parser(plant)

# Files are getting grabbed from "/opt/drake/share/drake/...
panda_arm_hand_file = FindResourceOrThrow("drake/manipulation/models/franka_description/urdf/panda_arm_hand.urdf")
brick_file = FindResourceOrThrow("drake/examples/manipulation_station/models/061_foam_brick.sdf")
bin_file = FindResourceOrThrow("drake/examples/manipulation_station/models/bin.sdf")

# Actually parse in the model
panda = parser.AddModelFromFile(panda_arm_hand_file, model_name="panda")

# Don't want panda to drop through the sky or fall over so...
# It would be nice if there was an option to create a floor...
# The X_AB part is "relative pose" (monogram notation: The pose (X) of B relative to A
plant.WeldFrames(
    plant.world_frame(), plant.GetFrameByName("panda_link0", panda), X_AB=RigidTransform(np.array([0.0, 0.5, 0.0]))
)

# Add a little example brick
brick = parser.AddModelFromFile(brick_file, model_name="brick")
# plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link"))

# Add some bins
bin_1 = parser.AddModelFromFile(bin_file, model_name="bin_1")
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("bin_base", bin_1))

bin_2 = parser.AddModelFromFile(bin_file, model_name="bin_2")
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("bin_base", bin_2), X_AB=RigidTransform(np.array([0.65, 0.5, 0.0])))

scene_graph.AddRenderer("renderer", MakeRenderEngineVtk(RenderEngineVtkParams()))
DrakeVisualizer.AddToBuilder(builder, scene_graph)

# Important to do tidying work
plant.Finalize()

# Add equivalent of a "hold in place against natural forces" controller
kp = np.full(9, 100)
ki = 2 * np.sqrt(kp)
kd = np.full(9, 1)

controller_plant = MultibodyPlant(time_step)
controller_parser = Parser(controller_plant)
control_only_panda = controller_parser.AddModelFromFile(panda_arm_hand_file)
controller_plant.WeldFrames(controller_plant.world_frame(), controller_plant.GetFrameByName("panda_link0", control_only_panda))
controller_plant.Finalize()


pseudo_inv_controller = PseudoInverseController(plant)
integrator = builder.AddSystem(Integrator(7))

builder.Connect(pseudo_inv_controller.get_output_port(), integrator.get_input_port())
builder.Connect(integrator.get_output_port(), control_only_panda.GetInputPort(""))

builder.Connect(plant.get_state_output_port(panda), pseudo_inv_controller.get_input_port())
# invd_controller = InverseDynamicsController(controller_plant, kp, ki, kd, False)
# panda_controller = builder.AddSystem(invd_controller)
# panda_controller.set_name("panda_controller")
#
# # Feed the controller the current state of the panda
# builder.Connect(
#     plant.get_state_output_port(panda),
#     panda_controller.get_input_port_estimated_state()
# )
# # Feed the controller's output commands into the panda's actuation port
# builder.Connect(panda_controller.get_output_port_control(),
#                 plant.get_actuation_input_port())

diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()

# Get the context by searching for it in the top level simulator context?
plant_context = plant.GetMyMutableContextFromRoot(diagram_context)
# print(plant_context)

# Provide fixed value for the actuator ports *in the given context*
desired_pos = np.array([0.0, 0.3, 0.0, -1.3, 0.0, 1.65, 0.9, 0.02, 0.02])
desired_state = np.hstack((desired_pos, 0.0 * desired_pos))
plant.SetPositions(plant_context, panda, desired_pos)

# Set the position of the brick
body_indices = plant.GetBodyIndices(brick)
print(plant.get_body(body_indices[0]))
# plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("base_link", brick), RigidTransform(np.array([0.0, 0.0, 0.015])))
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("base_link", brick), RigidTransform(np.array([0.0, 0.0, 0.5])))

# Set a fixed desired state so that it stays in place
panda_controller.GetInputPort('desired_state').FixValue(
    panda_controller.GetMyMutableContextFromRoot(diagram_context), desired_state)

# print(diagram_context)
# kinematic_chain_diagram(plant)

G = plant.GetBodyByName("panda_hand").body_frame()
W = plant.world_frame()
J_G = plant.CalcJacobianSpatialVelocity(plant_context, JacobianWrtVariable.kQDot, G, [0.0, 0.0, 0.0], W, W)


# Set actuation to 0 (floppy arm)
# plant.get_actuation_input_port(panda).FixValue(plant_context, np.zeros((9, 1)))

# This slows it down to roughly real time. Without it, it just blasts through simulation as fast as possible
print(diagram_context.num_total_states())
simulator = Simulator(diagram, diagram_context)
simulator.Initialize()
simulator.get_mutable_context().SetTime(0.0)
simulator.set_target_realtime_rate(1.0)
simulator.AdvanceTo(2.0)


print(diagram_context.num_total_states())

# plant.

# TODO: Put in the minimum version of the chapter 3 controller in
# TODO: Robustify if with the additional exercises
# TODO: See if it is possible to rip simulator.Initialize()it all out and replace with a pre-built drake control class?
