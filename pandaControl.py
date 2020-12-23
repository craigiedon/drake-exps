# Set up the drake boilerplate system and vis (maybe just multibody plant without "manipulation station"?)
import numpy as np
from pydrake.common.eigen_geometry import AngleAxis, Quaternion
from pydrake.common.value import Value, AbstractValue
from pydrake.manipulation.planner import DifferentialInverseKinematicsIntegrator, DifferentialInverseKinematicsParameters
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.systems.analysis import Simulator
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.systems.framework import DiagramBuilder, LeafSystem, BasicVector
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.geometry import DrakeVisualizer, SceneGraph
from pydrake.common import FindResourceOrThrow
from pydrake.geometry.render import (MakeRenderEngineVtk, RenderEngineVtkParams)
from graphviz import Source
from pydrake.systems.primitives import Integrator, Demultiplexer, TrajectorySource, Multiplexer, LogOutput
from pydrake.systems.primitives import StateInterpolatorWithDiscreteDerivative
from pydrake.systems.rendering import MultibodyPositionToGeometryPose
from pydrake.trajectories import PiecewisePolynomial, PiecewiseQuaternionSlerp


def make_gripper_position_trajectory(X_G, times):
    """ Constructs a gripper position trajectory from the plan "sketch" """
    traj = PiecewisePolynomial.FirstOrderHold(
        [times["initial"], times["prepick"]], np.vstack([X_G["initial"].translation(), X_G["prepick"].translation()]).T
        # [times["initial"], times["prepick"]], np.vstack([X_G["initial"].translation(), X_G["initial"].translation()]).T
    )

    traj.AppendFirstOrderSegment(times["pick_start"], X_G["pick"].translation())
    traj.AppendFirstOrderSegment(times["pick_end"], X_G["pick"].translation())
    traj.AppendFirstOrderSegment(times["postpick"], X_G["prepick"].translation())
    traj.AppendFirstOrderSegment(times["clearance"], X_G["clearance"].translation())
    traj.AppendFirstOrderSegment(times["preplace"], X_G["preplace"].translation())
    traj.AppendFirstOrderSegment(times["place_start"], X_G["place"].translation())
    traj.AppendFirstOrderSegment(times["place_end"], X_G["place"].translation())
    traj.AppendFirstOrderSegment(times["postplace"], X_G["preplace"].translation())

    return traj


def make_gripper_orientation_trajectory(X_G, times):
    """ Constructs a gripper oreintation trajectory from the plant "sketch" """
    traj = PiecewiseQuaternionSlerp()

    traj.Append(times["initial"], X_G["initial"].rotation())
    traj.Append(times["prepick"], X_G["prepick"].rotation())
    traj.Append(times["pick_start"], X_G["pick"].rotation())
    traj.Append(times["pick_end"], X_G["pick"].rotation())
    traj.Append(times["postpick"], X_G["prepick"].rotation())
    traj.Append(times["clearance"], X_G["clearance"].rotation())
    traj.Append(times["preplace"], X_G["preplace"].rotation())
    traj.Append(times["place_start"], X_G["place"].rotation())
    traj.Append(times["place_end"], X_G["place"].rotation())
    traj.Append(times["postplace"], X_G["preplace"].rotation())

    return traj


def make_finger_trajectory(times):
    opened = np.array([0.08])
    closed = np.array([0.00])
    traj_hand_command = PiecewisePolynomial.FirstOrderHold(
        [times["initial"], times["pick_start"]], np.hstack([[opened], [opened]]))
    traj_hand_command.AppendFirstOrderSegment(times["pick_end"], closed)
    traj_hand_command.AppendFirstOrderSegment(times["place_start"], closed)
    traj_hand_command.AppendFirstOrderSegment(times["place_end"], opened)
    traj_hand_command.AppendFirstOrderSegment(times["postplace"], opened)

    return traj_hand_command


class PseudoInverseController(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._panda = plant.GetModelInstanceByName("panda")
        self._G = plant.GetBodyByName("panda_hand").body_frame()
        self._W = plant.world_frame()

        self.DeclareVectorInputPort("panda_position", BasicVector(9))
        self.DeclareVectorOutputPort("panda_velocity", BasicVector(9), self.CalcOutput)

    def CalcOutput(self, context, output):
        q = self.get_input_port().Eval(context)
        self._plant.SetPositions(self._plant_context, self._panda, q)
        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context, JacobianWrtVariable.kQDot, self._G, [0, 0, 0], self._W, self._W
        )
        J_G = J_G[:, 0:7]  # ignore gripper terms
        V_G_desired = np.array([0.0, 0.0, 0, 0, 0.0, -0.1])  # rot-x, rot-y, rot-z, x, y, z
        v = np.linalg.pinv(J_G).dot(V_G_desired)
        v_and_fingers = np.concatenate((v, [0.0, 0.0]))
        output.SetFromVector(v_and_fingers)


class PseudoInverseController2(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._panda = plant.GetModelInstanceByName("panda")
        self._G = plant.GetBodyByName("panda_hand").body_frame()
        self._W = plant.world_frame()

        # Desired Rotational Velocity of gripper
        self.w_G_port = self.DeclareVectorInputPort("omega_WG", BasicVector(3))

        # Desired Translational Velocity of gripper
        self.v_G_port = self.DeclareVectorInputPort("v_WG", BasicVector(3))

        # Current configuration of the panda arm
        self.q_port = self.DeclareVectorInputPort("panda_position", BasicVector(9))

        # Output commanded velocity of arm
        self.DeclareVectorOutputPort("panda_velocity", BasicVector(7), self.CalcOutput)

        self.panda_start = plant.GetJointByName("panda_joint1").velocity_start()
        self.panda_end = plant.GetJointByName("panda_joint7").velocity_start()

    def CalcOutput(self, context, output):
        w_G = self.w_G_port.Eval(context)
        v_G = self.v_G_port.Eval(context)
        V_G = np.hstack([w_G, v_G])
        q = self.q_port.Eval(context)
        self._plant.SetPositions(self._plant_context, self._panda, q)
        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context, JacobianWrtVariable.kV,
            self._G, [0, 0, 0], self._W, self._W)
        J_G = J_G[:, self.panda_start:self.panda_end + 1]  # Only panda terms
        v = np.linalg.pinv(J_G).dot(V_G)
        output.SetFromVector(v)


class TrajToRB(LeafSystem):
    def __init__(self, traj_pos, traj_rot):
        LeafSystem.__init__(self)
        self.traj_pos = traj_pos
        self.traj_rot = traj_rot
        self.DeclareAbstractOutputPort(Value[RigidTransform], self.CalcOutput)

    def CalcOutput(self, context, output):
        t = context.get_time()
        pos_vec = self.traj_pos.value(t)
        rot_mat_vec = self.traj_rot.value(t)

        rb = RigidTransform(Quaternion(rot_mat_vec), pos_vec)
        output.SetFrom(Value[RigidTransform](rb))


class GripperTrajectoriesToPosition(LeafSystem):
    def __init__(self, plant, traj_hand):
        LeafSystem.__init__(self)
        self.plant = plant
        self.gripper_body = plant.GetBodyByName("panda_hand")
        self.left_finger_joint = plant.GetJointByName("panda_finger_joint1")
        self.right_finger_joint = plant.GetJointByName("panda_finger_joint2")
        self.traj_hand = traj_hand
        self.plant_context = plant.CreateDefaultContext()

        self.DeclareVectorOutputPort("finger_position", BasicVector(2), self.CalcPositionOutput)

    def CalcPositionOutput(self, context, output):
        t = context.get_time()
        hand_command = self.traj_hand.value(t)
        self.left_finger_joint.set_translation(self.plant_context, hand_command / 2.0)
        self.right_finger_joint.set_translation(self.plant_context, hand_command / 2.0)
        output.SetFromVector(self.plant.GetPositions(self.plant_context)[-2:])


builder = DiagramBuilder()

# A multibody plant is a *system* (in the drake sense) holding all the info for multi-body rigid bodies, providing ports for forces and continuous state, geometry etc.
# The "builder" as an argument is a kind of a "side-effectey" thing, we want to add our created multibodyplant *to* this builder
# The 0.0 is the "discrete time step". A value of 0.0 means that we have made this a continuous system
# Note also, this constructor is overloaded (but this is not a thing you can do in python naturally. It is an artefact of the C++ port)
time_step = 0.002
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

# Gripper pose relative to object when in grasp
p_GgraspO = [0, 0, 0.13]
R_GgraspO = RotationMatrix.MakeXRotation(np.pi)

X_GgraspO = RigidTransform(R_GgraspO, p_GgraspO)
X_OGgrasp = X_GgraspO.inverse()

# Pregrasp is negative z in the gripper frame
X_GgraspGpregrasp = RigidTransform([0, 0.0, -0.08])

X_O = {"initial": RigidTransform([0.65, 0.5, 0.015]),
       "goal": RigidTransform(RotationMatrix.MakeZRotation(np.pi / 2.0), [0.0, 0.0, 0.015])}

temp_context = plant.CreateDefaultContext()
temp_plant_context = plant.GetMyContextFromRoot(temp_context)
desired_initial_state = np.array([0.0, 0.3, 0.0, -1.3, 0.0, 1.65, 0.9, 0.040, 0.040])
plant.SetPositions(temp_plant_context, panda, desired_initial_state)

X_G = {"initial": plant.EvalBodyPoseInWorld(temp_plant_context, plant.GetBodyByName("panda_hand"))}

# X_G = {"initial": RigidTransform(RotationMatrix.MakeXRotation(np.pi), [0, -0.25, 0.25])}
X_G["pick"] = X_O["initial"].multiply(X_OGgrasp)
X_G["prepick"] = X_G["pick"].multiply(X_GgraspGpregrasp)
X_G["place"] = X_O["goal"].multiply(X_OGgrasp)
X_G["preplace"] = X_G["place"].multiply(X_GgraspGpregrasp)

# Interpolate a halfway orientation by converting to axis angle and halving angle
X_GprepickGpreplace = X_G["prepick"].inverse().multiply(X_G["preplace"])
angle_axis = X_GprepickGpreplace.rotation().ToAngleAxis()
X_GprepickGclearance = RigidTransform(AngleAxis(angle=angle_axis.angle() / 2.0, axis=angle_axis.axis()),
                                      X_GprepickGpreplace.translation() / 2.0 + np.array([0, 0.0, -0.3]))
X_G["clearance"] = X_G["prepick"].multiply(X_GprepickGclearance)

# Precise timings of trajectory
times = {"initial": 0}
X_GinitialGprepick = X_G["initial"].inverse().multiply(X_G["prepick"])
times["prepick"] = times["initial"] + 10.0 * np.linalg.norm(X_GinitialGprepick.translation())

# Allow some time for gripper to close
times["pick_start"] = times["prepick"] + 2.0
times["pick_end"] = times["pick_start"] + 2.0
times["postpick"] = times["pick_end"] + 2.0
time_to_from_clearance = 10.0 * np.linalg.norm(X_GprepickGclearance.translation())
times["clearance"] = times["postpick"] + time_to_from_clearance
times["preplace"] = times["clearance"] + time_to_from_clearance
times["place_start"] = times["preplace"] + 2.0
times["place_end"] = times["place_start"] + 2.0
times["postplace"] = times["place_end"] + 2.0

traj_p_G = make_gripper_position_trajectory(X_G, times)
traj_R_G = make_gripper_orientation_trajectory(X_G, times)
traj_h = make_finger_trajectory(times)


controller_plant = MultibodyPlant(time_step)
controller_parser = Parser(controller_plant)
control_only_panda = controller_parser.AddModelFromFile(panda_arm_hand_file)
controller_plant.WeldFrames(controller_plant.world_frame(), controller_plant.GetFrameByName("panda_link0", control_only_panda), X_AB=RigidTransform(np.array([0.0, 0.5, 0.0])))
controller_plant.Finalize()

diff_inv_controller = builder.AddSystem(
    DifferentialInverseKinematicsIntegrator(controller_plant,
                                            controller_plant.GetFrameByName("panda_hand", control_only_panda),
                                            time_step,
                                            DifferentialInverseKinematicsParameters(num_positions=9, num_velocities=9))
)
diff_inv_controller.set_name("Inverse Kinematics")

rb_conv = builder.AddSystem(TrajToRB(traj_p_G, traj_R_G))
rb_conv.set_name("RB Conv")
builder.Connect(rb_conv.get_output_port(), diff_inv_controller.get_input_port())

diff_arm_demux = builder.AddSystem(Demultiplexer(np.array([7, 2])))
diff_arm_demux.set_name("Diff Arm Demux")
builder.Connect(diff_inv_controller.get_output_port(), diff_arm_demux.get_input_port())

kp = np.full(9, 100)
ki = np.full(9, 1)
kd = 2 * np.sqrt(kp)

inv_d_controller = builder.AddSystem(InverseDynamicsController(controller_plant, kp, ki, kd, False))
inv_d_controller.set_name("inv_d")
builder.Connect(plant.get_state_output_port(panda), inv_d_controller.get_input_port_estimated_state())

hand_comms = builder.AddSystem(GripperTrajectoriesToPosition(controller_plant, traj_h))

arm_hand_mux = builder.AddSystem(Multiplexer(np.array([7, 2])))
arm_hand_mux.set_name("Arm-Hand Mux")
builder.Connect(diff_arm_demux.get_output_port(0), arm_hand_mux.get_input_port(0))
builder.Connect(hand_comms.get_output_port(), arm_hand_mux.get_input_port(1))

# builder.Connect(s_interp.get_output_port(), arm_hand_mux.get_input_port(0))
# builder.Connect(h_interp.get_output_port(), arm_hand_mux.get_input_port(1))

# arm_hand_mux.get_input_port(1).FixValue(np.array([0.0, 0.0, 0.0, 0.0]))
# Connect result of integrator to a derivative getter
s_interp = builder.AddSystem(StateInterpolatorWithDiscreteDerivative(9, time_step, True))
s_interp.set_name("s_interp")
builder.Connect(arm_hand_mux.get_output_port(), s_interp.get_input_port())
builder.Connect(s_interp.get_output_port(), inv_d_controller.GetInputPort("desired_state"))

# Connect inverse dynamics control into desired actuation of panda
builder.Connect(inv_d_controller.get_output_port_control(), plant.get_actuation_input_port())

logger = LogOutput(diff_inv_controller.get_output_port(), builder)

diagram = builder.Build()
diagram.set_name("pick_and_place")

simulator = Simulator(diagram)
sim_context = simulator.get_mutable_context()

plant_context = plant.GetMyMutableContextFromRoot(sim_context)
plant.SetPositions(plant_context, panda, desired_initial_state)
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("base_link", brick), X_O["initial"])

# integrator.GetMyContextFromRoot(simulator.get_mutable_context()) \
#     .get_mutable_continuous_state_vector() \
#     .SetFromVector(plant.GetPositions(plant_context, panda)[:7])
diff_inv_controller.SetPositions(
    diff_inv_controller.GetMyMutableContextFromRoot(simulator.get_mutable_context()),
    plant.GetPositions(plant_context, panda))
print(plant.GetPositions(plant_context, panda))

# arm_hand_mux.get_input_port(1).FixValue(
#     arm_hand_mux.GetMyMutableContextFromRoot(simulator.get_mutable_context()),
#     np.array([0.20, 0.20, 1.0, 1.0]))

simulator.Initialize()
simulator.set_target_realtime_rate(2.0)
simulator.AdvanceTo(traj_p_G.end_time())
# simulator.AdvanceTo(0.3)

log_data = logger.data()
# print(log_data[:, 1])

# g = Source(diagram.GetGraphvizString(max_depth=1))
# g.view()
