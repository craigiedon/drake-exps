from ipywidgets.widgets.widget_layout import Layout
import meshcat
import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import (
    AddMultibodyPlantSceneGraph, AngleAxis, BasicVector, ConnectMeshcatVisualizer, 
    DiagramBuilder, FindResourceOrThrow, Integrator, JacobianWrtVariable, 
    LeafSystem, MultibodyPlant, MultibodyPositionToGeometryPose, Parser, 
    PiecewisePolynomial, PiecewiseQuaternionSlerp, Quaternion, RigidTransform, 
    RollPitchYaw, RotationMatrix, SceneGraph, Simulator, TrajectorySource
)
from pydrake.examples.manipulation_station import ManipulationStation
from pydrake.forwarddiff import jacobian
from pydrake.multibody.jupyter_widgets import MakeJointSlidersThatPublishOnCallback
from graphviz import Source

# Example showing the kinematic tree for the hand with fingers
def kinematic_tree_example():
    plant = MultibodyPlant(time_step=0.0)
    parser = Parser(plant)

    parser.AddModelFromFile(FindResourceOrThrow(
        "drake/manipulation/models/allegro_hand_description/sdf/allegro_hand_description_right.sdf"
    ))

    parser.AddModelFromFile(FindResourceOrThrow(
        "drake/examples/manipulation_station/models/061_foam_brick.sdf"
    ))

    plant.Finalize()

    g = Source(plant.GetTopologyGraphvizString())
    print(g.format)
    g.view()

# kinematic_tree_example()

def gripper_forward_kinematics_example():
    builder = DiagramBuilder()

    station = builder.AddSystem(ManipulationStation())
    station.SetupClutterClearingStation()
    station.Finalize()

    frames_to_draw = {"iiwa": {"iiwa_link_1", "iiwa_link_2",
                               "iiwa_link_3", "iiwa_link_4", 
                               "iiwa_link_5", "iiwa_link_6", "iiwa_link_7"},
                      "gripper": {"body"}}

    meshcat = ConnectMeshcatVisualizer(builder,
        station.get_scene_graph(),
        output_port=station.GetOutputPort("pose_bundle"),
        frames_to_draw=frames_to_draw,
        axis_length=0.3,
        axis_radius=0.01)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    # xyz = Text(value="", description="gripper position (m): ", layout=Layout(width='500px'), style={'description_width':'initial'})
    # rpy = Text(value="", description="gripper roll-pitch-yaw (rad): ", layout=Layout(width='500px'), style={'description_width':'initial'})
    plant = station.get_multibody_plant()

    gripper = plant.GetBodyByName("body")
    def pose_callback(context):
        pose = plant.EvalBodyPoseInWorld(context, gripper) # Important
        # xyz.value = np.array2string(pose.translation(), formatter={'float': lambda x: "{:3.2f}".format(x)})
        # rpy.value = np.array2string(RollPitchYaw(pose.rotation()).vector(), formatter={'float': lambda x: "{:3.2f}".format(x)})

    meshcat.load()
    MakeJointSlidersThatPublishOnCallback(station.get_multibody_plant(), meshcat, context, my_callback=pose_callback)
    # display(xyz)
    # display(rpy)


gripper_forward_kinematics_example()


def num_positions_velocities_example():
    plant = MultibodyPlant(time_step = 0.0)
    Parser(plant).AddModelFromFile(FindResourceOrThrow(
        "drake/examples/manipulation_station/models/061_foam_brick.sdf"
    ))
    plant.Finalize()

    context=  plant.CreateDefaultContext()
    print(context)

    print(f"plant.num_positions() = {plant.num_positions()}")
    print(f"plant.num_velocities() = {plant.num_velocities()}")


num_positions_velocities_example()


class PseudoInverseController(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()

        self.DeclareVectorInputPort("iiwa_position", BasicVector(7))
        self.DeclareVectorOutputPort("iiwa_velocity", BasicVector(7), 
                                     self.CalcOutput)

    def CalcOutput(self, context, output):
        q = self.get_input_port().Eval(context)
        self._plant.SetPositions(self._plant_context, self._iiwa, q)
        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context, JacobianWrtVariable.kQDot, 
            self._G, [0,0,0], self._W, self._W)
        J_G = J_G[:,0:7] # Ignore gripper terms
        
        V_G_desired = np.array([0,    # rotation about x
                                -.1,  # rotation about y
                                0,    # rotation about z
                                0,    # x
                                -.05, # y
                                -.1]) # z
        v = np.linalg.pinv(J_G).dot(V_G_desired)
        output.SetFromVector(v)

def jacobian_controller_example():
    builder = DiagramBuilder()

    station = builder.AddSystem(ManipulationStation())
    station.SetupClutterClearingStation()
    station.Finalize()

    controller = builder.AddSystem(PseudoInverseController(station.get_multibody_plant()))
    integrator = builder.AddSystem(Integrator(7))

    builder.Connect(controller.get_output_port(), integrator.get_input_port())
    builder.Connect(integrator.get_output_port(), station.GetInputPort("iiwa_position"))
    builder.Connect(station.GetOutputPort("iiwa_position_measured"), controller.get_input_port())

    meshcat = ConnectMeshcatVisualizer(builder, station.get_scene_graph(), output_port=station.GetOutputPort("pose_bundle"))

    diagram = builder.Build()
    simulator = Simulator(diagram)
    station_context = station.GetMyContextFromRoot(simulator.get_mutable_context())
    station.GetInputPort("iiwa_feedforward_torque").FixValue(station_context, np.zeros((7,1)))
    station.GetInputPort("wsg_position").FixValue(station_context, [0.1])

    integrator.GetMyContextFromRoot(simulator.get_mutable_context()).get_mutable_continuous_state_vector().SetFromVector(station.GetIiwaPosition(station_context))

    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(0.01)

    return simulator

simulator = jacobian_controller_example()
simulator.AdvanceTo(5.0)


def grasp_poses_example():
    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step = 0.0)
    parser = Parser(plant, scene_graph)
    grasp = parser.AddModelFromFile(FindResourceOrThrow(
        "drake/manipulation/models/wsg_50_description/sdf/schunk_wsg_50_no_tip.sdf"), "grasp"
    )
    brick = parser.AddModelFromFile(FindResourceOrThrow(
        "drake/examples/manipulation_station/models/061_foam_brick.sdf"
    ))
    plant.Finalize()

    meshcat = ConnectMeshcatVisualizer(builder, scene_graph)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    B_O = plant.GetBodyByName("base_link", brick)
    X_WO = plant.EvalBodyPoseInWorld(plant_context, B_O)

    B_Ggrasp = plant.GetBodyByName("body", grasp)
    p_GgraspO = [0, 0.12, 0]
    R_GgraspO = RotationMatrix.MakeXRotation(np.pi / 2.0).multiply(
        RotationMatrix.MakeZRotation(np.pi / 2.0)
    )

    X_GgraspO = RigidTransform(R_GgraspO, p_GgraspO)
    X_OGgrasp = X_GgraspO.inverse()
    X_WGgrasp = X_WO.multiply(X_OGgrasp)

    plant.SetFreBodyPose(plant_context, B_Ggrasp, X_WGgrasp)
    # Open the fingers, too
    plant.GetJointByName("left_finger_sliding_joint", grasp).set_translation(plant_context, -0.054)
    plant.GetJointByName("right_finger_sliding_joint", grasp).set_translation(plant_context, 0.054)

    meshcat.load()
    diagram.Publish(context)

grasp_poses_example()


def make_gripper_frames(X_G, X_O):
    """
    Takes a partial specification with X_G["initial"] and X_O["initial"] and X_0["goal"], and 
    returns a X_G and times with all of the pick and place frames populated.
    """
    # Define (again) the gripper pose relative to the object when in grasp.
    p_GgraspO = [0, 0.12, 0]
    R_GgraspO = RotationMatrix.MakeXRotation(np.pi/2.0).multiply(
        RotationMatrix.MakeZRotation(np.pi/2.0))
    X_GgraspO = RigidTransform(R_GgraspO, p_GgraspO)
    X_OGgrasp = X_GgraspO.inverse()
    # pregrasp is negative y in the gripper frame (see the figure!).
    X_GgraspGpregrasp = RigidTransform([0, -0.08, 0])

    X_G["pick"] = X_O["initial"].multiply(X_OGgrasp)
    X_G["prepick"] = X_G["pick"].multiply(X_GgraspGpregrasp)
    X_G["place"] = X_O["goal"].multiply(X_OGgrasp)
    X_G["preplace"] = X_G["place"].multiply(X_GgraspGpregrasp)

    # I'll interpolate to a halfway orientation by converting to axis angle and halving the angle
    X_GprepickGpreplace = X_G["prepick"].inverse().multiply(X_G["preplace"])
    angle_axis = X_GprepickGpreplace.rotation().ToAngleAxis()
    X_GprepickGclearance = RigidTransform(AngleAxis(angle=angle_axis.angle()/2.0, axis=angle_axis.axis()),
                                          X_GprepickGpreplace.translation()/2.0 + np.array([0, -0.3, 0]))
    X_G["clearance"] = X_G["prepick"].multiply(X_GprepickGclearance)

    # Not lets set timings
    times = {"initial": 0.0}
    X_GinitialGprepick = X_G["initial"].inverse().multiply(X_G["prepick"])
    times["prepick"] = times["initial"] + 10.0 * \
        np.linalg.norm(X_GinitialGprepick.translation())
    # Allow some time for the gripper to close.
    times["pick_start"] = times["prepick"] + 2.0
    times["pick_end"] = times["pick_start"] + 2.0
    times["postpick"] = times["pick_end"] + 2.0
    time_to_from_clearance = 10.0 * \
        np.linalg.norm(X_GprepickGclearance.translation())
    times["clearance"] = times["postpick"] + time_to_from_clearance
    times["preplace"] = times["clearance"] + time_to_from_clearance
    times["place_start"] = times["preplace"] + 2.0
    times["place_end"] = times["place_start"] + 2.0
    times["postplace"] = times["place_end"] + 2.0

    return X_G, times

X_G = {"initial": RigidTransform(RotationMatrix.MakeXRotation(-np.pi/2.0), [0, -0.25, 0.25])}
X_O = {"initial": RigidTransform(RotationMatrix.MakeZRotation(np.pi/2.0), [-.2, -.75, 0.025]),
       "goal": RigidTransform(RotationMatrix.MakeZRotation(np.pi),[.75, 0, 0.025])}
X_G, times = make_gripper_frames(X_G, X_O)
print(f"Sanity check: The entire maneuver will take {times['postplace']} seconds to execute.")


def make_gripper_position_trajectory(X_G, times):
    """
    Constructs a gripper position trajectory from the plan "sketch".
    """
    # The syntax here is a little ugly; we need to pass in a matrix with the samples in the columns.
    # TODO(russt): Add python bindings for the std::vector constructor.
    traj = PiecewisePolynomial.FirstOrderHold(
        [times["initial"], times["prepick"]], np.vstack([X_G["initial"].translation(), X_G["prepick"].translation()]).T)

    # TODO(russt): I could make this less brittle if I was more careful on the names above, and just look up the pose for every time (in order)
    traj.AppendFirstOrderSegment(times["pick_start"], X_G["pick"].translation())
    traj.AppendFirstOrderSegment(times["pick_end"], X_G["pick"].translation())
    traj.AppendFirstOrderSegment(times["postpick"], X_G["prepick"].translation())
    traj.AppendFirstOrderSegment(times["clearance"], X_G["clearance"].translation())
    traj.AppendFirstOrderSegment(times["preplace"], X_G["preplace"].translation())
    traj.AppendFirstOrderSegment(times["place_start"], X_G["place"].translation())
    traj.AppendFirstOrderSegment(times["place_end"], X_G["place"].translation())
    traj.AppendFirstOrderSegment(times["postplace"], X_G["preplace"].translation())

    return traj

traj_p_G = make_gripper_position_trajectory(X_G, times)


def make_gripper_orientation_trajectory(X_G, times):
    """
    Constructs a gripper position trajectory from the plan "sketch".
    """
    traj = PiecewiseQuaternionSlerp();
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

traj_R_G = make_gripper_orientation_trajectory(X_G, times)

opened = np.array([0.107])
closed = np.array([0.0])

def make_wsg_command_trajectory(times):
    traj_wsg_command = PiecewisePolynomial.FirstOrderHold(
        [times["initial"], times["pick_start"]], np.hstack([[opened], [opened]]))
    traj_wsg_command.AppendFirstOrderSegment(times["pick_end"], closed)
    traj_wsg_command.AppendFirstOrderSegment(times["place_start"], closed)
    traj_wsg_command.AppendFirstOrderSegment(times["place_end"], opened)
    traj_wsg_command.AppendFirstOrderSegment(times["postplace"], opened)
    return traj_wsg_command

traj_wsg_command = make_wsg_command_trajectory(times)


class GripperTrajectoriesToPosition(LeafSystem):
    def __init__(self, plant, traj_p_G, traj_R_G, traj_wsg_command):
        LeafSystem.__init__(self)
        self.plant = plant
        self.gripper_body = plant.GetBodyByName("body")
        self.left_finger_joint = plant.GetJointByName("left_finger_sliding_joint")
        self.right_finger_joint = plant.GetJointByName("right_finger_sliding_joint")
        self.traj_p_G = traj_p_G
        self.traj_R_G = traj_R_G
        self.traj_wsg_command = traj_wsg_command
        self.plant_context = plant.CreateDefaultContext()

        self.DeclareVectorOutputPort("position", 
                                     BasicVector(plant.num_positions()),
                                     self.CalcPositionOutput)

    def CalcPositionOutput(self, context, output):
        t = context.get_time()
        X_G = RigidTransform(Quaternion(self.traj_R_G.value(t)), self.traj_p_G.value(t))
        self.plant.SetFreeBodyPose(self.plant_context, self.gripper_body, X_G)
        wsg = self.traj_wsg_command.value(t)
        self.left_finger_joint.set_translation(self.plant_context, -wsg/2.0)
        self.right_finger_joint.set_translation(self.plant_context, wsg/2.0)
        output.SetFromVector(self.plant.GetPositions(self.plant_context))

# We can write a new System by deriving from the LeafSystem class.
# There is a little bit of boiler plate, but hopefully this example makes sense.
class PseudoInverseController(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()

        self.w_G_port = self.DeclareVectorInputPort("omega_WG", BasicVector(3))
        self.v_G_port = self.DeclareVectorInputPort("v_WG", BasicVector(3))
        self.q_port = self.DeclareVectorInputPort("iiwa_position", BasicVector(7))
        self.DeclareVectorOutputPort("iiwa_velocity", BasicVector(7), 
                                     self.CalcOutput)
        # TODO(russt): Add missing binding
        #joint_indices = plant.GetJointIndices(self._iiwa)
        #self.position_start = plant.get_joint(joint_indices[0]).position_start()
        #self.position_end = plant.get_joint(joint_indices[-1]).position_start()
        self.iiwa_start = plant.GetJointByName("iiwa_joint_1").velocity_start()
        self.iiwa_end = plant.GetJointByName("iiwa_joint_7").velocity_start()

    def CalcOutput(self, context, output):
        w_G = self.w_G_port.Eval(context)
        v_G = self.v_G_port.Eval(context)
        V_G = np.hstack([w_G, v_G])
        q = self.q_port.Eval(context)
        self._plant.SetPositions(self._plant_context, self._iiwa, q)
        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context, JacobianWrtVariable.kV, 
            self._G, [0,0,0], self._W, self._W)
        J_G = J_G[:,self.iiwa_start:self.iiwa_end+1] # Only iiwa terms.
        v = np.linalg.pinv(J_G).dot(V_G)
        output.SetFromVector(v)

X_O = {"initial": RigidTransform(RotationMatrix.MakeZRotation(np.pi/2.0), [-.2, -.65, 0.09]),
       "goal": RigidTransform(RotationMatrix.MakeZRotation(np.pi),[.5, 0, 0.09])}

builder = DiagramBuilder()

station = builder.AddSystem(ManipulationStation())
station.SetupClutterClearingStation()
station.AddManipulandFromFile(
    "drake/examples/manipulation_station/models/061_foam_brick.sdf",
    X_O["initial"])
station.Finalize()

# Find the initial pose of the gripper (as set in the default Context)
temp_context = station.CreateDefaultContext()
plant = station.get_multibody_plant()
temp_plant_context = plant.GetMyContextFromRoot(temp_context)
X_G = {"initial": plant.EvalBodyPoseInWorld(temp_plant_context, plant.GetBodyByName("body"))}
X_G, times = make_gripper_frames(X_G, X_O)
print(f"Sanity check: The entire maneuver will take {times['postplace']} seconds to execute.")

# Make the trajectories
traj_p_G = make_gripper_position_trajectory(X_G, times)
traj_v_G = traj_p_G.MakeDerivative()
traj_R_G = make_gripper_orientation_trajectory(X_G, times)
traj_w_G = traj_R_G.MakeDerivative()

v_G_source = builder.AddSystem(TrajectorySource(traj_v_G))
v_G_source.set_name("v_WG")
w_G_source = builder.AddSystem(TrajectorySource(traj_w_G))
w_G_source.set_name("omega_WG")
controller = builder.AddSystem(PseudoInverseController(plant))
controller.set_name("PseudoInverseController")
builder.Connect(v_G_source.get_output_port(), controller.GetInputPort("v_WG"))
builder.Connect(w_G_source.get_output_port(), controller.GetInputPort("omega_WG"))

integrator = builder.AddSystem(Integrator(7))
integrator.set_name("integrator")
builder.Connect(controller.get_output_port(), 
                integrator.get_input_port())
builder.Connect(integrator.get_output_port(),
                station.GetInputPort("iiwa_position"))
builder.Connect(station.GetOutputPort("iiwa_position_measured"),
                controller.GetInputPort("iiwa_position"))

traj_wsg_command = make_wsg_command_trajectory(times)
wsg_source = builder.AddSystem(TrajectorySource(traj_wsg_command))
wsg_source.set_name("wsg_command")
builder.Connect(wsg_source.get_output_port(), station.GetInputPort("wsg_position"))

meshcat = ConnectMeshcatVisualizer(builder,
    station.get_scene_graph(),
    output_port=station.GetOutputPort("pose_bundle"),
#    delete_prefix_on_load=False,  # Use this if downloading is a pain.
    zmq_url=zmq_url,
)

diagram = builder.Build()
diagram.set_name("pick_and_place")

simulator = Simulator(diagram)
station_context = station.GetMyContextFromRoot(simulator.get_mutable_context())
# TODO(russt): Add this missing python binding
#integrator.set_integral_value(
#    integrator.GetMyContextFromRoot(simulator.get_mutable_context()), 
#        station.GetIiwaPosition(station_context))
integrator.GetMyContextFromRoot(simulator.get_mutable_context()).get_mutable_continuous_state_vector().SetFromVector(station.GetIiwaPosition(station_context))

simulator.set_target_realtime_rate(1.0)
meshcat.start_recording()
simulator.AdvanceTo(traj_p_G.end_time() if running_as_notebook else 0.1)
meshcat.stop_recording()
meshcat.publish_recording()