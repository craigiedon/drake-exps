import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from pydrake.common import FindResourceOrThrow
from pydrake.geometry import DrakeVisualizer
from pydrake.geometry.render import (
    ClippingRange,
    DepthRange,
    DepthRenderCamera,
    RenderCameraCore,
    RenderLabel,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
)

from pydrake.geometry import ConnectDrakeVisualizer

from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import BodyIndex
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer
from pydrake.systems.sensors import (
    CameraInfo,
    RgbdSensor,
)

def xyz_rpy_deg(xyz, rpy_deg):
    """Shorthand for defining a pose."""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi / 180), xyz)

reserved_labels = [
    RenderLabel.kDoNotRender,
    RenderLabel.kDontCare,
    RenderLabel.kEmpty,
    RenderLabel.kUnspecified,
]

def colorize_labels(image):
    """Colorizes labels."""
    # TODO(eric.cousineau): Revive and use Kuni's palette.
    cc = mpl.colors.ColorConverter()
    color_cycle = plt.rcParams["axes.prop_cycle"]
    colors = np.array([cc.to_rgb(c["color"]) for c in color_cycle])
    bg_color = [0, 0, 0]
    image = np.squeeze(image)
    background = np.zeros(image.shape[:2], dtype=bool)
    for label in reserved_labels:
        background |= image == int(label)
    color_image = colors[image % len(colors)]
    color_image[background] = bg_color
    return color_image

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)

parser = Parser(plant)
iiwa_file = FindResourceOrThrow(
   "drake/manipulation/models/iiwa_description/sdf/"
   "iiwa14_no_collision.sdf")

iiwa_1 = parser.AddModelFromFile(iiwa_file, model_name="iiwa_1")
plant.WeldFrames(
    plant.world_frame(), plant.GetFrameByName("iiwa_link_0", iiwa_1),
    X_AB=xyz_rpy_deg([0, 0, 0], [0, 0, 0]))

iiwa_2 = parser.AddModelFromFile(iiwa_file, model_name="iiwa_2")
plant.WeldFrames(
    plant.world_frame(), plant.GetFrameByName("iiwa_link_0", iiwa_2),
    X_AB=xyz_rpy_deg([0, 1, 0], [0, 0, 0]))

renderer_name = "renderer"
scene_graph.AddRenderer(
    renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams()))

# N.B. These properties are chosen arbitrarily.
depth_camera = DepthRenderCamera(
    RenderCameraCore(
        renderer_name,
        CameraInfo(width=640, height=480, fov_y=np.pi/4),
        ClippingRange(0.01, 10.0),
        RigidTransform()), 
    DepthRange(0.01, 10.0))

world_id = plant.GetBodyFrameIdOrThrow(plant.world_body().index())
X_WB = xyz_rpy_deg([4, 0, 0], [-90, 0, 90])
sensor = RgbdSensor(
    world_id, X_PB=X_WB,
    depth_camera=depth_camera)
builder.AddSystem(sensor)
builder.Connect(
    scene_graph.get_query_output_port(),
    sensor.query_object_input_port())

DrakeVisualizer.AddToBuilder(builder, scene_graph)

meshcat_vis = ConnectMeshcatVisualizer(builder, scene_graph, zmq_url="new", open_browser=True)


plant.Finalize()
diagram = builder.Build()

diagram_context = diagram.CreateDefaultContext()
sensor_context = sensor.GetMyMutableContextFromRoot(diagram_context)
sg_context = scene_graph.GetMyMutableContextFromRoot(diagram_context)

simulator = Simulator(diagram)
simulator.Initialize()

color = sensor.color_image_output_port().Eval(sensor_context).data
label = sensor.label_image_output_port().Eval(sensor_context).data
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(color)
ax[1].imshow(colorize_labels(label))

query_object = scene_graph.get_query_output_port().Eval(sg_context)
inspector = query_object.inspector()

label_by_model = label.copy()
for geometry_id in inspector.GetAllGeometryIds():
    body = plant.GetBodyFromFrameId(inspector.GetFrameId(geometry_id))
    geometry_label = inspector.GetPerceptionProperties(
        geometry_id).GetProperty("label", "id")
    # WARNING: If you do not cast the `geometry_label` to `int`, this
    # comparison will take a long time since NumPy will do
    # element-by-element comparison using `RenderLabel.__eq__`.
    mask = (label == int(geometry_label))
    label_by_model[mask] = int(body.model_instance())

plt.imshow(colorize_labels(label_by_model))

meshcat_vis.vis.render_static()

# Set the context to make the simulation valid and slightly interesting.
simulator.get_mutable_context().SetTime(.0)
plant_context = plant.GetMyContextFromRoot(simulator.get_mutable_context())
plant.get_actuation_input_port(iiwa_1).FixValue(plant_context, np.zeros((7,1)))
plant.get_actuation_input_port(iiwa_2).FixValue(plant_context, np.zeros((7,1)))
plant.SetPositions(plant_context, iiwa_1, [0.2, 0.4, 0, 0, 0, 0, 0])

# Reset the recording (in case you are running this cell more than once).
meshcat_vis.reset_recording()

# Start recording and simulate.
meshcat_vis.start_recording()
simulator.AdvanceTo(1.0)

# Publish the recording to meshcat.
meshcat_vis.publish_recording()

# Render meshcat's current state.
meshcat_vis.vis.render_static()