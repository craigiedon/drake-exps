import numpy as np
from pydrake.all import (MultibodyPlant, AddMultibodyPlantSceneGraph, FindResourceOrThrow, Parser, Simulator, DiagramBuilder)
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer
from pydrake.geometry import DrakeVisualizer

# plant = MultibodyPlant(time_step=1e-4)
# Parser(plant).AddModelFromFile(
#     FindResourceOrThrow("drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf")
# )

# # Actually bolt the thing to the ground
# plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))
# plant.Finalize()

# # Context: state, parameters, inputs, time
# context = plant.CreateDefaultContext()
# # print(context)

# # Set all of the joint positions at once in a single vector
# plant.SetPositions(context, [-1.57, 0.1, 0, 0, 0, 1.6, 0])

# # You can also set them by referencing particular joints
# plant.GetJointByName("iiwa_joint_4").set_angle(context, -1.2)

# # Set initial actuator values
# plant.get_actuation_input_port().FixValue(context, np.zeros(7))

# simulator = Simulator(plant, context)
# simulator.AdvanceTo(5.0)

builder = DiagramBuilder()
# Adds both MultibodyPlant and SceneGraph and wires them together
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)

# Note that we parse into both the plant and the scene_graph here
Parser(plant, scene_graph).AddModelFromFile(
    FindResourceOrThrow("drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf")
)

plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))

# Adds the MeshcatVisualizer and wires it to the Scene Graph
meshcat = ConnectMeshcatVisualizer(builder, scene_graph, open_browser=True)

plant.Finalize()
diagram = builder.Build()

context = diagram.CreateDefaultContext()
meshcat.load()
diagram.Publish(context)

plant_context = plant.GetMyMutableContextFromRoot(context)
plant.SetPositions(plant_context, [-1.57, 0.1, 0, -1.2, 0, 1.6, 0])
plant.get_actuation_input_port().FixValue(plant_context, np.zeros(7))

simulator = Simulator(diagram, context)
simulator.set_target_realtime_rate(1.0)
simulator.AdvanceTo(5)
