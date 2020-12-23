def vis_gripper_frames(X_G, X_O):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-2)
    parser = Parser(plant, scene_graph)
    gripper = FindResourceOrThrow(
        "drake/manipulation/models/franka_description/urdf/hand.urdf"
    )

    schunk_grip = FindResourceOrThrow(
        "drake/manipulation/models/wsg_50_description/sdf/schunk_wsg_50_no_tip.sdf")

    brick = FindResourceOrThrow(
        "drake/examples/manipulation_station/models/061_foam_brick.sdf"
    )

    arm = FindResourceOrThrow(
        "drake/manipulation/models/franka_description/urdf/panda_arm_hand.urdf"
    )

    # g = parser.AddModelFromFile(gripper, f"gripper_standard")
    # plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_hand", g), RigidTransform())

    # sch_g = parser.AddModelFromFile(schunk_grip, f"sch_gripper_standard")
    # plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("body", sch_g), RigidTransform())

    # g = parser.AddModelFromFile(arm, "full_arm")
    # plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0", g), RigidTransform())

    for key, pose in X_G.items():
        g = parser.AddModelFromFile(gripper, f"gripper_{key}")
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_hand", g), pose)

    for key, pose in X_O.items():
        o = parser.AddModelFromFile(brick, f"object_{key}")
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link", o), pose)

    plant.Finalize()

    scene_graph.AddRenderer("renderer", MakeRenderEngineVtk(RenderEngineVtkParams()))
    DrakeVisualizer.AddToBuilder(builder, scene_graph)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    diagram.Publish(context)


def visualize_pick_and_place_trajectory(traj_p_G, traj_R_G, traj_hand_command, X_O):
    builder = DiagramBuilder()

    # Note: Don't use AddMultibodyPlantSceneGraph because we are only using
    # MultibodyPlant for parsing, then wiring directly to SceneGraph
    scene_graph = builder.AddSystem(SceneGraph())
    plant = MultibodyPlant(time_step=1e-4)
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    parser = Parser(plant, scene_graph)
    gripper = parser.AddModelFromFile(FindResourceOrThrow(
        "drake/manipulation/models/franka_description/urdf/hand.urdf"
    ))
    brick = FindResourceOrThrow(
        "drake/examples/manipulation_station/models/061_foam_brick.sdf"
    )

    for key, pose in X_O.items():
        o = parser.AddModelFromFile(brick, f"object_{key}")
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link", o), pose)
    plant.Finalize()

    to_pose = builder.AddSystem(MultibodyPositionToGeometryPose(plant))
    builder.Connect(to_pose.get_output_port(), scene_graph.get_source_pose_port(plant.get_source_id()))

    traj_to_position = builder.AddSystem(GripperTrajectoriesToPosition(plant, traj_p_G, traj_R_G, traj_hand_command))
    builder.Connect(traj_to_position.get_output_port(), to_pose.get_input_port())

    scene_graph.AddRenderer("renderer", MakeRenderEngineVtk(RenderEngineVtkParams()))
    DrakeVisualizer.AddToBuilder(builder, scene_graph)

    diagram = builder.Build()

    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(3.0)
    simulator.AdvanceTo(traj_p_G.end_time())

def kinematic_chain_diagram(p):
    g = Source(p.GetTopologyGraphvizString())
    print(g.format)
    g.view()
