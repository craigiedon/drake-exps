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


class BabySystem(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self.DeclareVectorInputPort("the_ins", BasicVector(1))
        self.DeclareVectorOutputPort("the_outs", BasicVector(1),
                                     self.CalcOutput)

    def CalcOutput(self, context, output):
        print(output)
        output.SetFromVector(np.array([0.1]))


builder = DiagramBuilder()
controller = builder.AddSystem(BabySystem())
integrator = builder.AddSystem(Integrator(1))

builder.Connect(controller.get_output_port(), integrator.get_input_port())
# builder.Connect(integrator)

diagram = builder.Build()
context = diagram.CreateDefaultContext()
print(context)


simulator = Simulator(diagram)

integrator.GetMyContextFromRoot(simulator.get_mutable_context()).get_mutable_continuous_state_vector().SetFromVector(np.array([1.0]))


simulator.Initialize()
# simulator.set_target_realtime_rate(1.0)
simulator.AdvanceTo(2.0)

print(context)