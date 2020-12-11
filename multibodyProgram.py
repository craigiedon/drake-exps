import numpy as np

from pydrake.common import FindResourceOrThrow
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.systems.analysis import Simulator
from pydrake.all import MultibodyPlant

from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve

plant_f = MultibodyPlant(0.0)
iiwa_file = FindResourceOrThrow(
    "drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf"
)
iiwa = Parser(plant_f).AddModelFromFile(iiwa_file)

# Define some short aliases from frames
W = plant_f.world_frame()
L0 = plant_f.GetFrameByName("iiwa_link_0", iiwa)
L7 = plant_f.GetFrameByName("iiwa_link_7", iiwa)

plant_f.WeldFrames(W, L0)
plant_f.Finalize()

# Allocate float context to be used by evluators.
context_f = plant_f.CreateDefaultContext()
# Create AutoDiffXd plant and corresponding context
plant_ad = plant_f.ToAutoDiffXd()
context_ad = plant_ad.CreateDefaultContext()

def resolve_frame(plant, F):
    """Gets a frame from a plant whose scalar type may be different."""
    return plant.GetFrameByName(F.name(), F.model_instance())

# Define target position.
p_WT = [0.1, 0.1, 0.6]

def link_7_distance_to_target(q):
    """Evalutes squared distance between L7 origin and target T."""
    # Choose plant and context based on dtype
    if q.dtype == float:
        plant = plant_f
        context = context_f
    else:
        # Assume AutoDiff
        plant = plant_ad
        context = context_ad
    # Do forward kinematics
    plant.SetPositions(context, iiwa, q)
    X_WL7 = plant.CalcRelativeTransform(context,
                                        resolve_frame(plant, W),
                                        resolve_frame(plant, L7))
    p_TL7 = X_WL7.translation() - p_WT


# WARNING: If you return a scalar for a constraint, or a vector for
# a cost, you may get the following cryptic error:
# "Unable to cast Python instance to c++ type"
link_7_distance_to_target_vector = lambda q: [link_7_distance_to_target(q)]

# New program: Using a custom evaluator
prog = MathematicalProgram()
q = prog.NewContinuousVariables(plant_f.num_positions())

# Define nominal configuration
q0 = np.zeros(plant_f.num_positions())

# Add basic cost. (This will be parsed into a quadratic cost)
prog.AddCost((q - q0).dot(q - q0))

# Add constraint based on custom evaluator.
prog.AddConstraint(link_7_distance_to_target_vector, lb=[0.1], ub=[0.2], vars=q)

result = Solve(prog, initial_guess=q0)

print(f"Success? {result.is_success()}")
print(result.get_solution_result())
q_sol = result.GetSolution(q)
print(q_sol)

prog = MathematicalProgram()

q = prog.NewContinuousVariables(plant_f.num_positions())
# Define nominal configuration.
q0 = np.zeros(plant_f.num_positions())

# Add custom cost.
prog.AddCost(link_7_distance_to_target, vars=q)

result = Solve(prog, initial_guess=q0)

print(f"Success? {result.is_success()}")
print(result.get_solution_result())
q_sol = result.GetSolution(q)
print(q_sol)

print(f"Initial distance: {link_7_distance_to_target(q0):.3f}")
print(f"Solution distance: {link_7_distance_to_target(q_sol):.3f}")