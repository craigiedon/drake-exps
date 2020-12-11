from pydrake.all import (DiagramBuilder, SignalLogger, Variable, PidController,
                         Simulator, SymbolicVectorSystem, ConstantVectorSource,
                         Multiplexer, GenerateHtml, SceneGraph, sin)



def pendulum_dynamics(x, u, p):
    q = x[0]
    qdot = x[1]
    tau = u[0]
    return [
        qdot,((-p["m"] * p["g"] * p["l"] * sin(q) + tau) / (p["m"] * p["l"] ** 2))
    ]

def pendulum_with_motor_dynamics(x, u, p):
    q = x[0]
    qdot = x[1]
    tau = u[0]

    arm_intertia = (-p["m"] * p["g"] * p["l"] * sin(q) + tau) / (p["m"] * p["l"] ** 2)

    # Note: should tau actually be tau_motor? Which one do we get as input?
    q_dd_motor = (p["g"] / p["N"] + tau) / (p["I"] + arm_intertia / p["N"] ** 2)
    q_dd = q_dd_motor / p["N"]

    return [
        qdot,
        q_dd
    ]



x = [Variable("theta"), Variable("thetadot")]
u = [Variable("tau")]

# Example parameters of pendulum dynamics
p = {"m": 1.0, "g": 9.81, "l": 0.5}

system = SymbolicVectorSystem(state=x, output=x, input=u, dynamics=pendulum_dynamics(x, u, p))

context = system.CreateDefaultContext()
print(context)