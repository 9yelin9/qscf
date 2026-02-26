#!/usr/bin/env python3

import pennylane as qml
from pennylane import numpy as np
from pyscf.pbc import gto, scf

"""
mol = gto.M(atom = 'Si 0 0 0',
            a = np.eye(3) * 3.9,
            basis = 'gth-dzv',
            pseudo = 'gth-lda')
mf = scf.RKS(mol, xc='lda') # Restricted KS-DFT (RKS): the spin-orbitals are either spin-up or spin-down
mf.kernel()
"""

num_wires = 4
num_layers = 5

device = qml.device("default.qubit", wires=num_wires, shots=1000)

ansatz = qml.StronglyEntanglingLayers

all_pauliz_tensor_prod = qml.prod(*[qml.PauliZ(i) for i in range(num_wires)])


def circuit(param):
    ansatz(param, wires=list(range(num_wires)))
    return qml.expval(all_pauliz_tensor_prod)

qnode = qml.QNode(circuit, device)

def cost(param):
    return sum([qnode(param) for _ in range(3)])

#cost_function = qml.set_shots(qml.QNode(circuit, device), shots = 1000)
cost_function = cost

np.random.seed(50)

param_shape = ansatz.shape(num_layers, num_wires)
init_param = np.random.normal(scale=0.1, size=param_shape, requires_grad=True)

def run_optimizer(opt, cost_function, init_param, num_steps, interval, execs_per_step):
    # Copy the initial parameters to make sure they are never overwritten
    param = init_param.copy()

    # Obtain the device used in the cost function
    #dev = cost_function.device

    # Initialize the memory for cost values during the optimization
    cost_history = []
    # Monitor the initial cost value
    cost_history.append(cost_function(param))
    exec_history = [0]

    print(
        f"\nRunning the {opt.__class__.__name__} optimizer for {num_steps} iterations."
    )
    for step in range(num_steps):
        # Print out the status of the optimization
        if step % interval == 0:
            print(
                f"Step {step:3d}: Circuit executions: {exec_history[step]:4d}, "
                f"Cost = {cost_history[step]}"
            )

        # Perform an update step
        param = opt.step(cost_function, param)

        # Monitor the cost value
        cost_history.append(cost_function(param))
        exec_history.append((step + 1) * execs_per_step)

    print(
        f"Step {num_steps:3d}: Circuit executions: {exec_history[-1]:4d}, "
        f"Cost = {cost_history[-1]}"
    )
    return cost_history, exec_history

num_steps_spsa = 200
opt = qml.SPSAOptimizer(maxiter=num_steps_spsa, c=0.15, a=0.2)
# We spend 2 circuit evaluations per step:
execs_per_step = 2
cost_history_spsa, exec_history_spsa = run_optimizer(
    opt, cost_function, init_param, num_steps_spsa, 20, execs_per_step
)

num_steps_grad = 15
opt = qml.GradientDescentOptimizer(stepsize=0.3)
# We spend 2 circuit evaluations per parameter per step:
execs_per_step = 2 * np.prod(param_shape)
cost_history_grad, exec_history_grad = run_optimizer(
    opt, cost_function, init_param, num_steps_grad, 3, execs_per_step
)
