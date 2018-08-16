import numpy as np
from line_world.cycles_machine import CyclesMachine
from line_world.params import generate_cycles_machine_layer_params
from line_world.perturbation import draw_samples_markov_backbone
import torch
import torch.optim

# Set basic parameters
n_layers = 3
n_channels_list = np.ones(n_layers - 1, dtype=int)
d_image = 4
kernel_size_list = np.array([2, 3], dtype=int)
stride_list = np.ones(n_layers - 1, dtype=int)
self_rooting_prob_list = np.array([0.5, 0.001, 0.001])
thickness = 1
length = 3
n_rotations = 4
n_parts = 2
order = 0
cycles_perturbation_implementation = 'toy_perturbation'
cycles_perturbation_params = {
    'perturbed_distribution': torch.tensor([0.01, 0.01, 0.97, 0.01]),
    'sigma': 0.2
}
n_samples = int(2e3)

# Initialize the CyclesMachine
layer_params_list = generate_cycles_machine_layer_params(
    n_layers, n_channels_list, d_image, kernel_size_list, stride_list, self_rooting_prob_list,
    thickness, length, n_rotations, n_parts, order
)
cycles_machine = CyclesMachine({
    'layer_params_list': layer_params_list,
    'cycles_perturbation_implementation': cycles_perturbation_implementation,
    'cycles_perturbation_params': cycles_perturbation_params,
    'n_samples': n_samples
})

image = torch.zeros(cycles_machine.layer_list[2].state_shape)
for ii in range(4):
    for jj in range(4):
        if ii == jj:
            image[0, ii, jj, 1] = 1
        else:
            image[0, ii, jj, 0] = 1

optimal_top_layer = torch.zeros(cycles_machine.layer_list[0].state_shape)
optimal_top_layer[0, 0, 0, 3] = 1
optimal_middle_layer = torch.zeros(cycles_machine.layer_list[1].state_shape)
optimal_middle_layer[0, 0, 0, 8] = 1
optimal_middle_layer[0, 0, 1, 0] = 1
optimal_middle_layer[0, 1, 0, 0] = 1
optimal_middle_layer[0, 1, 1, 8] = 1
optimal_state_list = [
    optimal_top_layer,
    optimal_middle_layer,
    image
]
optimal_log_prob = cycles_machine.get_energy(optimal_state_list)
state_list = [
    torch.rand(cycles_machine.layer_list[0].state_shape).requires_grad_(),
    torch.rand(cycles_machine.layer_list[1].state_shape).requires_grad_(),
    image
]
optimizer = torch.optim.SGD(state_list, lr=0.1)
n_steps = 2000
for ii in range(n_steps):
    optimizer.zero_grad()
    log_prob = cycles_machine.evaluate_energy_gradients(state_list)
    print('Step #{}, log_prob: {}'.format(ii, log_prob))
    optimizer.step()

print('Final log_prob: {}'.format(cycles_machine.get_energy(state_list)))
print('Optimal log_prob: {}'.format(optimal_log_prob))
