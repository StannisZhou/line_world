import numpy as np
from line_world.cycles_machine import CyclesMachine
from line_world.params import generate_cycles_machine_layer_params
from line_world.sample.markov_backbone import draw_sample_markov_backbone
import torch
import torch.optim

# Set basic parameters
n_layers = 3
n_channels_list = np.ones(n_layers - 1, dtype=int)
d_image = 6
kernel_size_list = np.array([4, 3], dtype=int)
stride_list = np.ones(n_layers - 1, dtype=int)
self_rooting_prob_list = np.array([0.9, 0.01, 0.01])
thickness = 1
length = 3
n_rotations = 4
n_parts = 3
order = 0
cycles_perturbation_implementation = 'toy_perturbation'
perturbed_distribution = 0.0001 * torch.ones(6, dtype=torch.float)
perturbed_distribution[5] = 0.9995
cycles_perturbation_params = {
    'perturbed_distribution': perturbed_distribution,
    'sigma': 1.0
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
    'n_samples': n_samples,
})

image = torch.zeros(cycles_machine.layer_list[2].state_shape)
for ii in range(6):
    for jj in range(6):
        if ii == jj:
            if ii == 0:
                image[0, ii, jj, 0] = 1
            else:
                image[0, ii, jj, 1] = 1
        else:
            image[0, ii, jj, 0] = 1

expanded_templates = cycles_machine.layer_list[0].expanded_templates.to_dense().numpy()
for ind in range(expanded_templates.shape[3]):
    if np.all(expanded_templates[0, 0, 0, ind] == np.array([[
        [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
    ]])):
        break

optimal_top_layer = torch.zeros(cycles_machine.layer_list[0].state_shape)
optimal_top_layer[0, 0, 0, ind] = 1
optimal_middle_layer = torch.zeros(cycles_machine.layer_list[1].state_shape)
for ii in range(4):
    for jj in range(4):
        if ii == jj:
            if ii == 0:
                optimal_middle_layer[0, ii, jj, 0] = 1
            else:
                optimal_middle_layer[0, ii, jj, 8] = 1
        else:
            optimal_middle_layer[0, ii, jj, 0] = 1

optimal_state_list = [
    optimal_top_layer,
    optimal_middle_layer,
    image
]
optimal_log_prob = cycles_machine.get_energy(optimal_state_list)
state_list = [
    torch.rand(cycles_machine.layer_list[0].state_shape).requires_grad_(),
    torch.rand(cycles_machine.layer_list[1].state_shape).requires_grad_(),
    #  (10 * optimal_top_layer).requires_grad_(),
    #  (10 * optimal_middle_layer).requires_grad_(),
    image
]
initial_weight_decay = 1.0
n_steps = 50
for ww in range(40):
    optimizer = torch.optim.SGD(state_list[:-1], lr=0.1, weight_decay=initial_weight_decay * 1.5**(-ww))
    for ii in range(n_steps):
        optimizer.zero_grad()
        log_prob = cycles_machine.evaluate_energy_gradients(state_list)
        print('Step #{}, log_prob: {}'.format(ww * n_steps + ii, log_prob))
        optimizer.step()

on_bricks_prob_list = [
    cycles_machine.layer_list[ii].get_on_bricks_prob(state) for ii, state in enumerate(state_list)
]
ind = torch.nonzero(state_list[0] > torch.max(state_list[0]) - 1)
expanded_templates = cycles_machine.layer_list[0].expanded_templates.to_dense()
print('Final log_prob: {}'.format(cycles_machine.get_energy(state_list)))
print('Optimal log_prob: {}'.format(optimal_log_prob))
