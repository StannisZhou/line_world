import numpy as np
from line_world.cycles_machine import CyclesMachine
from line_world.params import generate_cycles_machine_layer_params
from line_world.perturbation import draw_samples_markov_backbone


# Set basic parameters
n_layers = 3
n_channels_list = 2 * np.ones(n_layers - 1, dtype=int)
d_image = 16
kernel_size_list = 4 * np.ones(n_layers - 1, dtype=int)
kernel_size_list[-1] = 6
stride_list = 2 * np.ones(n_layers - 1, dtype=int)
self_rooting_prob_list = np.array([0.2, 0.01, 0.01])
thickness = 1
length = 4
n_rotations = 10

# Initialize the CyclesMachine
layer_params_list = generate_cycles_machine_layer_params(
    n_layers, n_channels_list, d_image, kernel_size_list, stride_list, self_rooting_prob_list,
    thickness, length, n_rotations
)
cycles_machine = CyclesMachine({'layer_params_list': layer_params_list})
layer_sample_list = draw_samples_markov_backbone(cycles_machine.layer_list)
for layer_sample in layer_sample_list:
    layer_sample.requires_grad_()

cycles_machine.evaluate_energy_gradients(layer_sample_list)
