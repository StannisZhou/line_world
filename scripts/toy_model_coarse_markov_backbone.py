import os
import torch
import torch.optim
import numpy as np
import sacred
#  from sacred.observers import FileStorageObserver
import line_world.coarse.coarse_ops as co
from line_world.core.cycles_machine import CyclesMachine
from line_world.params import generate_cycles_machine_layer_params, generate_image_templates
from line_world.diagnostic.states import visualize_states, visualize_gradients

log_folder = os.path.expanduser('~/logs/cycles/coarse_layer')
ex = sacred.Experiment('coarse_layer')
#  ex.observers.append(FileStorageObserver.create(log_folder))


@ex.config
def config():
    n_layers = 3
    d_image = 6
    kernel_size_list = [4, 3]
    thickness = 1
    length = 3
    n_rotations = 4
    n_parts = 3
    order = 0
    cycles_perturbation_implementation = 'markov_backbone'
    cycles_perturbation_params = {}


@ex.main
def run(n_layers, d_image, kernel_size_list, thickness, length, n_rotations, n_parts, order,
        cycles_perturbation_implementation, cycles_perturbation_params):
    # Process parameters
    n_channels_list = np.ones(n_layers - 1, dtype=int)
    kernel_size_list = np.array(kernel_size_list, dtype=int)
    stride_list = np.ones(n_layers - 1, dtype=int)

    # Get parameters for coarse layer
    coarse_stride, coarse_kernel_size = co.get_coarse_stride_kernel_size(stride_list[:2], kernel_size_list[:2])
    coarse_layer_params = {
        'index_to_duplicate': 0,
        'index_to_point_to': 2,
        'stride': coarse_stride,
        'templates': generate_image_templates(coarse_kernel_size, thickness, 5, n_rotations, order)
    }

    # Initialize the CyclesMachine
    n_samples = int(5e4)
    self_rooting_prob_list = np.array([0.1, 0.01, 0.01])
    layer_params_list = generate_cycles_machine_layer_params(
        n_layers, n_channels_list, d_image, kernel_size_list, stride_list, self_rooting_prob_list,
        thickness, length, n_rotations, n_parts, order
    )

    cycles_machine = CyclesMachine({
        'layer_params_list': layer_params_list,
        'cycles_perturbation_implementation': cycles_perturbation_implementation,
        'cycles_perturbation_params': cycles_perturbation_params,
        'coarse_layer_params_list': [coarse_layer_params],
        'n_samples': n_samples
    })

    # Set optimal states
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

    coarse_expanded_templates = cycles_machine.coarse_layer_collections[0][0].expanded_templates.to_dense().numpy()
    expected_image = cycles_machine.layer_list[-1].get_on_bricks_prob(image).numpy()
    for coarse_ind in range(coarse_expanded_templates.shape[3]):
        if np.all(coarse_expanded_templates[0, 0, 0, coarse_ind] == expected_image):
            break

    coarse_layer_shape = cycles_machine.layer_list[0].state_shape[:3] + torch.Size([coarse_expanded_templates.shape[3]])
    optimal_coarse_layer = torch.zeros(coarse_layer_shape)
    optimal_coarse_layer[0, 0, 0, coarse_ind] = 1
    optimal_coarse_state_collections = [[optimal_coarse_layer], [], []]
    optimal_log_prob = cycles_machine.get_energy(optimal_state_list, optimal_coarse_state_collections)

    # Initialize state
    state_list = [
        torch.rand(cycles_machine.layer_list[0].state_shape).requires_grad_(),
        torch.rand(cycles_machine.layer_list[1].state_shape).requires_grad_(),
        optimal_state_list[-1]
    ]

    coarse_state_collections = [
        [torch.rand(coarse_layer_shape).requires_grad_()], [], []
    ]

    cycles_machine_state = state_list + coarse_state_collections[0]
    n_steps = 500
    optimizer = torch.optim.SGD(
        cycles_machine_state, lr=0.05, momentum=0.9, nesterov=True
    )
    for ii in range(n_steps):
        optimizer.zero_grad()
        log_prob = cycles_machine.evaluate_energy_gradients(state_list, coarse_state_collections)
        print('Step #{}, log_prob: {}'.format(ii, log_prob))
        optimizer.step()

    visualize_states(
        state_list, coarse_state_collections, cycles_machine.layer_list, cycles_machine.coarse_layer_collections
    )
    visualize_gradients(
        state_list, coarse_state_collections, cycles_machine.layer_list, cycles_machine.coarse_layer_collections
    )
    print('Final log_prob: {}'.format(cycles_machine.get_energy(state_list, coarse_state_collections)))
    print('Optimal log_prob: {}'.format(cycles_machine.get_energy(optimal_state_list, optimal_coarse_state_collections)))


ex.run()
