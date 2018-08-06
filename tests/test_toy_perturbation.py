import numpy as np
from line_world.cycles_machine import CyclesMachine, log_prob_markov_backbone
from line_world.params import generate_cycles_machine_layer_params
from line_world.perturbation import draw_samples_markov_backbone
import pytest
import torch

@pytest.fixture
def toy_model():
    # Set basic parameters
    n_layers = 3
    n_channels_list = np.ones(n_layers - 1, dtype=int)
    d_image = 4
    kernel_size_list = np.array([2, 3], dtype=int)
    stride_list = np.ones(n_layers - 1, dtype=int)
    self_rooting_prob_list = np.array([0.5, 0.1, 0.01])
    thickness = 1
    length = 3
    n_rotations = 4
    n_parts = 2
    order = 0
    cycles_perturbation_implementation = 'toy_perturbation'
    cycles_perturbation_params = {
        'perturbed_distribution': torch.tensor([0.01, 0.01, 0.97, 0.01]),
        'sigma': 0.5
    }
    n_samples = int(1e3)

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
    return cycles_machine


def test_perturbation(toy_model):
    layer_sample_list = draw_samples_markov_backbone(toy_model.layer_list)
    for layer_sample in layer_sample_list:
        layer_sample.requires_grad_()

    toy_model.evaluate_energy_gradients(layer_sample_list)


def test_log_prob_markov_backbone(toy_model):
    image = torch.zeros(toy_model.layer_list[2].state_shape)
    for ii in range(4):
        for jj in range(4):
            if ii == jj:
                image[0, ii, jj, 1] = 1
            else:
                image[0, ii, jj, 0] = 1

    optimal_top_layer = torch.zeros(toy_model.layer_list[0].state_shape)
    optimal_top_layer[0, 0, 0, 3] = 1
    optimal_middle_layer = torch.zeros(toy_model.layer_list[1].state_shape)
    optimal_middle_layer[0, 0, 0, 8] = 1
    optimal_middle_layer[0, 0, 1, 0] = 1
    optimal_middle_layer[0, 1, 0, 0] = 1
    optimal_middle_layer[0, 1, 1, 8] = 1
    optimal_state_list = [
        optimal_top_layer,
        optimal_middle_layer,
        image
    ]
    log_prob = log_prob_markov_backbone(optimal_state_list, toy_model.layer_list).item()
    expected_prob = 0.5 * (1 / 6) * 0.9**2 * (1 / 8)**2 * 0.99**12
    expected_log_prob = np.log(expected_prob)
    assert np.isclose(log_prob, expected_log_prob)
