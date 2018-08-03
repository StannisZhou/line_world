import numpy as np
from line_world.cycles_machine import CyclesMachine
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
