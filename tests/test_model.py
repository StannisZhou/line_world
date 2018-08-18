import numpy as np
from line_world.cycles_machine import CyclesMachine
from line_world.params import generate_cycles_machine_layer_params
from line_world.sample.markov_backbone import draw_sample_markov_backbone
import pytest

@pytest.fixture
def toy_model():
    # Set basic parameters
    n_layers = 3
    n_channels_list = np.ones(n_layers - 1, dtype=int)
    d_image = 8
    kernel_size_list = np.array([3, 4], dtype=int)
    stride_list = 2 * np.ones(n_layers - 1, dtype=int)
    self_rooting_prob_list = np.array([1.0, 0, 0])
    thickness = 2
    length = 3
    n_rotations = 10
    n_parts = 3

    # Initialize the CyclesMachine
    layer_params_list = generate_cycles_machine_layer_params(
        n_layers, n_channels_list, d_image, kernel_size_list, stride_list, self_rooting_prob_list,
        thickness, length, n_rotations, n_parts
    )
    cycles_machine = CyclesMachine({'layer_params_list': layer_params_list})
    return cycles_machine


@pytest.fixture
def simple_model():
    # Set basic parameters
    n_layers = 3
    n_channels_list = 2 * np.ones(n_layers - 1, dtype=int)
    d_image = 16
    kernel_size_list = np.array([3, 4], dtype=int)
    stride_list = 2 * np.ones(n_layers - 1, dtype=int)
    self_rooting_prob_list = np.array([0.2, 0.01, 0.01])
    thickness = 1
    length = 3
    n_rotations = 10
    n_parts = 3

    # Initialize the CyclesMachine
    layer_params_list = generate_cycles_machine_layer_params(
        n_layers, n_channels_list, d_image, kernel_size_list, stride_list, self_rooting_prob_list,
        thickness, length, n_rotations, n_parts
    )
    cycles_machine = CyclesMachine({'layer_params_list': layer_params_list})
    return cycles_machine


def test_model(simple_model):
    layer_sample_list = draw_sample_markov_backbone(simple_model.layer_list)
    for layer_sample in layer_sample_list:
        layer_sample.requires_grad_()

    simple_model.evaluate_energy_gradients(layer_sample_list)
