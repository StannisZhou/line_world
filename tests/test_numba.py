import numpy as np
import torch
import timeit
import pytest
from test_model import simple_model
from line_world.sample.markov_backbone import draw_sample_markov_backbone
from line_world.sample.fast_markov_backbone import fast_sample_markov_backbone
from line_world.numba.layer import get_no_parents_prob_numba, get_no_parents_prob_separate_numba
from line_world.numba.perturbation import get_n_cycles_three_layers_numba, toy_perturbation_numba
from line_world.sample.fast_rejection_sampling import fast_sample_rejection_sampling
from line_world.perturb.perturbation import get_n_cycles
from line_world.toy.perturbation import ToyPerturbation
from line_world.cycles_machine import CyclesMachine
from line_world.params import generate_cycles_machine_layer_params

N_SAMPLES = int(5e3)


def test_no_parents_prob(simple_model):
    layer_sample_list = draw_sample_markov_backbone(simple_model.layer_list)
    for ll, layer in enumerate(simple_model.layer_list[:-1]):
        no_parents_prob = layer.get_no_parents_prob(layer_sample_list[ll]).numpy()
        no_parents_prob_separate = layer.get_no_parents_prob(layer_sample_list[ll], False).numpy()
        n_channels, grid_size, _ = tuple(layer.shape)
        n_channels_next_layer, grid_size_next_layer, _ = tuple(simple_model.layer_list[ll + 1].shape)
        expected_no_parents_prob = get_no_parents_prob_numba(
            layer_sample_list[ll].numpy(), layer.expanded_templates.to_dense().float().numpy(),
            n_channels, grid_size, layer.n_templates, n_channels_next_layer, grid_size_next_layer
        )
        expected_no_parents_prob_separate = get_no_parents_prob_separate_numba(
            layer_sample_list[ll].numpy(), layer.expanded_templates.to_dense().float().numpy(),
            n_channels, grid_size, layer.n_templates, n_channels_next_layer, grid_size_next_layer
        )
        assert np.all(no_parents_prob == expected_no_parents_prob)
        assert np.all(no_parents_prob_separate == expected_no_parents_prob_separate)


def test_sampling_markov_backbone(simple_model):
    fast_sample_markov_backbone(simple_model.layer_list, 100)


@pytest.fixture
def toy_model_two_parts():
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

    # Initialize the CyclesMachine
    layer_params_list = generate_cycles_machine_layer_params(
        n_layers, n_channels_list, d_image, kernel_size_list, stride_list, self_rooting_prob_list,
        thickness, length, n_rotations, n_parts, order
    )
    cycles_machine = CyclesMachine({
        'layer_params_list': layer_params_list,
        'cycles_perturbation_implementation': 'toy_perturbation',
        'cycles_perturbation_params': {
            'perturbed_distribution': torch.tensor([0.01, 0.01, 0.97, 0.01]),
            'sigma': 0.2,
            'fast_sample': True
        },
        'n_samples': N_SAMPLES
    })
    return cycles_machine

@pytest.fixture
def toy_model_three_parts():
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
    perturbed_distribution = 0.01 * torch.ones(6, dtype=torch.float)
    perturbed_distribution[5] = 0.95
    cycles_perturbation_params = {
        'perturbed_distribution': perturbed_distribution,
        'sigma': 1.0,
        'fast_sample': True
    }

    # Initialize the CyclesMachine
    layer_params_list = generate_cycles_machine_layer_params(
        n_layers, n_channels_list, d_image, kernel_size_list, stride_list, self_rooting_prob_list,
        thickness, length, n_rotations, n_parts, order
    )
    cycles_machine = CyclesMachine({
        'layer_params_list': layer_params_list,
        'cycles_perturbation_implementation': cycles_perturbation_implementation,
        'cycles_perturbation_params': cycles_perturbation_params,
        'n_samples': N_SAMPLES,
    })
    return cycles_machine


def test_null_distribution_two_parts(toy_model_two_parts):
    params = {
        'perturbed_distribution': torch.tensor([0.01, 0.01, 0.97, 0.01]),
        'sigma': 0.2,
        'fast_sample': False
    }
    n_samples = N_SAMPLES
    slow_null_distribution = ToyPerturbation(toy_model_two_parts.layer_list, n_samples, params).null_distribution
    fast_null_distribution = toy_model_two_parts.cycles_perturbation.null_distribution


def test_null_distribution_three_parts(toy_model_three_parts):
    perturbed_distribution = 0.01 * torch.ones(6, dtype=torch.float)
    perturbed_distribution[5] = 0.95
    params = {
        'perturbed_distribution': perturbed_distribution,
        'sigma': 1.0,
        'fast_sample': False
    }
    n_samples = N_SAMPLES
    slow_null_distribution = ToyPerturbation(toy_model_three_parts.layer_list, n_samples, params).null_distribution
    fast_null_distribution = toy_model_three_parts.cycles_perturbation.null_distribution


def test_rejection_sampling(toy_model_two_parts):
    samples_list = fast_sample_rejection_sampling(
        toy_model_two_parts.layer_list, toy_model_two_parts.params['cycles_perturbation_implementation'],
        toy_model_two_parts.cycles_perturbation, 5
    )


def test_n_cycles_three_layers(toy_model_three_parts):
    samples_list = fast_sample_markov_backbone(toy_model_three_parts.layer_list, 1000)
    n_channels_list = []
    grid_size_list = []
    n_templates_list = []
    expanded_templates_list = []
    for layer in toy_model_three_parts.layer_list:
        n_channels, grid_size, _ = layer.shape
        n_channels_list.append(n_channels)
        grid_size_list.append(grid_size)
        n_templates_list.append(layer.n_templates)

    for layer in toy_model_three_parts.layer_list[:-1]:
        expanded_templates_list.append(layer.expanded_templates.to_dense().float().numpy())

    for layer_sample_list in samples_list:
        n_cycles_list = get_n_cycles(layer_sample_list, toy_model_three_parts.layer_list)
        n_cycles = n_cycles_list[0][0, 0, 0].item()

        n_cycles_list_numba = get_n_cycles_three_layers_numba(
            layer_sample_list, n_channels_list, grid_size_list, n_templates_list, expanded_templates_list
        )
        expected_n_cycles = n_cycles_list_numba[0, 0, 0]
        assert n_cycles == expected_n_cycles


def test_toy_perturbation(toy_model_three_parts):
    samples_list = fast_sample_markov_backbone(toy_model_three_parts.layer_list, 1000)
    n_channels_list = []
    grid_size_list = []
    n_templates_list = []
    expanded_templates_list = []
    for layer in toy_model_three_parts.layer_list:
        n_channels, grid_size, _ = layer.shape
        n_channels_list.append(n_channels)
        grid_size_list.append(grid_size)
        n_templates_list.append(layer.n_templates)

    for layer in toy_model_three_parts.layer_list[:-1]:
        expanded_templates_list.append(layer.expanded_templates.to_dense().float().numpy())

    for layer_sample_list in samples_list:
        layer_sample_list_torch = [
            torch.tensor(state, dtype=torch.float) for state in layer_sample_list
        ]
        perturbation = torch.exp(
            toy_model_three_parts.cycles_perturbation.get_log_prob_cycles_perturbation(layer_sample_list_torch)
        ).item()
        expected_perturbation = toy_perturbation_numba(
            layer_sample_list, n_channels_list, grid_size_list, n_templates_list, expanded_templates_list,
            toy_model_three_parts.cycles_perturbation.null_distribution.numpy(),
            toy_model_three_parts.cycles_perturbation.params['perturbed_distribution'].numpy(),
            toy_model_three_parts.cycles_perturbation.params['sigma']
        )
        assert np.isclose(perturbation, expected_perturbation)
