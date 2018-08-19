import numpy as np
import torch
import timeit
from test_model import simple_model
from line_world.sample.markov_backbone import draw_sample_markov_backbone
from line_world.sample.fast_markov_backbone import get_no_parents_prob_numba, fast_sample_markov_backbone
from line_world.perturbation import ToyPerturbation
from line_world.cycles_machine import CyclesMachine
from line_world.params import generate_cycles_machine_layer_params


def test_no_parents_prob(simple_model):
    layer_sample_list = draw_sample_markov_backbone(simple_model.layer_list)
    for ll, layer in enumerate(simple_model.layer_list[:-1]):
        no_parents_prob = layer.get_no_parents_prob(layer_sample_list[ll]).numpy()
        n_channels, grid_size, _ = tuple(layer.shape)
        n_channels_next_layer, grid_size_next_layer, _ = tuple(simple_model.layer_list[ll + 1].shape)
        expected_no_parents_prob = get_no_parents_prob_numba(
            layer_sample_list[ll].numpy(), layer.expanded_templates.to_dense().float().numpy(),
            n_channels, grid_size, layer.n_templates, n_channels_next_layer, grid_size_next_layer
        )
        assert np.all(no_parents_prob == expected_no_parents_prob)


def test_sampling_markov_backbone(simple_model):
    fast_sample_markov_backbone(simple_model.layer_list, 100)


def test_null_distribution():
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
        'cycles_perturbation_implementation': 'markov_backbone',
        'cycles_perturbation_params': {},
        'n_samples': int(1e2)
    })

    params = {
        'perturbed_distribution': torch.tensor([0.01, 0.01, 0.97, 0.01]),
        'sigma': 0.2,
    }
    n_samples = int(5e3)
    params['fast_sample'] = False
    slow_null_distribution = ToyPerturbation(cycles_machine.layer_list, n_samples, params).null_distribution
    params['fast_sample'] = True
    fast_null_distribution = ToyPerturbation(cycles_machine.layer_list, n_samples, params).null_distribution
