import numba
import numpy as np
from line_world.sample.fast_markov_backbone import draw_sample_markov_backbone_numba
from line_world.numba.layer import get_no_parents_prob_numba
from line_world.numba.perturbation import toy_perturbation_numba


def fast_sample_rejection_sampling(layer_list, perturbation_implementation, cycles_perturbation, n_samples):
    assert perturbation_implementation == 'toy_perturbation'
    n_channels_list = []
    grid_size_list = []
    n_templates_list = []
    expanded_templates_list = []
    self_rooting_prob_list = []
    parent_prob_list = []
    for layer in layer_list:
        n_channels, grid_size, _ = layer.shape
        n_channels_list.append(n_channels)
        grid_size_list.append(grid_size)
        n_templates_list.append(layer.n_templates)
        self_rooting_prob_list.append(layer.params['brick_self_rooting_prob'].numpy())
        parent_prob_list.append(layer.params['brick_parent_prob'].numpy())

    for layer in layer_list[:-1]:
        expanded_templates_list.append(layer.expanded_templates.to_dense().float().numpy())

    sigma = cycles_perturbation.params['sigma']
    null_distribution = cycles_perturbation.null_distribution.numpy()
    perturbed_distribution = cycles_perturbation.params['perturbed_distribution'].numpy()
    perturbation_upperbound = cycles_perturbation.perturbation_upperbound.item()
    samples_list = draw_multiple_samples_rejection_sampling_numba(
        n_samples, n_channels_list, grid_size_list, n_templates_list, expanded_templates_list,
        self_rooting_prob_list, parent_prob_list, null_distribution,
        perturbed_distribution, perturbation_upperbound, sigma
    )
    return samples_list


@numba.jit(nopython=True, nogil=True, cache=True)
def draw_multiple_samples_rejection_sampling_numba(n_samples, n_channels_list, grid_size_list,
                                                   n_templates_list, expanded_templates_list,
                                                   self_rooting_prob_list, parent_prob_list,
                                                   null_distribution, perturbed_distribution,
                                                   perturbation_upperbound, sigma):
    samples_list = []
    for ii in range(n_samples):
        print(ii + 1)
        samples_list.append(draw_sample_rejection_sampling_numba(
            n_channels_list, grid_size_list, n_templates_list, expanded_templates_list,
            self_rooting_prob_list, parent_prob_list, null_distribution,
            perturbed_distribution, perturbation_upperbound, sigma
        ))

    return samples_list


@numba.jit(nopython=True, nogil=True, cache=True)
def draw_sample_rejection_sampling_numba(n_channels_list, grid_size_list, n_templates_list,
                                         expanded_templates_list, self_rooting_prob_list,
                                         parent_prob_list, null_distribution, perturbed_distribution,
                                         perturbation_upperbound, sigma):
        while True:
            layer_sample_list = draw_sample_markov_backbone_numba(
                n_channels_list, grid_size_list, n_templates_list, expanded_templates_list,
                self_rooting_prob_list, parent_prob_list

            )
            perturbation = toy_perturbation_numba(
                layer_sample_list, n_channels_list, grid_size_list, n_templates_list,
                expanded_templates_list, null_distribution, perturbed_distribution, sigma
            )
            acceptance_prob = perturbation / perturbation_upperbound
            if np.random.binomial(1, acceptance_prob):
                break

        return layer_sample_list
