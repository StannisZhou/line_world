import numba
import numpy as np
from line_world.numba.layer import get_no_parents_prob_numba, draw_sample_from_layer_numba


def fast_sample_markov_backbone(layer_list, n_samples):
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

    samples_list = draw_multiple_samples_markov_backbone_numba(
        n_samples, n_channels_list, grid_size_list, n_templates_list, expanded_templates_list, self_rooting_prob_list, parent_prob_list
    )
    return samples_list


@numba.jit(nopython=True, nogil=True, cache=True)
def draw_multiple_samples_markov_backbone_numba(n_samples, n_channels_list, grid_size_list, n_templates_list,
                                                expanded_templates_list, self_rooting_prob_list, parent_prob_list):
    samples_list = []
    for ii in range(n_samples):
        if np.mod(ii + 1, 1000) == 0:
            print(ii + 1)

        samples_list.append(draw_sample_markov_backbone_numba(
            n_channels_list, grid_size_list, n_templates_list, expanded_templates_list, self_rooting_prob_list, parent_prob_list
        ))

    return samples_list


@numba.jit(nopython=True, nogil=True, cache=True)
def draw_sample_markov_backbone_numba(n_channels_list, grid_size_list, n_templates_list, expanded_templates_list,
                                      self_rooting_prob_list, parent_prob_list):
    layer_sample_list = []
    no_parents_prob = np.ones((n_channels_list[0], grid_size_list[0], grid_size_list[0]))
    for ii, (n_channels, grid_size, n_templates, expanded_templates, self_rooting_prob, parent_prob) in enumerate(zip(
        n_channels_list, grid_size_list, n_templates_list, expanded_templates_list, self_rooting_prob_list, parent_prob_list
    )):
        layer_sample = draw_sample_from_layer_numba(
            n_channels, grid_size, n_templates, no_parents_prob, self_rooting_prob, parent_prob
        )
        n_channels_next_layer = n_channels_list[ii + 1]
        grid_size_next_layer = grid_size_list[ii + 1]
        no_parents_prob = get_no_parents_prob_numba(
            layer_sample, expanded_templates, n_channels, grid_size, n_templates, n_channels_next_layer, grid_size_next_layer
        )
        layer_sample_list.append(layer_sample)

    layer_sample_list.append(
        draw_sample_from_layer_numba(
            n_channels_list[-1], grid_size_list[-1], n_templates_list[-1], no_parents_prob, self_rooting_prob_list[-1], parent_prob[-1]
        )
    )
    return layer_sample_list
