import numba
import numpy as np
from line_world.numba.layer import get_on_bricks_prob_numba, get_no_parents_prob_separate_numba


@numba.jit(nopython=True, nogil=True, cache=True)
def toy_perturbation_numba(state_list, n_channels_list, grid_size_list, n_templates_list,
                           expanded_templates_list, null_distribution, perturbed_distribution, sigma):
    n_cycles = get_n_cycles_three_layers_numba(
        state_list, n_channels_list, grid_size_list, n_templates_list, expanded_templates_list
    )[0, 0, 0]
    null_prob = np.sum(
        null_distribution * np.exp(
            -(np.arange(null_distribution.size) - n_cycles)**2 / (2 * sigma**2)
        )
    )
    perturbed_prob = np.sum(
        perturbed_distribution * np.exp(
            -(np.arange(perturbed_distribution.size) - n_cycles)**2 / (2 * sigma**2)
        )
    )
    perturbation = perturbed_prob / null_prob
    return perturbation


@numba.jit(nopython=True, nogil=True, cache=True)
def get_n_cycles_three_layers_numba(state_list, n_channels_list, grid_size_list, n_templates_list,
                                    expanded_templates_list):
    assert len(state_list) == 3
    assert len(n_channels_list) == 3
    assert len(grid_size_list) == 3
    assert len(n_templates_list) == 3
    assert len(expanded_templates_list) == 2
    n_bricks_list = [n_channels_list[ii] * grid_size_list[ii]**2 for ii in range(3)]
    on_bricks_prob_list = [get_on_bricks_prob_numba(state_list[ii]) for ii in range(3)]
    parents_prob_list = [
        1 - get_no_parents_prob_separate_numba(
            state_list[ii], expanded_templates_list[ii], n_channels_list[ii], grid_size_list[ii],
            n_templates_list[ii], n_channels_list[ii + 1], grid_size_list[ii + 1]
        ) for ii in range(2)
    ]
    temp_1 = on_bricks_prob_list[0].reshape((-1, 1)) * parents_prob_list[0].reshape((
        n_bricks_list[0], n_bricks_list[1]
    ))
    temp_2 = temp_1 * on_bricks_prob_list[1].reshape((1, -1))
    temp_3 = np.dot(temp_2, parents_prob_list[1].reshape(
        n_bricks_list[1], n_bricks_list[2]
    ))
    temp_4 = temp_3 * on_bricks_prob_list[2].reshape((1, -1))
    temp_5 = temp_4 * (temp_4 - 1) / 2
    temp_6 = temp_5 * (temp_5 > 0)
    n_cycles = np.sum(temp_6, axis=1).reshape((n_channels_list[0], grid_size_list[0], grid_size_list[0]))
    return n_cycles
