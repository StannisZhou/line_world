import numba
import numpy as np


@numba.jit(nopython=True, nogil=True, cache=True)
def get_no_parents_prob_separate_numba(state, expanded_templates, n_channels, grid_size, n_templates, n_channels_next_layer, grid_size_next_layer):
    assert np.all(np.sum(state, axis=3) == 1)
    no_parents_prob = np.sum(
        state.reshape((
            n_channels, grid_size, grid_size, n_templates + 1, 1, 1, 1
        )) * (1 - expanded_templates), axis=3
    )
    return no_parents_prob


@numba.jit(nopython=True, nogil=True, cache=True)
def get_no_parents_prob_numba(state, expanded_templates, n_channels, grid_size, n_templates, n_channels_next_layer, grid_size_next_layer):
    n_bricks = n_channels * grid_size**2
    temp = get_no_parents_prob_separate_numba(
        state, expanded_templates, n_channels, grid_size, n_templates, n_channels_next_layer, grid_size_next_layer
    ).reshape((n_bricks, n_channels_next_layer, grid_size_next_layer, grid_size_next_layer))
    no_parents_prob = np.zeros((n_channels_next_layer, grid_size_next_layer, grid_size_next_layer))
    for ii in range(n_channels_next_layer):
        for jj in range(grid_size_next_layer):
            for kk in range(grid_size_next_layer):
                no_parents_prob[ii, jj, kk] = np.prod(temp[:, ii, jj, kk])
    return no_parents_prob


@numba.jit(nopython=True, nogil=True, cache=True)
def draw_sample_from_layer_numba(n_channels, grid_size, n_templates, no_parents_prob, self_rooting_prob, parent_prob):
    prob = np.zeros((n_channels, grid_size, grid_size, n_templates + 1))
    for ii in range(n_channels):
        for jj in range(grid_size):
            for kk in range(grid_size):
                if no_parents_prob[ii, jj, kk] == 1:
                    prob[ii, jj, kk] = self_rooting_prob
                elif no_parents_prob[ii, jj, kk] == 0:
                    prob[ii, jj, kk] = parent_prob
                else:
                    assert False

    sample = sample_from_categorical_distribution_numba(prob)
    return sample


@numba.jit(nopython=True, nogil=True, cache=True)
def sample_from_categorical_distribution_numba(prob):
    """sample_from_categorical_distribution_numba

    Parameters
    ----------

    prob : np.ndarray
        prob is an multidimensional array where the last dimension represents a probability distribution

    Returns
    -------

    sample : np.ndarray
        sample is an binary array of the same shape as prob. In addition, the last dimension contains only
        one non-zero entry, and represents one sample from the categorical distribution.

    """
    flat_prob = prob.reshape((-1, prob.shape[-1]))
    sample = np.zeros_like(flat_prob)
    for ii in range(flat_prob.shape[0]):
        sample[ii] = np.random.multinomial(1, flat_prob[ii])

    return sample.reshape(prob.shape)


@numba.jit(nopython=True, nogil=True, cache=True)
def get_on_bricks_prob_numba(state):
    assert np.all(np.sum(state, axis=3) == 1)
    on_bricks_prob = np.sum(state[..., 1:], axis=-1)
    return on_bricks_prob
