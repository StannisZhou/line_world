import numpy as np
import torch


def get_coarse_stride_kernel_size(stride_list, kernel_size_list):
    """get_coarse_stride_kernel_size
    Use information from the list of layers to infer the stride and the kernel_size involved with
    a coarse layer

    Parameters
    ----------

    stride_list : np.ndarray
        The list of all the strides in the relevant layers
    kernel_size_list : np.ndarray
        The list of all the kernel sizes in the relevant layers

    Returns
    -------

    stride : int
        The stride associated with the coarse layer
    kernel_size : int
        The kernel size associated with the coarse layer

    """
    assert len(stride_list) == len(kernel_size_list)
    stride = np.prod(stride_list)
    kernel_size = 1
    for ii in range(len(stride_list)):
        kernel_size = (kernel_size - 1) * stride_list[ii] + kernel_size_list[ii]

    return stride, kernel_size


def get_interlayer_connections(indices, state_list, layer_list):
    n_layers = indices[1] - indices[0]
    layer_list = layer_list[indices[0]:indices[1] + 1]
    state_list = state_list[indices[0]:indices[1] + 1]
    on_bricks_prob_list = [
        layer_list[ii].get_on_bricks_prob(state_list[ii]) for ii in range(n_layers)
    ]
    parents_prob_list = [
        1 - layer_list[ii].get_no_parents_prob(state_list[ii], False) for ii in range(n_layers)
    ]
    connections = on_bricks_prob_list[0].reshape((-1, 1)) * parents_prob_list[0].reshape((
        layer_list[0].n_bricks, layer_list[1].n_bricks
    ))
    for ii in range(1, n_layers):
        connections = connections * on_bricks_prob_list[ii].reshape((1, -1))
        connections = torch.matmul(connections, parents_prob_list[ii].reshape(
            layer_list[ii].n_bricks, layer_list[ii + 1].n_bricks
        ))

    connections = connections * layer_list[-1].get_on_bricks_prob(state_list[-1]).reshape((1, -1))
    connections = connections.reshape(layer_list[0].shape + layer_list[-1].shape)
    return connections


def get_n_coarse_cycles(state_list, coarse_state_collections, layer_list, coarse_layer_collections):
    list_of_n_coarse_cycles_list = []
    for cc, coarse_state_for_layer in enumerate(coarse_state_collections):
        list_of_n_coarse_cycles_list.append(get_n_coarse_cycles_for_layer(
            coarse_state_for_layer, state_list, layer_list, coarse_layer_collections[cc]
        ))

    return list_of_n_coarse_cycles_list


def get_n_coarse_cycles_for_layer(coarse_state_for_layer, state_list, layer_list, list_of_coarse_layer):
    n_coarse_cycles_list = []
    for cc, coarse_state in enumerate(coarse_state_for_layer):
        coarse_layer = list_of_coarse_layer[cc]
        indices = (coarse_layer.params['index_to_duplicate'], coarse_layer.params['index_to_point_to'])
        fine_connections = get_interlayer_connections(indices, state_list, layer_list)
        coarse_connections = coarse_layer.get_connections(coarse_state, layer_list)
        n_coarse_cycles_list.append(
            torch.sum(fine_connections * coarse_connections, dim=[3, 4, 5])
        )

    return n_coarse_cycles_list
