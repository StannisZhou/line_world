import numpy as np
import torch


def expand_templates(templates, stride, n_channels, grid_size, n_channels_next_layer, grid_size_next_layer):
    """expand_templates
    Expand the templates array, which is of shape (n_templates, n_channels_next_layer, kernel_size, kernel_size),
    to an expanded_templates sparse matrix, which is of shape
    (n_channels, grid_size, grid_size, n_templates + 1, n_channels_next_layer, grid_size_next_layer, grid_size_next_layer)
    I.e. we are expanding the templates to be full size of next layer, and also add the template for the off state.
    Exists for easy calculation of the no_parents_prob array

    Parameters
    ----------

    templates : torch.Tensor
        templates is the array containing all the templates
    stride : int
        stride is the stride associated with this layer
    n_channels : int
        n_channels is the number of channels for this layer
    grid_size : int
        grid_size is the grid size for this layer
    n_channels_next_layer : int
        n_channels_next_layer is the number of channels for the next layer
    grid_size_next_layer : int
        grid_size_next_layer is the grid size for the next layer

    Returns
    -------

    """
    n_templates = templates.size(0)
    templates_coords = templates._indices().cpu().numpy()
    templates_data = templates._values().cpu().numpy()
    n_nonzeros = templates_data.size
    coords_x = np.repeat(np.arange(grid_size, dtype=int), grid_size)
    coords_y = np.tile(np.arange(grid_size, dtype=int), grid_size)
    coords_x = np.repeat(coords_x, n_nonzeros)
    coords_y = np.repeat(coords_y, n_nonzeros)
    templates_coords = np.tile(templates_coords, grid_size**2)
    templates_data = np.tile(templates_data, grid_size**2)
    templates_coords[0] = templates_coords[0] + 1
    templates_coords[2] = templates_coords[2] + stride * coords_x
    templates_coords[3] = templates_coords[3] + stride * coords_y
    n_nonzeros = templates_data.size
    coords_channel = np.repeat(np.arange(n_channels, dtype=int), n_nonzeros)
    coords_x = np.tile(coords_x, n_channels)
    coords_y = np.tile(coords_y, n_channels)
    templates_coords = np.tile(templates_coords, n_channels)
    templates_data = np.tile(templates_data, n_channels)
    n_nonzeros = templates_data.size
    coords = np.zeros((7, n_nonzeros), dtype=int)
    coords[0] = coords_channel
    coords[1] = coords_x
    coords[2] = coords_y
    coords[3:] = templates_coords
    expanded_templates = torch.sparse_coo_tensor(coords, templates_data, (
        n_channels, grid_size, grid_size, n_templates + 1,
        n_channels_next_layer, grid_size_next_layer, grid_size_next_layer
    ))
    return expanded_templates
