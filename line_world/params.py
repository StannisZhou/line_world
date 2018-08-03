'''Various helper functions for generating viable parameters for the model'''
import numpy as np
import itertools
import scipy.special
from line_world.data_generator import get_rotated_prototype
import torch
import sparse


def generate_latent_templates(n_channels, kernel_size, n_parts):
    """generate_templates
    Automatically generate the templates in COO sparse multidimensional array format for a particular number
    of channels and a particular kernel size.
    Each one of the templates would be binary, and have three nonzero entries. We are generating all possible
    templates given the n_channels and the kernel_size, conditioned on no two points being at the same location.
    The way we are going to generate the templates is, we are going to first sample three points, without
    replacement, on one channel, and then we are going to vary the channel number at each one of the three points,
    to get the various other templates.
    The total number of templates would be n_templates = choose(kernel_size**2, n_parts) * n_channels**n_parts

    Parameters
    ----------

    n_channels : int
        n_channels is the number of channels for the templates
    kernel_size : int
        kernel_size is the kernel size for the templates
    n_parts : int
        The number of parts we have in a particular object

    Returns
    -------

    templates : torch sparse COO tensor
        templates is a sparse COO multidimensional array of shape (n_templates, n_channels, kernel_size, kernel_size)

    """
    coords_x, coords_y = np.meshgrid(np.arange(kernel_size, dtype=int), np.arange(kernel_size, dtype=int), indexing='ij')
    coords_x = coords_x.flatten()
    coords_y = coords_y.flatten()
    location_indices = np.concatenate([
        np.array(x) for x in itertools.combinations(np.arange(kernel_size**2, dtype=int), n_parts)
    ])
    coords_x = coords_x[location_indices]
    coords_y = coords_y[location_indices]
    coords_x = np.tile(coords_x, n_channels**n_parts)
    coords_y = np.tile(coords_y, n_channels**n_parts)
    channel_indices = np.stack([
        np.array(x) for x in itertools.product(np.arange(n_channels, dtype=int), repeat=n_parts)
    ])
    coords_channel = np.repeat(channel_indices, int(location_indices.size / n_parts), axis=0).flatten()
    n_templates = int(scipy.special.comb(kernel_size**2, n_parts) * n_channels**n_parts)
    coords_templates = np.repeat(
        np.arange(n_templates, dtype=int), n_parts
    )
    coords = np.zeros((4, coords_templates.size), dtype=int)
    coords[0] = coords_templates
    coords[1] = coords_channel
    coords[2] = coords_x
    coords[3] = coords_y
    data = np.ones(coords_templates.size)
    latent_templates = torch.sparse_coo_tensor(coords, data, (n_templates, n_channels, kernel_size, kernel_size))
    return latent_templates


def generate_image_templates(kernel_size, thickness, length, n_rotations, order=1):
    """generate_image_templates
    Generate the image templates, which are linelets in the same form of those used in the data generator.

    Parameters
    ----------

    kernel_size : int
        kernel_size is the kernel size for the image templates
    thickness : int
        thickness is the thickness of our linelets
    length : int
        length is the length of our linelets
    n_rotations : int
        n_rotations is the number of rotations we are going to consider
    order : int
        The interpolation order used in rotate. See http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.rotate

    Returns
    -------

    image_templates : torch sparse COO Tensor
        A sparse COO multidimensional array of shape (n_templates, 1, kernel_size, kernel_size)
        For each rotation, we are first going to get a prototype. Then we are going to move this prototype
        around the allowed kernel_size to get all possible templates. The n_templates is a sum of the
        number of templates for each one of the rotations

    """
    rotation_angles = np.linspace(0, 180, n_rotations, endpoint=False)
    prototype_list = [
        get_rotated_prototype(thickness, length, angle, order) for angle in rotation_angles
    ]
    n_rows = np.zeros(n_rotations, dtype=int)
    n_cols = np.zeros(n_rotations, dtype=int)
    for ii, prototype in enumerate(prototype_list):
        n_rows[ii], n_cols[ii] = prototype.shape
        assert n_rows[ii] <= kernel_size and n_cols[ii] <= kernel_size

    n_row_locations = kernel_size - n_rows + 1
    n_col_locations = kernel_size - n_cols + 1
    n_templates = np.sum(n_row_locations * n_col_locations)
    image_templates = np.zeros((n_templates, 1, kernel_size, kernel_size), dtype=int)
    template_index = 0
    for ii in range(n_rotations):
        for jj in range(n_row_locations[ii]):
            for kk in range(n_col_locations[ii]):
                image_templates[template_index, 0, jj:(jj + n_rows[ii]), kk:(kk + n_cols[ii])] = prototype_list[ii]
                template_index += 1

    assert template_index == n_templates
    image_templates = sparse.COO.from_numpy(image_templates)
    image_templates = torch.sparse_coo_tensor(
        image_templates.coords, image_templates.data, image_templates.shape
    )
    return image_templates


def get_latent_layer_grid_size_list(n_layers, d_image, kernel_size_list, stride_list):
    """_get_layer_dimensions
    Get the grid sizes at each layer.

    Parameters
    ----------

    n_layers : int
        n_layers is the number of layers in the model.
    d_image : int
        d_image is the dimension of the image we are considering.
    kernel_size_list: np array
        kernel_size_list is an np array of shape (n_layers - 1,), and gives us the
        kernel_sizes at each layer
    stride_list : np array
        stride_list is an np array of shape (n_layers - 1), and gives us the strides
        at each layer.

    Returns
    -------

    grid_size_list : np array
        An np array of shape (n_layers - 1,), which gives us the dimension of the latent
        layers in the model

    """
    grid_size_list = np.zeros(n_layers, dtype=int)
    grid_size_list[-1] = d_image
    for layer in range(n_layers - 2, -1, -1):
        grid_size_list[layer] = (
            grid_size_list[layer + 1] - kernel_size_list[layer]
        ) / stride_list[layer] + 1

    assert np.all(grid_size_list > 0)
    return grid_size_list


def generate_cycles_machine_layer_params(n_layers, n_channels_list, d_image, kernel_size_list, stride_list,
                                    self_rooting_prob_list, thickness, length, n_rotations, n_parts, order=1):
    """generate_cycles_machine_layer_params
    Build the list of parameters for the different layers in the CyclesMachine.

    Parameters
    ----------

    n_layers : int
        n_layers is the number of layers in the model, including the image layer
    n_channels_list : np.ndarray
        n_channels_list is an array of shape (n_layers - 1,), and contains the number of channels for all
        the latent layers
    d_image : int
        d_image is the dimension, or the grid_size, of the image
    kernel_size_list : np.ndarray
        kernel_size_list is an array of shape (n_layers - 1,), and contains the kernel sizes for all
        the latent layers
    stride_list : np.ndarray
        stride_list is an array of shape (n_layers - 1,), and contains the strides for all
        the latent layers
    self_rooting_prob_list : np.ndarray
        self_rooting_prob_list is an array of shape (n_layers,), and contains the self rooting probability
        for all the layers, including the image layer
    thickness : int
        thickness is the thickness of the linelet prototypes for the image templates
    length : int
        length is the length of the linelet prototypes for the image templates
    n_rotations : int
        n_rotations is the number of rotations we are going to apply to the linelet prototypes for
        the image templates
    n_parts : int
        The number of parts we have in a particular object
    order : int
        The interpolation order used in rotate. See http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.rotate

    Returns
    -------

    layer_params_list : list
        A list of params for the different layers in the CyclesMachine, which we can feed into the CyclesMachine.

    """
    grid_size_list = get_latent_layer_grid_size_list(n_layers, d_image, kernel_size_list, stride_list)
    templates_list = [
        generate_latent_templates(n_channels_list[ii], kernel_size_list[ii], n_parts) for ii in range(n_layers - 2)
    ]
    templates_list.append(generate_image_templates(kernel_size_list[-1], thickness, length, n_rotations, order))
    layer_params_list = [
        {
            'n_channels': int(n_channels_list[ii]),
            'grid_size': int(grid_size_list[ii]),
            'stride': int(stride_list[ii]),
            'templates': templates_list[ii],
            'self_rooting_prob': float(self_rooting_prob_list[ii])
        } for ii in range(n_layers - 1)
    ]
    layer_params_list.append(
        {
            'n_channels': 1,
            'grid_size': d_image,
            'self_rooting_prob': float(self_rooting_prob_list[-1])
        }
    )
    return layer_params_list
