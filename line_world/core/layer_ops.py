import torch
import numpy as np
from line_world.utils import ZERO_PROB_SUBSTITUTE, NO_PARENTS_PROB_MARGIN


def fast_sample_from_categorical_distribution(prob):
    """fast_categorical_distribution

    Parameters
    ----------

    prob : torch.Tensor
        prob is an multidimensional array where the last dimension represents a probability distribution

    Returns
    -------

    sample : torch.Tensor
        sample is an binary array of the same shape as prob. In addition, the last dimension contains only
        one non-zero entry, and represents one sample from the categorical distribution.

    """
    prob = torch.cumsum(prob, dim=-1)
    r = torch.rand(prob.size()[:-1])
    indices = (prob <= torch.unsqueeze(r, -1)).sum(dim=-1).view(-1)
    sample = torch.zeros((torch.prod(torch.tensor(prob.size()[:-1])), prob.size(-1)))
    sample[torch.arange(torch.prod(torch.tensor(prob.size()[:-1]))), indices] = 1
    sample = sample.reshape(prob.size())
    return sample


def calc_log_prob(state, prob, penalty=ZERO_PROB_SUBSTITUTE):
    prob[prob == 0] += penalty
    prob = prob / torch.sum(prob, dim=-1, keepdim=True)
    return torch.sum(state * torch.log(prob))


def normalize_state(state):
    """_normalize_state
    Normalize the state if it's unnormalized, and check to make sure the state is valid

    Parameters
    ----------

    state : torch.Tensor
        state is a tensor of shape state_shape. Can be unnormalized.

    Returns
    -------

    """
    if not np.allclose(torch.sum(state, dim=3).detach().numpy(), 1):
        state = torch.nn.functional.softmax(state, dim=3)

    assert np.allclose(torch.sum(state, dim=3).detach().numpy(), 1)
    return state


def get_no_parents_prob(state, expanded_templates, n_channels, grid_size, n_templates,
                        n_channels_next_layer, grid_size_next_layer, aggregate=True):
    """get_no_parents_prob
    Get the no_parents_prob array of the next layer

    Parameters
    ----------

    state : torch.Tensor
        state is a tensor of shape self.state_shape, which contains the state for which we
        want to get the no_parents_prob array for the next layer
    expanded_templates : torch.Tensor
        expanded_templates is the expanded version of the templates, which is of shape
        (n_channels, grid_size, grid_size, n_templates + 1, n_channels_next_layer, grid_size_next_layer, grid_size_next_layer)
    n_channels : int
        n_channels is the number of channels in the layer
    grid_size : int
        grid_size is the size of the grid in the layer
    n_templates : int
        The number of templates at each node
    n_channels_next_layer : int
        The number of channels in the next layer
    grid_size_next_layer : int
        The grid size in the next layer
    aggregate : bool
        aggregate indicates whether we want to aggregate the connectivity information across
        all the bricks in this layer. Defaults to be true.

    Returns
    -------

    no_parents_prob : torch.Tensor
        When aggregate is true, no_parents_prob is a tensor of shape
        (n_channels_next_layer, grid_size_next_layer, grid_size_next_layer), i.e. it has the
        same shape as the bricks in the next layer. Each component contains the information going
        to the next layer in terms of the probability that there's no parents pointing to the corresponding brick
        When aggregate is False, no_parents_prob is a tensor of shape
        (n_channels, grid_size, grid_size, n_channels_next_layer, grid_size_next_layer, grid_size_next_layer),
        and gives us the connectivity information between the two layers.

    """
    n_bricks = n_channels * grid_size**2
    assert expanded_templates.shape == torch.Size([
        n_channels, grid_size, grid_size, n_templates + 1,
        n_channels_next_layer, grid_size_next_layer, grid_size_next_layer
    ])
    state_shape = torch.Size([n_channels, grid_size, grid_size, n_templates + 1])
    assert state.shape == state_shape
    state = normalize_state(state)
    no_parents_prob = torch.sum(
        state.view(
            n_channels, grid_size, grid_size, n_templates + 1, 1, 1, 1
        ) * (1 - expanded_templates), dim=3
    )
    if aggregate:
        no_parents_prob = torch.prod(
            no_parents_prob.view(
                n_bricks, n_channels_next_layer, grid_size_next_layer, grid_size_next_layer
            ), dim=0
        )
    return no_parents_prob


def get_on_bricks_prob(state):
    """get_on_bricks_prob
    Get the probability that different bricks are on in this layer

    Parameters
    ----------

    state : torch.Tensor
        state is a tensor of shape state_shape, and contains the states for which we want to
        get the on_bricks_prob

    Returns
    -------

    on_bricks_prob : torch.Tensor
        on_bricks_prob is a tensor of shape self.shape, and contains the probability that the brciks
        are on in this layer

    """
    state = normalize_state(state)
    on_bricks_prob = torch.sum(state[..., 1:], dim=-1)
    return on_bricks_prob


def validate_no_parents_prob(no_parents_prob, sampling, shape):
    """_validate_no_parents_prob
    Validate the no_parents_prob array to make sure it's valid

    Parameters
    ----------

    no_parents_prob : torch.Tensor
        no_parents_prob is a tensor of shape self.shape, i.e. it has the same shape as the bricks in
        this layer. Each component contains the information coming from the previous layer in terms of
        the probability that there's no parents pointing to the corresponding brick.
    sampling : bool
        sampling is indicating whether we are doing sampling or not. If we are doing sampling, the
        no_parents_prob should be binary.

    Returns
    -------

    """
    assert no_parents_prob.size() == shape
    if sampling:
        assert torch.sum(no_parents_prob == 0) + torch.sum(no_parents_prob == 1) == no_parents_prob.numel()
    else:
        assert torch.sum((no_parents_prob >= 0) * (no_parents_prob <= 1 + NO_PARENTS_PROB_MARGIN)) == no_parents_prob.numel()


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


def get_log_prob(state, no_parents_prob, brick_self_rooting_prob, brick_parent_prob):
    state = normalize_state(state)
    log_prob = calc_log_prob(
        torch.unsqueeze(no_parents_prob, -1) * state, brick_self_rooting_prob
    ) + calc_log_prob(
        torch.unsqueeze((1 - no_parents_prob), -1) * state, brick_parent_prob
    )
    return log_prob
