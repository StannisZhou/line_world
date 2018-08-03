from line_world.utils import ParamsProc, Component, Optional
import numpy as np
import torch


class Layer(Component):
    @staticmethod
    def get_proc():
        proc = ParamsProc()
        proc.add(
            'n_channels', int,
            'The number of channels at this layer'
        )
        proc.add(
            'grid_size', int,
            'The size of the layer. We are assuming the grid in each channel to be square'
        )
        proc.add(
            'stride', int,
            'The size of the stride. Exists because each brick is only looking at a particular receptive field',
            Optional()
        )
        proc.add(
            'templates', torch.Tensor,
            '''The array containing all the templates. The shape of the array is (T, C, kernel_size, kernel_size),
            where T is the number of templates for the bricks in this layer, C is the number of channels in the
            layer below this layer, and kernel_size is the size of the receptive field. The array is binary.
            ''',
            Optional()
        )
        proc.add(
            'self_rooting_prob', float,
            'The self-rooting probability for this layer'
        )
        return proc

    @staticmethod
    def params_proc(params):
        if type(params['templates']) is not Optional:
            params['n_templates'] = params['templates'].size(0)
            params['kernel_size'] = params['templates'].size(2)
            params['n_channels_next_layer'] = params['templates'].size(1)
            params['grid_size_next_layer'] = (params['grid_size'] - 1) * params['stride'] + params['kernel_size']
        else:
            params['n_templates'] = 1

        params['brick_self_rooting_prob'] = torch.cat((
            torch.tensor([1 - params['self_rooting_prob']]),
            params['self_rooting_prob'] * torch.ones(params['n_templates']) / params['n_templates']
        ))
        params['brick_parent_prob'] = torch.cat((
            torch.zeros(1), torch.ones(params['n_templates']) / params['n_templates']
        ))

    @staticmethod
    def params_test(params):
        if type(params['templates']) is not Optional:
            assert params['templates'].size(2) == params['templates'].size(3)
            templates = params['templates'].to_dense().float()
            assert torch.sum(templates == 0) + torch.sum(templates == 1) == params['templates'].numel()

        assert params['self_rooting_prob'] >= 0 and params['self_rooting_prob'] <= 1

    def __init__(self, params):
        super().__init__(params)
        if type(self.params['templates']) is not Optional:
            self._expand_templates()

    @property
    def n_channels(self):
        """n_channels
        Number of channels in this layer
        """
        return self.params['n_channels']

    @property
    def grid_size(self):
        """grid_size
        Size of the grid for this layer
        """
        return self.params['grid_size']

    @property
    def n_bricks(self):
        """n_bricks
        Total number of bricks in this layer
        """
        return self.n_channels * self.grid_size**2

    @property
    def shape(self):
        """shape
        The shape of the channels of grids of bricks in this layer
        """
        return torch.Size([self.n_channels, self.grid_size, self.grid_size])

    @property
    def n_templates(self):
        return self.params['n_templates']

    @property
    def state_shape(self):
        """state_shape
        The shape of the state in this layer
        """
        return torch.Size([self.n_channels, self.grid_size, self.grid_size, self.n_templates + 1])

    def get_log_prob(self, state, no_parents_prob):
        """get_log_prob
        Get the log probability related to the Markov backbone for this layer

        Parameters
        ----------

        state : torch.Tensor
            state is a tensor of shape self.state_shape containing the states for which we want to get
            the log probability. It can be unnormalized. If unnormalized, it would be normalized by going
            through a softmax function.
        no_parents_prob : torch.Tensor
            no_parents_prob is a tensor of shape self.shape, i.e. it has the same shape as the bricks in
            this layer. Each component contains the information coming from the previous layer in terms of
            the probability that there's no parents pointing to the corresponding brick.

        Returns
        -------

        log_prob : float
            The log probability contributed by this layer. Note that this number is calculated using Autograd,
            and as a result contains the information necessary to calculate the gradients

        """
        self._validate_no_parents_prob(no_parents_prob, False)
        state = self._normalize_state(state)
        log_prob = calc_log_prob(
            state, torch.unsqueeze(no_parents_prob, -1) * self.params['brick_self_rooting_prob']
        ) + calc_log_prob(
            state, torch.unsqueeze((1 - no_parents_prob), -1) * self.params['brick_parent_prob']
        )
        return log_prob

    def get_no_parents_prob(self, state):
        """get_no_parents_prob
        Get the no_parents_prob array of the next layer

        Parameters
        ----------

        state : torch.Tensor
            state is a tensor of shape self.state_shape, which contains the state for which we
            want to get the no_parents_prob array for the next layer

        Returns
        -------

        no_parents_prob : torch.Tensor
            no_parents_prob is a tensor of shape (n_channels_next_layer, grid_size_next_layer, grid_size_next_layer),
            i.e. it has the same shape as the bricks in the next layer. Each component contains the information going
            to the next layer in terms of the probability that there's no parents pointing to the corresponding brick

        """
        state = self._normalize_state(state)
        no_parents_prob = torch.sum(
            state.view(
                self.n_channels, self.grid_size, self.grid_size, self.n_templates + 1, 1, 1, 1
            ) * (1 - self.expanded_templates.to_dense().float()), dim=3
        )
        no_parents_prob = torch.prod(
            no_parents_prob.view(
                self.n_bricks, self.params['n_channels_next_layer'],
                self.params['grid_size_next_layer'], self.params['grid_size_next_layer']
            ), dim=0
        )
        return no_parents_prob

    def get_on_bricks_prob(self, state):
        """get_on_bricks_prob
        Get the probability that different bricks are on in this layer

        Parameters
        ----------

        state : torch.Tensor
            state is a tensor of shape self.state_shape, and contains the states for which we want to
            get the on_bricks_prob

        Returns
        -------

        on_bricks_prob : torch.Tensor
            on_bricks_prob is a tensor of shape self.shape, and contains the probability that the brciks
            are on in this layer

        """
        state = self._normalize_state(state)
        on_bricks_prob = torch.sum(state[..., 1:], dim=-1)
        return on_bricks_prob

    def draw_sample(self, no_parents_prob):
        """draw_sample
        Draw samples from this layer, given no_parents_prob from the previous layer

        Parameters
        ----------

        no_parents_prob : torch.Tensor
            no_parents_prob is a tensor of shape self.shape, i.e. it has the same shape as the bricks in
            this layer. Each component contains the information coming from the previous layer in terms of
            the probability that there's no parents pointing to the corresponding brick. Binary when we are
            doing sampling.

        Returns
        -------

        sample : torch.Tensor
            sample is a binary array of shape self.state_shape, and is a sample from the Markov backbone for the
            current layer

        """
        self._validate_no_parents_prob(no_parents_prob, True)
        prob = torch.zeros(self.state_shape)
        if torch.sum(no_parents_prob == 1) > 0:
            prob[no_parents_prob == 1] = self.params['brick_self_rooting_prob']

        if torch.sum(no_parents_prob == 0) > 0:
            prob[no_parents_prob == 0] = self.params['brick_parent_prob']

        sample = fast_sample_from_categorical_distribution(prob)
        return sample

    def _validate_no_parents_prob(self, no_parents_prob, sampling):
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
        assert no_parents_prob.size() == self.shape
        if sampling:
            assert torch.sum(no_parents_prob == 0) + torch.sum(no_parents_prob == 1) == no_parents_prob.numel()
        else:
            assert torch.sum((no_parents_prob >= 0) * (no_parents_prob <= 1)) == no_parents_prob.numel()

    def _normalize_state(self, state):
        """_normalize_state
        Normalize the state if it's unnormalized, and check to make sure the state is valid

        Parameters
        ----------

        state : torch.Tensor
            state is a tensor of shape self.state_shape. Can be unnormalized.

        Returns
        -------

        """
        if not np.allclose(torch.sum(state, dim=3).detach().numpy(), 1):
            exp_state = torch.exp(state)
            state = exp_state / torch.sum(
                exp_state, dim=3, keepdim=True
            )

        assert state.size() == self.state_shape
        assert np.allclose(torch.sum(state, dim=3).detach().numpy(), 1)
        return state

    def _expand_templates(self):
        """_expand_templates
        Expand the templates array, which is of shape (n_templates, n_channels_next_layer, kernel_size, kernel_size),
        to an expanded_templates sparse matrix, which is of shape
        (n_channels, grid_size, grid_size, n_templates + 1, n_channels_next_layer, grid_size_next_layer, grid_size_next_layer)
        I.e. we are expanding the templates to be full size of next layer, and also add the template for the off state.
        Exists for easy calculation of the no_parents_prob array

        """
        templates = self.params['templates']
        templates_coords = templates._indices().cpu().numpy()
        templates_data = templates._values().cpu().numpy()
        n_nonzeros = templates_data.size
        coords_x = np.repeat(np.arange(self.grid_size, dtype=int), self.grid_size)
        coords_y = np.tile(np.arange(self.grid_size, dtype=int), self.grid_size)
        coords_x = np.repeat(coords_x, n_nonzeros)
        coords_y = np.repeat(coords_y, n_nonzeros)
        templates_coords = np.tile(templates_coords, self.grid_size**2)
        templates_data = np.tile(templates_data, self.grid_size**2)
        templates_coords[0] = templates_coords[0] + 1
        templates_coords[2] = templates_coords[2] + self.params['stride'] * coords_x
        templates_coords[3] = templates_coords[3] + self.params['stride'] * coords_y
        n_nonzeros = templates_data.size
        coords_channel = np.repeat(np.arange(self.n_channels, dtype=int), n_nonzeros)
        coords_x = np.tile(coords_x, self.n_channels)
        coords_y = np.tile(coords_y, self.n_channels)
        templates_coords = np.tile(templates_coords, self.n_channels)
        templates_data = np.tile(templates_data, self.n_channels)
        n_nonzeros = templates_data.size
        coords = np.zeros((7, n_nonzeros), dtype=int)
        coords[0] = coords_channel
        coords[1] = coords_x
        coords[2] = coords_y
        coords[3:] = templates_coords
        self.expanded_templates = torch.sparse_coo_tensor(coords, templates_data, (
            self.n_channels, self.grid_size, self.grid_size, self.n_templates + 1, self.params['n_channels_next_layer'],
            self.params['grid_size_next_layer'], self.params['grid_size_next_layer']
        ))


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


def calc_log_prob(state, prob):
    state = state[prob > 0]
    prob = prob[prob > 0]
    return torch.sum(state * torch.log(prob))
