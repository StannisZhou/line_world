from line_world.utils import ParamsProc, Component, Optional, NO_PARENTS_PROB_MARGIN
import line_world.core.layer_ops as lo
import numpy as np
import torch
import torch.nn.functional


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
            self.expanded_templates = lo.expand_templates(
                self.params['templates'], self.params['stride'], self.n_channels, self.grid_size,
                self.params['n_channels_next_layer'], self.params['grid_size_next_layer']
            )

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
        assert state.shape == self.state_shape
        log_prob = lo.get_log_prob(
            state, no_parents_prob, self.params['brick_self_rooting_prob'], self.params['brick_parent_prob']
        )
        return log_prob

    def get_no_parents_prob(self, state, aggregate=True):
        """get_no_parents_prob
        Get the no_parents_prob array of the next layer

        Parameters
        ----------

        state : torch.Tensor
            state is a tensor of shape self.state_shape, which contains the state for which we
            want to get the no_parents_prob array for the next layer
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
        no_parents_prob = lo.get_no_parents_prob(
            state, self.expanded_templates.to_dense().float(), self.n_channels,
            self.grid_size, self.n_templates, self.params['n_channels_next_layer'],
            self.params['grid_size_next_layer'], aggregate
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
        assert state.shape == self.state_shape
        on_bricks_prob = lo.get_on_bricks_prob(state)
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

        sample = lo.fast_sample_from_categorical_distribution(prob)
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
        lo.validate_no_parents_prob(no_parents_prob, sampling, self.shape)
