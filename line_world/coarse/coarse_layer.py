from line_world.utils import ParamsProc, Component, Optional
import line_world.core.layer_ops as lo


class CoarseLayer(Component):
    @staticmethod
    def get_proc():
        proc = ParamsProc()
        proc.add(
            'index_to_duplicate', int,
            'The index of the fine layer we are duplicating, starting from 0 for the top'
        )
        proc.add(
            'index_to_point_to', int,
            'The index of the fine layer we are pointing to'
        )
        proc.add(
            'stride', int,
            'The stride for the coarse layer, inferred from the various fine layers involved'
        )
        proc.add(
            'templates', torch.Tensor,
            '''The array containing all the templates. The shape of the array is (T, C, kernel_size, kernel_size),
            where T is the number of templates for the bricks in this layer, C is the number of channels in the
            layer below this layer, and kernel_size is the size of the receptive field. The array is binary.
            '''
        )
        return proc

    @staticmethod
    def params_proc(params):
        params['n_templates'] = params['templates'].size(0)
        params['kernel_size'] = params['templates'].size(2)
        params['n_channels_next_layer'] = params['templates'].size(1)
        params['brick_self_rooting_prob'] = torch.cat((
            torch.tensor([1]), torch.zeros(params['n_templates'])
        ))
        params['brick_parent_prob'] = torch.cat((
            torch.zeros(1), torch.ones(params['n_templates']) / params['n_templates']
        ))

    @staticmethod
    def params_test(params):
        assert params['index_to_duplicate'] >= 0
        assert params['index_to_point_to'] > params['index_to_duplicate'] + 1

    def __init__(self, params):
        super.__init__(params)

    def expand_templates(self, layer_list):
        """expand_templates
        Expand the templates for ease of later use

        Parameters
        ----------

        layer_list : list
            layer_list is the full list of layer objects from the Markov backbone

        Returns
        -------

        """
        n_channels, grid_size, _ = layer_list[self.params['index_to_duplicate']].shape
        n_channels_next_layer, grid_size_next_layer, _ = layer_list[self.params['index_to_point_to']].shape
        assert n_channels_next_layer == self.params['n_channels_next_layer']
        self.expanded_templates = lo.expand_templates(
            self.params['templates'], self.params['stride'], n_channels, grid_size,
            n_channels_next_layer, grid_size_next_layer
        )

    def get_no_parents_prob(self, coarse_state, layer_list, aggregate=True):
        top_layer = layer_list[self.params['index_to_duplicate']]
        bottom_layer = layer_list[self.params['index_to_point_to']]
        no_parents_prob = lo.get_no_parents_prob(
            coarse_state, self.expanded_templates.to_dense().float(), top_layer.n_channels,
            top_layer.grid_size, top_layer.n_templates, bottom_layer.n_channels,
            bottom_layer.grid_size, aggregate
        )
        return no_parents_prob

    def get_connections(self, coarse_state, layer_list):
        indices = (self.params['index_to_duplicate'], self.params['index_to_point_to'])
        on_bricks_prob = layer_list[indices[0]].get_on_bricks_prob(coarse_state)
        parents_prob = 1 - self.get_no_parents_prob(coarse_state, layer_list, False)
        connections = on_bricks_prob.reshape((-1, 1)) * parents_prob.reshape((
            layer_list[indices[0]].n_bricks, layer_list[indices[1]].n_bricks
        ))
        connections = connections.reshape(layer_list[indices[0]].shape + layer_list[indices[1]].shape)
        return connections

    def draw_sample(self, state):
        no_parents_prob = 1 - lo.get_on_bricks_prob(state, state.shape)
        lo.validate_no_parents_prob(no_parents_prob, True, state.shape[:3])
        prob = torch.zeros(state.shape)
        if torch.sum(no_parents_prob == 1) > 0:
            prob[no_parents_prob == 1] = self.params['brick_self_rooting_prob']

        if torch.sum(no_parents_prob == 0) > 0:
            prob[no_parents_prob == 0] = self.params['brick_parent_prob']

        sample = fast_sample_from_categorical_distribution(prob)
        return sample

    def get_log_prob(self, state, coarse_state):
        no_parents_prob = 1 - lo.get_on_bricks_prob(state, state.shape)
        lo.validate_no_parents_prob(
            no_parents_prob, False, state_list[self.params['index_to_duplicate']].shape[:3]
        )
        log_prob = lo.get_log_prob(
            coarse_state, no_parents_prob, self.params['brick_self_rooting_prob'], self.params['brick_parent_prob']
        )
        return log_prob
