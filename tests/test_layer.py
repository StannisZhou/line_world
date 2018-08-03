from test_model import simple_model
import torch
import numpy as np
from line_world.perturbation import draw_samples_markov_backbone


def test_expanded_templates(simple_model):
    for layer in simple_model.layer_list[:-1]:
        stride = layer.params['stride']
        templates = layer.params['templates'].to_dense().numpy()
        expanded_templates = layer.expanded_templates.to_dense().numpy()
        _, _, kernel_size, _ = templates.shape
        n_channels, grid_size, _, n_templates_plus_one, _, _, _ = expanded_templates.shape
        assert n_channels == layer.n_channels
        assert grid_size == layer.grid_size
        for channel in range(n_channels):
            for ii in range(grid_size):
                for jj in range(grid_size):
                    assert np.all(expanded_templates[channel, ii, jj, 0] == 0)
                    for tt in range(1, n_templates_plus_one):
                        temp_templates = expanded_templates[channel, ii, jj, tt]
                        temp_templates[
                            :, ii * stride:ii * stride + kernel_size, jj * stride:jj * stride + kernel_size
                        ] -= templates[tt - 1]
                        assert np.all(temp_templates == 0)


def test_no_parents_prob(simple_model):
    layer_sample_list = draw_samples_markov_backbone(simple_model.layer_list)
    for ll, layer in enumerate(simple_model.layer_list[:-1]):
        layer_sample = layer_sample_list[ll].numpy()
        no_parents_prob = layer.get_no_parents_prob(layer_sample_list[ll]).numpy()
        expected_no_parents_prob = np.zeros_like(no_parents_prob)
        expanded_templates = layer.expanded_templates.to_dense().numpy()
        n_channels, grid_size, _, n_templates_plus_one, _, _, _ = expanded_templates.shape
        for channel in range(n_channels):
            for ii in range(grid_size):
                for jj in range(grid_size):
                    ind = np.nonzero(layer_sample[channel, ii, jj])[0]
                    assert len(ind) == 1
                    ind = ind[0]
                    expected_no_parents_prob += expanded_templates[channel, ii, jj, ind]

        expected_no_parents_prob[expected_no_parents_prob != 0] = 1
        expected_no_parents_prob = 1 - expected_no_parents_prob
        assert np.all(no_parents_prob == expected_no_parents_prob)


def test_log_prob(simple_model):
    layer_sample_list = draw_samples_markov_backbone(simple_model.layer_list)
    no_parents_prob = torch.ones(simple_model.layer_list[0].shape)
    for ll, layer in enumerate(simple_model.layer_list):
        layer_sample = layer_sample_list[ll]
        on_bricks_prob = layer.get_on_bricks_prob(layer_sample)
        assert np.all(on_bricks_prob[no_parents_prob == 0].numpy() == 1)
        log_prob = layer.get_log_prob(layer_sample, no_parents_prob, 0).item()
        if ll < len(simple_model.layer_list) - 1:
            n_templates = layer.params['templates'].shape[0]
        else:
            n_templates = 1

        n_has_parents = torch.sum(1 - no_parents_prob).item()
        n_self_rooting = torch.sum(on_bricks_prob[no_parents_prob == 1] == 1).item()
        self_rooting_prob = layer.params['self_rooting_prob']
        expected_log_prob = -n_has_parents * np.log(n_templates) + \
            n_self_rooting * np.log(self_rooting_prob / n_templates) + \
            (layer.n_bricks - n_has_parents - n_self_rooting) * np.log(1 - self_rooting_prob)
        assert np.isclose(log_prob, expected_log_prob)
        if ll < len(simple_model.layer_list) - 1:
            no_parents_prob = layer.get_no_parents_prob(layer_sample)
