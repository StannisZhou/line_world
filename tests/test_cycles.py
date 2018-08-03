from test_model import toy_model
from line_world.perturbation import draw_samples_markov_backbone, get_n_cycles
import numpy as np


def test_count_cycles(toy_model):
    layer_sample_list = draw_samples_markov_backbone(toy_model.layer_list)
    n_cycles_list = get_n_cycles(layer_sample_list, toy_model.layer_list)
    ind = np.nonzero(layer_sample_list[1][..., 1:].numpy())
    expanded_templates = toy_model.layer_list[1].expanded_templates.to_dense().numpy()
    expected_n_cycles = 0
    for ii in range(2):
        for jj in range(ii + 1, 3):
            expected_n_cycles += np.sum(
                expanded_templates[ind[0][ii], ind[1][ii], ind[2][ii], ind[3][ii] + 1] *
                expanded_templates[ind[0][jj], ind[1][jj], ind[2][jj], ind[3][jj] + 1]
            )

    assert expected_n_cycles == n_cycles_list[0][0, 0, 0]
