from tqdm import tqdm
import torch
import logging
import numpy as np
from line_world.sample.markov_backbone import draw_sample_markov_backbone
from line_world.sample.fast_markov_backbone import fast_sample_markov_backbone


def get_n_cycles(state_list, layer_list):
    """get_n_cycles
    Get the number of cycles for different layers at a given state

    Parameters
    ----------

    state_list : list
        state_list is a list of torch tensors representing the states of different layers
    layer_list : list
        layer_list is a list of Layer objects representing the different layers in the model

    Returns
    -------

    n_cycles_list : list
        A list of length n_layers - 2 containing the number of cycles at each brick.
        Each element is a tensor of the same shape as the corresponding layer

    """
    assert len(state_list) == len(layer_list)
    if type(state_list[0]) is not torch.Tensor:
        state_list = [torch.tensor(state, dtype=torch.float) for state in state_list]

    n_cycles_list = [
        get_n_cycles_three_layers(state_list[ii:ii + 3], layer_list[ii:ii + 3])
        for ii in range(len(state_list) - 2)
    ]
    return n_cycles_list


def get_n_cycles_three_layers(state_list, layer_list):
    """get_n_cycles_three_layers
    Function for getting the number of cycles for three layers

    Parameters
    ----------

    state_list : list
        state_list is a list of length 3
    layer_list : list
        layer_list is a list of length 3

    Returns
    -------

    n_cycles : torch.Tensor
        n_cycles is a tensor of the same size as the top layer, containing the number of
        cycles for each brick in the top layer

    """
    assert len(state_list) == 3
    assert len(layer_list) == 3
    on_bricks_prob_list = [layer_list[ii].get_on_bricks_prob(state_list[ii]) for ii in range(3)]
    parents_prob_list = [1 - layer_list[ii].get_no_parents_prob(state_list[ii], False) for ii in range(2)]
    n_cycles = on_bricks_prob_list[0].reshape((-1, 1)) * parents_prob_list[0].reshape((
        layer_list[0].n_bricks, layer_list[1].n_bricks
    ))
    n_cycles = n_cycles * on_bricks_prob_list[1].reshape((1, -1))
    n_cycles = torch.matmul(n_cycles, parents_prob_list[1].reshape(
        layer_list[1].n_bricks, layer_list[2].n_bricks
    ))
    n_cycles = n_cycles * on_bricks_prob_list[2].reshape((1, -1))
    n_cycles = n_cycles * (n_cycles - 1) / 2
    n_cycles = n_cycles * (n_cycles > 0).float()
    n_cycles = torch.sum(n_cycles, dim=1).reshape(layer_list[0].shape)
    return n_cycles


class CyclesPerturbation(object):
    def __init__(self, layer_list, n_samples, fast_sample):
        self.layer_list = layer_list
        logging.info('Getting samples for the null distribution on the number of cycles')
        if fast_sample:
            state_list_samples = fast_sample_markov_backbone(layer_list, n_samples)
        else:
            state_list_samples = [
                draw_sample_markov_backbone(layer_list) for _ in tqdm(range(n_samples))
            ]

        self.n_cycles_statistics = [
            get_n_cycles(state_list, layer_list) for state_list in state_list_samples
        ]

    @property
    def perturbation_upperbound(self):
        raise Exception('Must be implemented')

    def get_log_prob_cycles_perturbation(self, state_list, coarse_state_dict={}):
        raise Exception("Must be implemented")

    def get_discrete_log_prob_cycles_perturbation(self, state_list, coarse_state_dict={}):
        raise Exception("Must be implemented")


class MarkovBackbone(CyclesPerturbation):
    def __init__(self, layer_list, n_samples, params):
        pass

    @property
    def perturbation_upperbound(self):
        return torch.tensor(1)

    def get_log_prob_cycles_perturbation(self, state_list, coarse_state_dict={}):
        return torch.tensor(0, dtype=torch.float)

    def get_discrete_log_prob_cycles_perturbation(self, state_list, coarse_state_dict={}):
        return torch.tensor(0, dtype=torch.float)
