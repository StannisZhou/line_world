import torch
import logging
import numpy as np
import line_world.coarse.coarse_ops as co


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
    n_cycles = co.get_interlayer_connections((0, 2), state_list, layer_list)
    n_cycles = n_cycles * (n_cycles - 1) / 2
    n_cycles = n_cycles * (n_cycles > 0).float()
    n_cycles = torch.sum(n_cycles, dim=[3, 4, 5])
    return n_cycles


class CyclesPerturbation(object):
    def __init__(self, layer_list):
        self.layer_list = layer_list

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
