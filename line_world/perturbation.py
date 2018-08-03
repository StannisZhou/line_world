from line_world.utils import ParamsProc, Component
from tqdm import tqdm
import torch
import logging


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


def draw_samples_markov_backbone(layer_list):
    """draw_samples_markov_backbone
    Draw samples from the Markov backbone

    Parameters
    ----------

    layer_list : list
        layer_list is a list of layers

    Returns
    -------

    layer_sample_list : list
        A list of samples for different layer. Each element is a tensor of state_shape of
        each layer

    """
    layer_sample_list = []
    no_parents_prob = torch.ones(layer_list[0].shape)
    for layer in layer_list[:-1]:
        layer_sample = layer.draw_sample(no_parents_prob)
        no_parents_prob = layer.get_no_parents_prob(layer_sample)
        layer_sample_list.append(layer_sample)

    layer_sample_list.append(layer_list[-1].draw_sample(no_parents_prob))
    return layer_sample_list


def create_cycles_perturbation(implementation, layer_list, n_samples, params):
    """create_cycles_perturbation
    Factory method for creating cycles perturbation

    Parameters
    ----------

    implementation : str
        implementation is the implementation we are going to use for the perturbation.
        Supported implementations include:
            markov_backbone : no perturbation
            toy_perturbation : a simple perturbation for the toy case where the top layer
            has only one brick
    layer_list : list
        layer_list is a list of layers in the model
    n_samples : int
        n_samples is the number of cycles we are going to use to estimate the null distribution
        on the number of cycles
    params : dict
        params is a dictionary containing the various parameters for the cycles perturbation

    Returns
    -------

    A class for cycles perturbation

    """
    if implementation == 'markov_backbone':
        return MarkovBackbone(layer_list, n_samples, params)
    elif implementation == 'toy_perturbation':
        return ToyPerturbation(layer_list, n_samples, params)
    else:
        raise Except('Unsupported cycles perturbation implementation')


class CyclesPerturbation(object):
    def __init__(self, layer_list, n_samples):
        self.layer_list = layer_lists
        logging.info('Getting samples for the null distribution on the number of cycles')
        state_list_samples = [
            draw_samples_markov_backbone(layer_list) for _ in tqdm(range(n_samples))
        ]
        self.n_cycles_statistics = [
            get_n_cycles(state_list, layer_list) for state_list in state_list_samples
        ]

    def get_log_prob_cycles_perturbation(self, state_list):
        raise Exception("Must be implemented")


class MarkovBackbone(CyclesPerturbation):
    def __init__(self, layer_list, n_samples, params):
        pass

    def get_log_prob_cycles_perturbation(self, state_list):
        return 0


class ToyPerturbation(CyclesPerturbation):
    def __init__(self, layer_list, n_samples, params):
        assert len(layer_list) == 3
        assert layer_list[0].shape == torch.Size([1, 1, 1])
        super().__init__(layer_list, n_samples)
        self.params = params

    def get_log_prob_cycles_perturbation(self, state_list):
        return 0
