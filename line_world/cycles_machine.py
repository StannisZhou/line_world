from line_world.utils import ParamsProc, Component, Optional
import numpy as np
from line_world.layer import Layer
from line_world.perturbation import create_cycles_perturbation
import torch


class CyclesMachine(Component):
    @staticmethod
    def get_proc():
        proc = ParamsProc()
        proc.add(
            'layer_params_list', list,
            'The parameters for the different layers in the Markov backbone'
        )
        proc.add(
            'cycles_perturbation_implementation', str,
            'The cycles perturbation implementation we are going to use',
            'markov_backbone'
        )
        proc.add(
            'cycles_perturbation_params', dict,
            'The parameters for the cycles perturbation',
            {}
        )
        proc.add(
            'n_samples', int,
            'The number of samples we are going to use to estimate the null distribution for the number of cycles',
            0
        )
        return proc

    @staticmethod
    def params_proc(params):
        pass

    @staticmethod
    def params_test(params):
        assert params['layer_params_list'][-1]['n_channels'] == 1
        for ll, layer_params in enumerate(params['layer_params_list'][:-1]):
            _, n_channels, kernel_size, _ = layer_params['templates'].shape
            grid_size = (layer_params['grid_size'] - 1) * layer_params['stride'] + kernel_size
            assert n_channels == params['layer_params_list'][ll + 1]['n_channels']
            assert grid_size == params['layer_params_list'][ll + 1]['grid_size']
            assert type(params['layer_params_list'][ll]['stride']) is not Optional
            assert type(params['layer_params_list'][ll]['templates']) is not Optional

    def __init__(self, params):
        super().__init__(params)
        self.layer_list = []
        for layer_params in self.params['layer_params_list']:
            self.layer_list.append(Layer(layer_params))

        self.cycles_perturbation = create_cycles_perturbation(
            self.params['cycles_perturbation_implementation'],
            self.layer_list, self.params['n_samples'],
            self.params['cycles_perturbation_params']
        )

    def get_energy(self, state_list):
        log_prob = log_prob_markov_backbone(state_list, self.layer_list) + \
            self.cycles_perturbation.get_log_prob_cycles_perturbation(state_list)

        return log_prob

    def evaluate_energy_gradients(self, state_list):
        flag = False
        for state in state_list:
            if state.requires_grad:
                flag = True

        assert flag

        loss = -self.get_energy(state_list)
        loss.backward()
        return -loss.item()


def log_prob_markov_backbone(state_list, layer_list):
    assert len(state_list) == len(layer_list)
    no_parents_prob = torch.ones(layer_list[0].shape)
    log_prob = 0
    for ll, layer in enumerate(layer_list[:-1]):
        log_prob = log_prob + layer.get_log_prob(state_list[ll], no_parents_prob)
        no_parents_prob = layer.get_no_parents_prob(state_list[ll])

    log_prob = log_prob + layer_list[-1].get_log_prob(state_list[-1], no_parents_prob)
    return log_prob
