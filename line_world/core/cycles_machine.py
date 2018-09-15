import numpy as np
import torch
import line_world.coarse.coarse_ops as co
from tqdm import tqdm
from line_world.utils import ParamsProc, Component, Optional
from line_world.core.layer import Layer
from line_world.perturb.factory import create_cycles_perturbation
from line_world.sample.markov_backbone import draw_sample_markov_backbone
from line_world.coarse.coarse_layer import CoarseLayer


class CyclesMachine(Component):
    @staticmethod
    def get_proc():
        proc = ParamsProc()
        proc.add(
            'layer_params_list', list,
            'The parameters for the different layers in the Markov backbone'
        )
        proc.add(
            'coarse_layer_params_list', list,
            'The parameters for the different coarse layers in the cycles machine',
            []
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

        self.coarse_layer_dict = {}
        for coarse_layer_params in self.params['coarse_layer_params_list']:
            self.add_coarse_layer(coarse_layer_params)

        self.cycles_perturbation = create_cycles_perturbation(
            self.params['cycles_perturbation_implementation'],
            self.layer_list, self.params['n_samples'],
            self.params['cycles_perturbation_params']
        )

    def add_coarse_layer(self, params):
        for key in ['index_to_duplicate', 'index_to_point_to', 'templates']:
            assert key in params

        layer_list = self.layer_list[params['index_to_duplicate']:params['index_to_point_to']]
        stride, kernel_size = co.get_coarse_stride_kernel_size(layer_list)
        assert params['templates'].size(2) == kernel_size
        assert params['templates'].size(3) == kernel_size
        params['stride'] = stride
        indices = (params['index_to_duplicate'], params['index_to_point_to'])
        coarse_layer = CoarseLayer(params)
        coarse_layer.expand_templates(self.layer_list)
        self.coarse_layer_dict.get(indices, []).append(coarse_layer)

    def draw_sample_markov_backbone(self):
        layer_sample_list = draw_sample_markov_backbone(self.layer_list)
        coarse_sample_dict = self.draw_coarse_sample(layer_sample_list)
        return layer_sample_list, coarse_sample_dict

    def draw_coarse_sample(self, state_list):
        coarse_sample_dict = co.draw_coarse_sample(
            self.layer_list, self.coarse_layer_dict, state_list
        )
        return coarse_sample_dict

    def draw_sample_rejection_sampling(self):
        print('Drawing sample using rejection sampling')
        with tqdm() as pbar:
            while True:
                layer_sample_list, coarse_sample_dict = self.draw_sample_markov_backbone()
                perturbation = torch.exp(
                    self.cycles_perturbation.get_discrete_log_prob_cycles_perturbation(
                        layer_sample_list, coarse_sample_dict
                    )
                )
                acceptance_prob = perturbation / self.cycles_perturbation.perturbation_upperbound
                if acceptance_prob > 1:
                    acceptance_prob = torch.tensor(1)

                if torch.bernoulli(acceptance_prob):
                    break

                pbar.update()

        return layer_sample_list, coarse_sample_dict

    def get_energy(self, state_list, coarse_state_dict={}):
        log_prob = log_prob_markov_backbone(state_list, self.layer_list) + \
            self.log_prob_markov_coarse_branches(state_list, coarse_state_dict) + \
            self.cycles_perturbation.get_log_prob_cycles_perturbation(state_list, coarse_state_dict)

        return log_prob

    def evaluate_energy_gradients(self, state_list, coarse_state_dict={}):
        flag = False
        for state in state_list:
            if state.requires_grad:
                flag = True

        for indices in coarse_state_dict:
            for state in coarse_state_dict[indices]:
                if state.requires_grad:
                    flag = True

        assert flag

        loss = -self.get_energy(state_list, coarse_state_dict)
        loss.backward()
        return -loss.item()

    def log_prob_markov_coarse_branches(self, state_list, coarse_state_dict):
        log_prob = 0
        assert set(coarse_state_dict.keys()) == set(self.coarse_layer_dict.keys())
        for indices in coarse_state_dict:
            for cc, coarse_layer in enumerate(self.coarse_layer_dict[indices]):
                log_prob += coarse_layer.get_log_prob(
                    state_list, self.layer_list, coarse_state_dict[indices]
                )

        return log_prob


def log_prob_markov_backbone(state_list, layer_list):
    assert len(state_list) == len(layer_list)
    no_parents_prob = torch.ones(layer_list[0].shape)
    log_prob = 0
    for ll, layer in enumerate(layer_list[:-1]):
        log_prob = log_prob + layer.get_log_prob(state_list[ll], no_parents_prob)
        no_parents_prob = layer.get_no_parents_prob(state_list[ll])

    log_prob = log_prob + layer_list[-1].get_log_prob(state_list[-1], no_parents_prob)
    return log_prob
