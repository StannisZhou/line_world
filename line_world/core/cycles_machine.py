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

        self.coarse_layer_collections = [
            [] for _ in range(len(self.layer_list))
        ]
        for coarse_layer_params in self.params['coarse_layer_params_list']:
            self._add_coarse_layer(coarse_layer_params)

        self.cycles_perturbation = create_cycles_perturbation(
            self.params['cycles_perturbation_implementation'],
            self.layer_list, self.coarse_layer_collections,
            self.params['n_samples'], self.params['cycles_perturbation_params']
        )

    def _add_coarse_layer(self, params):
        for key in ['index_to_duplicate', 'index_to_point_to', 'templates']:
            assert key in params

        layer_list = self.layer_list[params['index_to_duplicate']:params['index_to_point_to']]
        stride_list = np.array([layer.params['stride'] for layer in layer_list], dtype=int)
        kernel_size_list = np.array([layer.params['kernel_size'] for layer in layer_list], dtype=int)
        stride, kernel_size = co.get_coarse_stride_kernel_size(stride_list, kernel_size_list)
        assert params['templates'].size(2) == kernel_size
        assert params['templates'].size(3) == kernel_size
        params['stride'] = stride
        coarse_layer = CoarseLayer(params)
        coarse_layer.expand_templates(self.layer_list)
        self.coarse_layer_collections[params['index_to_duplicate']].append(coarse_layer)

    def draw_sample_markov_backbone(self):
        return draw_sample_markov_backbone(self.layer_list, self.coarse_layer_collections)

    def draw_sample_rejection_sampling(self):
        print('Drawing sample using rejection sampling')
        with tqdm() as pbar:
            while True:
                layer_sample_list, coarse_sample_collections = self.draw_sample_markov_backbone()
                perturbation = torch.exp(
                    self.cycles_perturbation.get_discrete_log_prob_cycles_perturbation(
                        layer_sample_list, coarse_sample_collections
                    )
                )
                acceptance_prob = perturbation / self.cycles_perturbation.perturbation_upperbound
                if acceptance_prob > 1:
                    acceptance_prob = torch.tensor(1)

                if torch.bernoulli(acceptance_prob):
                    break

                pbar.update()

        return layer_sample_list, coarse_sample_collections

    def get_energy(self, state_list, coarse_state_collections=None):
        if not coarse_state_collections:
            coarse_state_collections = [[] for _ in range(len(self.layer_list))]

        log_prob = log_prob_markov_backbone(
            state_list, self.layer_list, coarse_state_collections, self.coarse_layer_collections
        ) + self.cycles_perturbation.get_log_prob_cycles_perturbation(
            state_list, coarse_state_collections
        )
        return log_prob

    def evaluate_energy_gradients(self, state_list, coarse_state_collections=None):
        if not coarse_state_collections:
            coarse_state_collections = [[] for _ in range(len(self.layer_list))]

        flag = False
        for state in state_list:
            if state.requires_grad:
                flag = True

        for coarse_state_for_layer in coarse_state_collections:
            for coarse_state in coarse_state_for_layer:
                if coarse_state.requires_grad:
                    flag = True

        assert flag

        loss = -self.get_energy(state_list, coarse_state_collections)
        loss.backward()
        return -loss.item()


def log_prob_markov_backbone(state_list, layer_list, coarse_state_collections, coarse_layer_collections):
    n_layers = len(layer_list)
    assert len(state_list) == n_layers
    assert len(coarse_state_collections) == n_layers
    assert len(coarse_layer_collections) == n_layers
    for ii in range(n_layers):
        assert len(coarse_state_collections[ii]) == len(coarse_layer_collections[ii])

    no_parents_prob_from_coarse_layers = [[] for _ in range(n_layers)]
    no_parents_prob = torch.ones(layer_list[0].shape)
    log_prob = 0
    for ll, layer in enumerate(layer_list[:-1]):
        for coarse_no_parents_prob in no_parents_prob_from_coarse_layers[ll]:
            no_parents_prob = no_parents_prob * coarse_no_parents_prob

        log_prob = log_prob + layer.get_log_prob(state_list[ll], no_parents_prob)
        no_parents_prob = layer.get_no_parents_prob(state_list[ll])
        for cc, coarse_layer in enumerate(coarse_layer_collections[ll]):
            log_prob = log_prob + coarse_layer.get_log_prob(
                state_list[ll], coarse_state_collections[ll][cc]
            )
            no_parents_prob_from_coarse_layers[coarse_layer.params['index_to_point_to']].append(
                coarse_layer.get_no_parents_prob(coarse_state_collections[ll][cc], layer_list)
            )

    for coarse_no_parents_prob in no_parents_prob_from_coarse_layers[-1]:
        no_parents_prob = no_parents_prob * coarse_no_parents_prob

    assert len(coarse_layer_collections[-1]) == 0
    log_prob = log_prob + layer_list[-1].get_log_prob(state_list[-1], no_parents_prob)
    return log_prob
