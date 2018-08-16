import torch
from line_world.utils import SCALING_CONSTANT


class Noise(object):
    def __init__(self):
        pass

    def noisy_gradients(self, state_list):
        raise Exception('Must be implemented')


def create_noise(implementation, params):
    if implementation == 'istropic_gaussian':
        assert set(params.keys()) == set(['sigma'])
        return IstropicGaussian(**params)
    else:
        raise Exception('Unsupported implementation {}'.format(implementation))


class IstropicGaussian(Noise):
    def __init__(self, sigma):
        self.sigma = sigma

    def noisy_gradients(self, state_list):
        noise = [
            self.sigma * torch.randn(state.shape) for state in state_list
        ]
        return noise


class MarkovBackbone(Noise):
    def __init__(self, cycles_machine, sigma):
        self.cycles_machine = cycles_machine
        self.sigma = sigma

    def noisy_gradients(self, state_list):
        layer_sample_list = self.cycles_machine.draw_samples_markov_backbone()
        for layer_sample, state in zip(layer_sample_list, state_list):
            assert layer_sample.shape == state.shape

        layer_sample_list = [
            (SCALING_CONSTANT * layer_sample).requires_grad_() for layer_sample in layer_sample_list
        ]
        for layer_sample in layer_sample_list:
            layer_sample.grad.detach_()
            layer_sample.grad.zero_()

        self.cycles_machine.evaluate_energy_gradients(layer_sample_list)
        noise = [
            self.sigma * layer_sample.grad.data for layer_sample in layer_sample_list
        ]
        return noise
