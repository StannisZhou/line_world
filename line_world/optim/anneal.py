import numpy as np


class Annealer(object):
    def __init__(self):
        self.n_steps = 0

    @property
    def strength(self):
        raise Exception('Must be implemented')


def create_annealer(implementation, params):
    if implementation == 'constant':
        assert set(params.keys()) == 'init_strength'
        return ConstantAnnealer(**params)
    elif implementation == 'geometric':
        assert set(params.keys()) == set([
            'init_strength', 'decay_steps', 'decay_factors'
        ])
        return GeometricAnnealer(**params)
    else:
        raise Exception('Unsupported implementation {}'.format(implementation))


class ConstantAnnealer(Annealer):
    def __init__(self, init_strength):
        super().__init__()
        self.init_strength = init_strength

    @property
    def strength(self):
        return self.init_strength


class GeometricAnnealer(Annealer):
    def __init__(self, init_strength, decay_steps, decay_factors):
        super().__init__()
        self.init_strength = init_strength
        self.decay_steps = decay_steps
        self.decay_factors = decay_factors

    @property
    def strength(self):
        strength = self.init_strength / (self.decay_factors ** np.floor((self.n_steps + 1) / self.decay_steps))
        self.n_steps += 1
        return strength
