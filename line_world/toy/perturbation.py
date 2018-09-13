import torch
import numpy as np
from line_world.perturb.perturbation import CyclesPerturbation, get_n_cycles_three_layers


class ToyPerturbation(CyclesPerturbation):
    def __init__(self, layer_list, n_samples, params):
        """__init__

        Parameters
        ----------

        layer_list : list
            layer_list is a list of Layer objects
        n_samples : int
            n_samples is the number of samples we are going to use to estimate the null distribution
        params : dict
            params is a dictionary containing the relevant parameters
            params['perturbed_distribution'] : torch.Tensor
                A torch tensor containing the perturbed distribution on the number of cycles
            params['sigma'] : float
                The standard deviation we are going to use in the continuous interpolation of the cycles distribution
            params['fast_sample']: bool
                Whether we are going to use the fast sampler or not. Defaults to be True

        Returns
        -------

        """
        assert len(layer_list) == 3
        assert layer_list[0].shape == torch.Size([1, 1, 1])
        params['fast_sample'] = params.get('fast_sample', False)
        assert [key in params for key in ['perturbed_distribution', 'sigma']]
        super().__init__(layer_list, n_samples, params['fast_sample'])
        self.null_distribution = self._get_null_distribution()
        assert np.isclose(torch.sum(params['perturbed_distribution']).item(), 1)
        self.params = params

    @property
    def perturbation_upperbound(self):
        if hasattr(self, 'upper_bound'):
            return self.upper_bound
        else:
            upper_bound = 0
            assert len(self.params['perturbed_distribution']) <= len(self.null_distribution)
            for ii in range(len(self.params['perturbed_distribution'])):
                if self.null_distribution[ii] == 0:
                    assert self.params['perturbed_distribution'][ii] == 0
                    continue
                else:
                    temp = self.params['perturbed_distribution'][ii] / self.null_distribution[ii]
                    if temp > upper_bound:
                        upper_bound = temp

            self.upper_bound = upper_bound
            return upper_bound

    def get_log_prob_cycles_perturbation(self, state_list, coarse_state_dict={}):
        sigma = self.params['sigma']
        n_cycles = get_n_cycles_three_layers(state_list, self.layer_list)
        null_prob = torch.sum(
            self.null_distribution * torch.exp(
                -(torch.arange(self.null_distribution.numel()).float() - n_cycles)**2 / (2 * sigma**2)
            )
        )
        perturbed_prob = torch.sum(
            self.params['perturbed_distribution'] * torch.exp(
                -(torch.arange(self.params['perturbed_distribution'].numel()).float() - n_cycles)**2 / (2 * sigma**2)
            )
        )
        return torch.log(perturbed_prob) - torch.log(null_prob)

    def get_discrete_log_prob_cycles_perturbation(self, state_list, coarse_state_dict={}):
        self._validate_state(state_list)
        n_cycles = int(get_n_cycles_three_layers(state_list, self.layer_list)[0, 0, 0].item())
        null_prob = self.null_distribution[n_cycles]
        perturbed_prob = self.params['perturbed_distribution'][n_cycles]
        return torch.log(perturbed_prob) - torch.log(null_prob)

    def get_n_cycles(self, state_list):
        n_cycles = int(get_n_cycles_three_layers(state_list, self.layer_list)[0, 0, 0].item())
        return self.layer_list[-1].get_on_bricks_prob(state_list[-1]), n_cycles

    def _validate_state(self, state_list):
        for state in state_list:
            assert np.allclose(torch.sum(state, dim=3).detach().numpy(), 1)
            assert torch.sum((state == 0) + (state == 1)) == state.numel()

    def _get_null_distribution(self):
        n_cycles_list = np.array([x[0][0, 0, 0].item() for x in self.n_cycles_statistics], dtype=int)
        null_distribution = np.bincount(n_cycles_list) / len(n_cycles_list)
        return torch.tensor(null_distribution, dtype=torch.float)
