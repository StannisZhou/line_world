import logging
from tqdm import tqdm
import torch
import numpy as np
from line_world.perturb.perturbation import CyclesPerturbation, get_n_cycles, get_n_cycles_three_layers
from line_world.sample.markov_backbone import draw_sample_markov_backbone
from line_world.sample.fast_markov_backbone import fast_sample_markov_backbone
import line_world.coarse.coarse_ops as co


class ToyPerturbation(CyclesPerturbation):
    def __init__(self, layer_list, coarse_layer_collections, n_samples, params):
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
                Whether we are going to use the fast sampler or not. Defaults to be False

        Returns
        -------

        """
        assert len(layer_list) == 3
        assert layer_list[0].shape == torch.Size([1, 1, 1])
        for coarse_layer_for_layer in coarse_layer_collections:
            assert len(coarse_layer_for_layer) == 0

        params['fast_sample'] = params.get('fast_sample', False)
        assert [key in params for key in ['perturbed_distribution', 'sigma']]
        super().__init__(layer_list, coarse_layer_collections)
        logging.info('Getting samples for the null distribution on the number of cycles')
        if params['fast_sample']:
            state_list_samples = fast_sample_markov_backbone(layer_list, n_samples)
        else:
            state_list_samples = [
                draw_sample_markov_backbone(
                    layer_list, coarse_layer_collections
                )[0] for _ in tqdm(range(n_samples))
            ]

        self.n_cycles_statistics = [
            get_n_cycles(state_list, layer_list) for state_list in state_list_samples
        ]
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

    def get_log_prob_cycles_perturbation(self, state_list, coarse_state_collections=None):
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

    def get_discrete_log_prob_cycles_perturbation(self, state_list, coarse_state_collections=None):
        self._validate_state(state_list)
        n_cycles = int(get_n_cycles_three_layers(state_list, self.layer_list)[0, 0, 0].item())
        null_prob = self.null_distribution[n_cycles]
        perturbed_prob = self.params['perturbed_distribution'][n_cycles]
        return torch.log(perturbed_prob) - torch.log(null_prob)

    def _validate_state(self, state_list):
        for state in state_list:
            assert np.allclose(torch.sum(state, dim=3).detach().numpy(), 1)
            assert torch.sum((state == 0) + (state == 1)) == state.numel()

    def _get_null_distribution(self):
        n_cycles_list = np.array([x[0][0, 0, 0].item() for x in self.n_cycles_statistics], dtype=int)
        null_distribution = np.bincount(n_cycles_list) / len(n_cycles_list)
        return torch.tensor(null_distribution, dtype=torch.float)


class ToyCoarsePerturbation(CyclesPerturbation):
    def __init__(self, layer_list, coarse_layer_collections, n_samples, params):
        """__init__

        Parameters
        ----------

        layer_list : list
            layer_list is a list of Layer objects
        coarse_layer_collections : dict
            coarse_layer_collections is a dictionary containing all the coarse layer objects
        n_samples : int
            n_samples is the number of samples we are going to use to estimate the null distribution
        params : dict
            params is a dictionary containing the relevant parameters
            params['perturbed_distribution_fine'] : torch.Tensor
                A torch tensor containing the perturbed distribution on the number of cycles
                for the fine layers
            params['perturbed_distribution_coarse'] : torch.Tensor
                A torch tensor containing the perturbed distribution on the number of cycles
                involving the coarse layers
            params['Sigma'] : torch.Tensor
                A torch tensor of shape (2, 2), which represents the covariance matrix we are
                going to use for the interpolation kernel

        Returns
        -------

        """
        assert len(layer_list) == 3
        assert len(coarse_layer_collections) == 3
        assert layer_list[0].shape == torch.Size([1, 1, 1])
        assert len(coarse_layer_collections[0]) == 1
        assert len(coarse_layer_collections[1]) == 0
        assert len(coarse_layer_collections[2]) == 0
        assert [key in params for key in [
            'perturbed_distribution_fine', 'perturbed_distribution_coarse', 'Sigma']
        ]
        super().__init__(layer_list, coarse_layer_collections)
        logging.info('Getting samples for the null distribution on the number of cycles')
        state_list_samples = []
        coarse_state_collections_samples = []
        for _ in tqdm(range(n_samples)):
            layer_sample_list, coarse_sample_collections = draw_sample_markov_backbone(
                layer_list, coarse_layer_collections
            )
            state_list_samples.append(layer_sample_list)
            coarse_state_collections_samples.append(coarse_sample_collections)

        n_cycles_statistics = []
        for state_list, coarse_state_collections in zip(state_list_samples, coarse_state_collections_samples):
            n_cycles_statistics.append((
                get_n_cycles(state_list, layer_list),
                co.get_n_coarse_cycles(state_list, coarse_state_collections, layer_list, coarse_layer_collections)
            ))

        self.n_cycles_statistics = n_cycles_statistics
        self.null_distribution = self._get_null_distribution()
        assert np.isclose(torch.sum(params['perturbed_distribution_fine']).item(), 1)
        assert np.isclose(torch.sum(params['perturbed_distribution_coarse']).item(), 1)
        self._set_perturbed_distribution()
        self.params = params

    def get_log_prob_cycles_perturbation(self, state_list, coarse_state_collections):
        n_cycles = int(get_n_cycles_three_layers(state_list, self.layer_list)[0, 0, 0].item())
        n_coarse_cycles = int(co.get_n_coarse_cycles(
            state_list, coarse_state_collections, self.layer_list, self.coarse_layer_collections
        )[0][0][0, 0, 0].item())
        n_cycles_pair = torch.tensor([n_cycles, n_coarse_cycles], dtype=torch.float)
        perturbation_kernel = torch.distributions.multivariate_normal.MultivariateNormal(
            n_cycles_pair, covariance_matrix=self.params['Sigma']
        )
        null_prob = torch.sum(
            self.null_distribution_pmf * torch.exp(
                perturbation_kernel.log_prob(self.null_distribution_points)
            )
        )
        perturbed_prob = torch.sum(
            self.perturbed_distribution_pmf * torch.exp(
                perturbation_kernel.log_prob(self.perturbed_distribution_points)
            )
        )
        return torch.log(perturbed_prob) - torch.log(null_prob)

    def get_discrete_log_prob_cycles_perturbation(self, state_list, coarse_state_collections):
        self._validate_state(state_list, coarse_state_collections)
        n_cycles = int(get_n_cycles_three_layers(state_list, self.layer_list)[0, 0, 0].item())
        n_coarse_cycles = int(co.get_n_coarse_cycles(
            state_list, coarse_state_collections, self.layer_list, self.coarse_layer_collection
        )[0][0][0, 0, 0].item())
        n_cycles_pair = (n_cycles, n_coarse_cycles)
        null_prob = self.null_distribution_dict[n_cycles_pair]
        perturbed_prob = self.perturbed_distribution_dict[n_cycles_pair]
        return torch.log(perturbed_prob) - torch.log(null_prob)

    def _validate_state(self, state_list, coarse_state_collections):
        for state in state_list:
            assert np.allclose(torch.sum(state, dim=3).detach().numpy(), 1)
            assert torch.sum((state == 0) + (state == 1)) == state.numel()

        for coarse_state_for_layer in coarse_state_collections:
            for state in coarse_state_for_layer:
                assert np.allclose(torch.sum(state, dim=3).detach().numpy(), 1)
                assert torch.sum((state == 0) + (state == 1)) == state.numel()

    def _set_null_distribution(self):
        n_cycles_pair_list = [(
            int(x[0][0, 0, 0].item()), int(y[0][0][0, 0, 0].item())
        ) for x, y in self.n_cycles_statistics]
        n_samples = len(n_cycles_pair_list)
        null_distribution_dict = {}
        for n_cycles_pair in n_cycles_pair_list:
            null_distribution_dict[n_cycles_pair] = null_distribution_dict.get(
                n_cycles_pair, 0
            ) + (1 / n_samples)

        null_distribution_points = torch.tensor(
            list(null_distribution_dict.keys()), dtype=torch.float
        )
        null_distribution_pmf = torch.tensor(
            list(null_distribution_dict.values()), dtype=torch.float
        )
        self.null_distribution_dict = null_distribution_dict
        self.null_distribution_points = null_distribution_points
        self.null_distribution_pmf = null_distribution_pmf

    def _set_perturbed_distribution(self):
        perturbed_distribution_fine = self.params['perturbed_distribution_fine']
        perturbed_distribution_coarse = self.params['perturbed_distribution_coarse']
        perturbed_distribution_dict = {}
        perturbation_upperbound = 0
        for ii in range(len(perturbed_distribution_fine)):
            for jj in range(len(perturbed_distribution_coarse)):
                prob = perturbed_distribution_fine[ii] * perturbed_distribution_coarse[ii]
                if prob > perturbation_upperbound:
                    perturbation_upperbound = prob
                perturbed_distribution_dict[(ii, jj)] = prob

        perturbed_distribution_points = torch.tensor(
            list(perturbed_distribution_dict.keys()), dtype=torch.float
        )
        perturbed_distribution_pmf = torch.tensor(
            list(perturbed_distribution_dict.values()), dtype=torch.float
        )
        self.perturbed_distribution_dict = perturbed_distribution_dict
        self.perturbed_distribution_points = perturbed_distribution_points
        self.perturbed_distribution_pmf = perturbed_distribution_pmf
        self.perturbation_upperbound = perturbation_upperbound
