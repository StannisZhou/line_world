from line_world.core.layer_ops import fast_sample_from_categorical_distribution
import torch
import numpy as np
from collections import Counter
from line_world.numba.layer import sample_from_categorical_distribution_numba


def test_sampling():
    n_samples = 200000
    prob = torch.rand((1, 10))
    prob = prob / torch.sum(prob, dim=1, keepdim=True)
    expanded_prob = prob.expand((n_samples, -1))
    sample = fast_sample_from_categorical_distribution(expanded_prob)
    indices = np.nonzero(sample.numpy())[1]
    counts = torch.tensor([np.sum(indices == ii) for ii in range(10)], dtype=torch.float)
    expected_counts = n_samples * prob
    differences = counts - expected_counts


def test_numba_sampling():
    n_samples = 200000
    prob = np.random.rand(1, 10)
    prob = prob / np.sum(prob, axis=1, keepdims=True)
    expanded_prob = np.tile(prob, (n_samples, 1))
    sample = sample_from_categorical_distribution_numba(expanded_prob)
    indices = np.nonzero(sample)[1]
    counts = np.array([np.sum(indices == ii) for ii in range(10)], dtype=float)
    expected_counts = n_samples * prob
    differences = counts - expected_counts
