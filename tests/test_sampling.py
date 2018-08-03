from line_world.layer import fast_sample_from_categorical_distribution
import torch
import numpy as np
from collections import Counter


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
