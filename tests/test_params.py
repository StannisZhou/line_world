from line_world.params import generate_latent_templates
import torch
import numpy as np
import scipy.special
from scipy.spatial.distance import pdist
from tqdm import tqdm


def test_latent_templates():
    n_channels = 3
    kernel_size = 3
    n_parts = 3
    latent_templates = generate_latent_templates(n_channels, kernel_size).to_dense()
    n_templates = latent_templates.size(0)
    assert n_templates == int(scipy.special.comb(kernel_size**2, n_parts) * n_channels**n_parts)
    assert np.allclose(torch.sum(latent_templates.reshape((n_templates, -1)), dim=1).numpy(), n_parts)
    pairwise_distances = pdist(latent_templates.reshape((n_templates, -1)).numpy(), 'euclidean')
    assert np.all(pairwise_distances > 0)
