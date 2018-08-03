from line_world.utils import ParamsProc, Component
from tqdm import tqdm
import torch


def log_prob_cycles_perturbation(state_list, layer_list):
    return 0


def get_n_cycles(state_list, layer_list):
    assert len(state_list) == len(layer_list)
    n_cycles_list = [
        get_n_cycles_three_layers(state_list[ii:ii + 3], layer_list[ii:ii + 2])
        for ii in range(len(state_list) - 2)
    ]
    return n_cycles_list


def get_n_cycles_three_layers(state_list, layer_list):
    assert len(state_list) == 3
    assert len(layer_list) == 2
    on_bricks_prob_list = [layer_list[ii].get_on_bricks_prob(state_list[ii]) for ii in range(3)]
    parents_prob_list = [1 - layer_list[ii].get_no_parents_prob(state_list[ii]) for ii in range(2)]
    n_cycles = on_bricks_prob_list[0].reshape((-1, 1)) * parents_prob_list[0].reshape((
        layer_list[0].n_bricks, layer_list[1].n_bricks
    ))
    n_cycles = n_cycles * on_bricks_prob_list[1].reshape((1, -1))
    n_cycles = torch.matmul(n_cycles, parents_prob_list[1].reshape(
        layer_list[1].n_bricks, -1
    ))
    n_cycles = n_cycles * on_bricks_prob_list[2].reshape((1, -1))
    n_cycles = n_cycles * (n_cycles - 1) / 2
    n_cycles = n_cycles * (n_cycles > 0)
    n_cycles = torch.sum(n_cycles, dim=1).reshape(layer_list[0].shape)
    return n_cycles


def draw_samples_markov_backbone(layer_list):
    layer_sample_list = []
    no_parents_prob = torch.ones(layer_list[0].shape)
    for layer in layer_list[:-1]:
        layer_sample = layer.draw_sample(no_parents_prob)
        no_parents_prob = layer.get_no_parents_prob(layer_sample)
        layer_sample_list.append(layer_sample)

    layer_sample_list.append(layer_list[-1].draw_sample(no_parents_prob))
    return layer_sample_list


def get_n_cycles_null_distribution(state_list, layer_list, n_samples):
    state_list_samples = [
        draw_samples_markov_backbone(layer_list) for _ in tqdm(range(n_samples))
    ]
    n_cycles_statistics = [
        get_n_cycles(state_list, layer_list) for state_list in state_list_samples
    ]
    return n_cycles_statistics
