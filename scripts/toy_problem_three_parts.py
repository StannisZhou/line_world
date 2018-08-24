import numpy as np
from line_world.cycles_machine import CyclesMachine
from line_world.params import generate_cycles_machine_layer_params
from line_world.sample.markov_backbone import draw_sample_markov_backbone
from line_world.toy.model import three_parts
import torch
import torch.optim

self_rooting_prob_list = np.array([1.0, 0.01, 0.01])
perturbed_distribution = 0.0001 * torch.ones(6, dtype=torch.float)
perturbed_distribution[5] = 0.9995
sigma = 1.0
n_samples = int(3e3)

cycles_machine, optimal_state_list, optimal_log_prob = three_parts(
    self_rooting_prob_list, perturbed_distribution, sigma, n_samples
)
image = optimal_state_list[-1]
state_list = [
    torch.rand(cycles_machine.layer_list[0].state_shape).requires_grad_(),
    torch.rand(cycles_machine.layer_list[1].state_shape).requires_grad_(),
    #  (10 * optimal_top_layer).requires_grad_(),
    #  (10 * optimal_middle_layer).requires_grad_(),
    image
]
initial_weight_decay = 1.0
n_steps = 50
for ww in range(40):
    optimizer = torch.optim.SGD(state_list[:-1], lr=1.0, weight_decay=initial_weight_decay * 1.5**(-ww))
    for ii in range(n_steps):
        optimizer.zero_grad()
        log_prob = cycles_machine.evaluate_energy_gradients(state_list)
        print('Step #{}, log_prob: {}'.format(ww * n_steps + ii, log_prob))
        optimizer.step()

on_bricks_prob_list = [
    cycles_machine.layer_list[ii].get_on_bricks_prob(state) for ii, state in enumerate(state_list)
]
ind = torch.nonzero(state_list[0] > torch.max(state_list[0]) - 1)
expanded_templates = cycles_machine.layer_list[0].expanded_templates.to_dense()
print('Final log_prob: {}'.format(cycles_machine.get_energy(state_list)))
print('Optimal log_prob: {}'.format(optimal_log_prob))
