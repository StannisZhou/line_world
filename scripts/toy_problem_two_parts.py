import numpy as np
import torch
import torch.optim
from line_world.toy.model import two_parts


self_rooting_prob_list = np.array([0.5, 0.001, 0.001])
perturbed_distribution = torch.tensor([0.01, 0.01, 0.97, 0.01])
sigma = 0.2
n_samples = int(2e3)

cycles_machine, optimal_state_list, optimal_log_prob = two_parts(
    self_rooting_prob_list, perturbed_distribution, sigma, n_samples
)

image = optimal_state_list[-1]

state_list = [
    torch.rand(cycles_machine.layer_list[0].state_shape).requires_grad_(),
    torch.rand(cycles_machine.layer_list[1].state_shape).requires_grad_(),
    image
]
optimizer = torch.optim.SGD(state_list, lr=0.1)
n_steps = 2000
for ii in range(n_steps):
    optimizer.zero_grad()
    log_prob = cycles_machine.evaluate_energy_gradients(state_list)
    print('Step #{}, log_prob: {}'.format(ii, log_prob))
    optimizer.step()

print('Final log_prob: {}'.format(cycles_machine.get_energy(state_list)))
print('Optimal log_prob: {}'.format(optimal_log_prob))
