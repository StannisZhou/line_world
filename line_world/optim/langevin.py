import torch.optim.optimizer
from line_world.optim.anneal import create_annealer
from line_world.optim.noise import create_noise


class LangevinDynamics(torch.optim.optimizer.Optimizer):
    def __init__(self, state_list, lr, annealer_implementation, annealer_params, noise_implementation, noise_params):
        super.__init__(state_list, dict(lr=lr))
        self.annealer = create_annealer(annealer_implementation, annealer_params)
        self.noise = create_noise(noise_implementation, noise_params)
        assert len(self.params_groups) == 1

    def step(self, closure=None):
            """Performs a single optimization step.
            Arguments:
                closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
            """
            loss = None
            if closure is not None:
                loss = closure()

            group = self.params_groups[0]
            noisy_gradients = self.noise.noisy_gradients(group['params'])
            annealer_strength = self.annealer.strength
            for ii, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                d_p = p.grad.data + annealer_strength * noisy_gradients[ii]
                p.data.add_(-group['lr'], d_p)

            return loss
