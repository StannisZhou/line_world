from line_world.perturb.perturbation import MarkovBackbone
from line_world.toy.perturbation import ToyPerturbation


def create_cycles_perturbation(implementation, layer_list, n_samples, params):
    """create_cycles_perturbation
    Factory method for creating cycles perturbation

    Parameters
    ----------

    implementation : str
        implementation is the implementation we are going to use for the perturbation.
        Supported implementations include:
            markov_backbone : no perturbation
            toy_perturbation : a simple perturbation for the toy case where the top layer
            has only one brick
    layer_list : list
        layer_list is a list of layers in the model
    n_samples : int
        n_samples is the number of cycles we are going to use to estimate the null distribution
        on the number of cycles
    params : dict
        params is a dictionary containing the various parameters for the cycles perturbation

    Returns
    -------

    A class for cycles perturbation

    """
    if implementation == 'markov_backbone':
        return MarkovBackbone(layer_list, n_samples, params)
    elif implementation == 'toy_perturbation':
        return ToyPerturbation(layer_list, n_samples, params)
    else:
        raise Except('Unsupported cycles perturbation implementation')
