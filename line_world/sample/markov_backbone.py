import torch


def draw_sample_markov_backbone(layer_list):
    """draw_sample_markov_backbone
    Draw a single sample from the Markov backbone

    Parameters
    ----------

    layer_list : list
        layer_list is a list of layers

    Returns
    -------

    layer_sample_list : list
        A list of samples for different layer. Each element is a tensor of state_shape of
        each layer

    """
    layer_sample_list = []
    no_parents_prob = torch.ones(layer_list[0].shape)
    for layer in layer_list[:-1]:
        layer_sample = layer.draw_sample(no_parents_prob)
        no_parents_prob = layer.get_no_parents_prob(layer_sample)
        layer_sample_list.append(layer_sample)

    layer_sample_list.append(layer_list[-1].draw_sample(no_parents_prob))
    return layer_sample_list
