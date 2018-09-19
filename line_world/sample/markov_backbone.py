import torch


def draw_sample_markov_backbone(layer_list, coarse_layer_collections):
    """draw_sample_markov_backbone
    Draw a single sample from the Markov backbone

    Parameters
    ----------

    layer_list : list
        layer_list is a list of layers
    coarse_layer_collections : list
        coarse_layer_collections is a list of lists containing the list of coarse layers
        at different levels

    Returns
    -------

    layer_sample_list : list
        A list of samples for different layer. Each element is a tensor of state_shape of
        each layer
    coarse_sample_collections : list
        A list of lists containing all the samples for the coarse layers

    """
    n_layers = len(layer_list)
    layer_sample_list = []
    coarse_sample_collections = [[] for _ in range(n_layers)]
    no_parents_prob_from_coarse_layers = [[] for _ in range(n_layers)]
    no_parents_prob = torch.ones(layer_list[0].shape)
    for ll, layer in enumerate(layer_list[:-1]):
        for coarse_no_parents_prob in no_parents_prob_from_coarse_layers[ll]:
            no_parents_prob = no_parents_prob * coarse_no_parents_prob

        layer_sample = layer.draw_sample(no_parents_prob)
        no_parents_prob = layer.get_no_parents_prob(layer_sample)
        layer_sample_list.append(layer_sample)
        for coarse_layer in coarse_layer_collections[ll]:
            coarse_sample = coarse_layer.draw_sample(layer_sample)
            no_parents_prob_from_coarse_layers[coarse_layer.params['index_to_point_to']].append(
                coarse_layer.get_no_parents_prob(coarse_sample, layer_list)
            )
            coarse_sample_collections[ll].append(coarse_sample)


    for coarse_no_parents_prob in no_parents_prob_from_coarse_layers[-1]:
        no_parents_prob = no_parents_prob * coarse_no_parents_prob

    assert len(coarse_layer_collections[-1]) == 0
    layer_sample_list.append(layer_list[-1].draw_sample(no_parents_prob))
    return layer_sample_list, coarse_sample_collections
