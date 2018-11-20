import torch
import line_world.core.layer_ops as lo
import matplotlib.pyplot as plt


def visualize_states(state_list, coarse_state_collections, layer_list, coarse_layer_collections):
    fig_size = (12, 10)
    on_bricks_prob_list = [lo.get_on_bricks_prob(state) for state in state_list]
    coarse_on_bricks_prob_collections = []
    for coarse_state_list in coarse_state_collections:
        coarse_on_bricks_prob_collections.append([])
        for coarse_state in coarse_state_list:
            coarse_on_bricks_prob_collections[-1].append(lo.get_on_bricks_prob(coarse_state))

    print('Middle layer on prob: {}'.format(on_bricks_prob_list[1]))
    # Visualize the top layer
    top_max_prob, top_max_indices = torch.topk(torch.softmax(state_list[0], dim=3), 4)
    top_max_prob = top_max_prob.view(-1)
    top_max_indices = top_max_indices.view(-1)
    top_expanded_templates = layer_list[0].expanded_templates.to_dense()
    fig, ax = plt.subplots(2, 2, figsize=fig_size)
    for ii in range(2):
        for jj in range(2):
            ind = ii * 2 + jj
            ax[ii, jj].imshow(
                top_expanded_templates[0, 0, 0, top_max_indices[ind], 0].numpy(), cmap='gray'
            )
            ax[ii, jj].set_title('Index: {}, Prob {:.2e}'.format(top_max_indices[ind], top_max_prob[ind]))

    fig.suptitle('Top layer (on prob {}) important templates'.format(on_bricks_prob_list[0][0, 0, 0]))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Visualize the middle layer
    mid_max_prob, mid_max_location = torch.topk(on_bricks_prob_list[1].view(-1), 9)
    _, n_rows, n_cols = on_bricks_prob_list[1].shape
    mid_max_location = [torch.tensor([ind / n_cols, ind % n_cols], dtype=torch.long) for ind in mid_max_location]
    mid_max_indices = []
    for location in mid_max_location:
        mid_max_indices.append(torch.argmax(torch.softmax(state_list[1], dim=-1)[0][location[0], location[1]]))

    mid_expanded_templates = layer_list[1].expanded_templates.to_dense()
    fig, ax = plt.subplots(3, 3, figsize=fig_size)
    for ii in range(3):
        for jj in range(3):
            ind = ii * 3 + jj
            location = mid_max_location[ind]
            ax[ii, jj].imshow(
                mid_expanded_templates[0, location[0], location[1], mid_max_indices[ind], 0].numpy(), cmap='gray'
            )
            ax[ii, jj].set_title('Location: {}, Prob: {:.2e}'.format(location, mid_max_prob[ind]))

    fig.suptitle('Middle layer, templates at important locations')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Visualize the coarse layer
    coarse_max_prob, coarse_max_indices = torch.topk(torch.softmax(coarse_state_collections[0][0], dim=3), 4)
    coarse_max_prob = coarse_max_prob.view(-1)
    coarse_max_indices = coarse_max_indices.view(-1)
    coarse_expanded_templates = coarse_layer_collections[0][0].expanded_templates.to_dense()
    fig, ax = plt.subplots(2, 2, figsize=fig_size)
    for ii in range(2):
        for jj in range(2):
            ind = ii * 2 + jj
            ax[ii, jj].imshow(
                coarse_expanded_templates[0, 0, 0, coarse_max_indices[ind], 0].numpy(), cmap='gray'
            )
            ax[ii, jj].set_title('Index: {}, Prob {:.2e}'.format(coarse_max_indices[ind], coarse_max_prob[ind]))

    fig.suptitle('coarse layer (on prob {}) important templates'.format(coarse_on_bricks_prob_collections[0][0][0, 0, 0]))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def visualize_gradients(state_list, coarse_state_collections, layer_list, coarse_layer_collections):
    fig_size = (12, 10)
    grad_list = [torch.abs(state.grad.data) for state in state_list[:-1]]
    coarse_grad_collections = [[torch.abs(coarse_state_collections[0][0].grad.data)], [], []]
    total_grad_list = [torch.sum(grad, dim=-1) for grad in grad_list]
    total_on_grad_list = [torch.sum(grad[:, :, :, 1:], dim=-1) for grad in grad_list]
    coarse_total_grad_collections = []
    coarse_total_on_grad_collections = []
    for coarse_grad_list in coarse_grad_collections:
        coarse_total_grad_collections.append([])
        coarse_total_on_grad_collections.append([])
        for coarse_grad in coarse_grad_list:
            coarse_total_grad_collections[-1].append(torch.sum(coarse_grad, dim=-1))
            coarse_total_on_grad_collections[-1].append(torch.sum(coarse_grad[:, :, :, 1:], dim=-1))

    print('Top layer total grad: {}'.format(total_grad_list[0]))
    print('Top layer total on grad: {}'.format(total_on_grad_list[0]))
    print('Middle layer total grad: {}'.format(total_grad_list[1]))
    print('Middle layer total on grad: {}'.format(total_on_grad_list[1]))
    print('Coarse layer total grad: {}'.format(coarse_total_grad_collections[0][0]))
    print('Coarse layer total on grad: {}'.format(coarse_total_on_grad_collections[0][0]))
    # Visualize the top layer
    top_max_grad, top_max_indices = torch.topk(grad_list[0], 4)
    top_max_grad = top_max_grad.view(-1)
    top_max_indices = top_max_indices.view(-1)
    top_expanded_templates = layer_list[0].expanded_templates.to_dense()
    fig, ax = plt.subplots(2, 2, figsize=fig_size)
    for ii in range(2):
        for jj in range(2):
            ind = ii * 2 + jj
            ax[ii, jj].imshow(
                top_expanded_templates[0, 0, 0, top_max_indices[ind], 0].numpy(), cmap='gray'
            )
            ax[ii, jj].set_title('Index: {}, Grad: {:.2e}'.format(top_max_indices[ind], top_max_grad[ind]))

    fig.suptitle('Top layer (total grad {:2e}) important templates'.format(total_grad_list[0][0, 0, 0]))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Visualize the middle layer
    mid_max_grad, mid_max_location = torch.topk(total_grad_list[1].view(-1), 9)
    _, n_rows, n_cols = total_grad_list[1].shape
    mid_max_location = [torch.tensor([ind / n_cols, ind % n_cols], dtype=torch.long) for ind in mid_max_location]
    mid_max_indices = []
    for location in mid_max_location:
        mid_max_indices.append(torch.argmax(grad_list[1][0][location[0], location[1]]))

    mid_expanded_templates = layer_list[1].expanded_templates.to_dense()
    fig, ax = plt.subplots(3, 3, figsize=fig_size)
    for ii in range(3):
        for jj in range(3):
            ind = ii * 3 + jj
            location = mid_max_location[ind]
            ax[ii, jj].imshow(
                mid_expanded_templates[0, location[0], location[1], mid_max_indices[ind], 0].numpy(), cmap='gray'
            )
            ax[ii, jj].set_title('Location: {}, Grad: {:.2e}'.format(location, mid_max_grad[ind]))

    fig.suptitle('Middle layer, templates at important locations')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Visualize the coarse layer
    coarse_max_grad, coarse_max_indices = torch.topk(coarse_grad_collections[0][0], 4)
    coarse_max_grad = coarse_max_grad.view(-1)
    coarse_max_indices = coarse_max_indices.view(-1)
    coarse_expanded_templates = coarse_layer_collections[0][0].expanded_templates.to_dense()
    fig, ax = plt.subplots(2, 2, figsize=fig_size)
    for ii in range(2):
        for jj in range(2):
            ind = ii * 2 + jj
            ax[ii, jj].imshow(
                coarse_expanded_templates[0, 0, 0, coarse_max_indices[ind], 0].numpy(), cmap='gray'
            )
            ax[ii, jj].set_title('Index: {}, Prob {:.2e}'.format(coarse_max_indices[ind], coarse_max_grad[ind]))

    fig.suptitle('coarse layer (total grad {}) important templates'.format(coarse_total_grad_collections[0][0][0, 0, 0]))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
