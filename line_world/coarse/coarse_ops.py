import numpy as np


def get_coarse_stride_kernel_size(layer_list):
    """get_coarse_stride_kernel_size
    Use information from the list of layers to infer the stride and the kernel_size involved with
    a coarse layer

    Parameters
    ----------

    layer_list : list
        layer_list is a list of Layer objects, starting from the fine layer we are duplicating,
        all the way to the layer above the fine layer we are pointing to

    Returns
    -------

    stride : int
        The stride associated with the coarse layer
    kernel_size : int
        The kernel size associated with the coarse layer

    """
    stride_list = np.array([layer.params['stride'] for layer in layer_list], dtype=int)
    kernel_size_list = np.array([layer.params['kernel_size'] for layer in layer_list], dtype=int)
    stride = np.prod(stride_list)
    kernel_size = 1
    for ii in range(len(layer_list)):
        kernel_size = (kernel_size - 1) * stride_list[ii] + kernel_size_list[ii]

    return stride, kernel_size
