import numpy as np
import pandas as pd


mask2index = lambda mask: np.flatnonzero(mask)


def index2mask(idx_arr, n):
    mask = np.zeros(n, dtype=int)
    mask[idx_arr] = 1
    return mask.astype(np.bool)


def np_groupby(arr, groups):
    """Numpy implementation of `groupby` operation (a common method in pandas).
    """
    arr, groups = np.array(arr), np.array(groups)
    sort_idx = groups.argsort()
    arr = arr[sort_idx]
    groups = groups[sort_idx]
    return np.split(arr, np.unique(groups, return_index=True)[1])[1:]


def np_scatter(src, index, func):
    """Generalization of the `torch_scatter.scatter` operation for any reduce function.
    See https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html for how `scatter` works.

    Args:
        src (array): The source array.
        index (array of int): The indices of elements to scatter.
        func (function): Reduce function (e.g., mean, sum) that operates on elements with the same indices.

    :rtype: generator
    """
    return (func(g) for g in np_groupby(src, index))


def summary(model):
    """Returns a dataframe describing the numbers of trainable parameters in a torch model.
    """
    params = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    total_num = sum(n for _, n in params)
    params.append(('Total', total_num))
    return pd.DataFrame(params, columns=['Layer', 'Params'])
