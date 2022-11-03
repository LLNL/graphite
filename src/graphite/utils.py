import numpy as np
import pandas as pd


__all__ = [
    'mask2index',
    'index2mask',
    'np_groupby',
    'np_scatter',
    'encode_labels',
    'summary',
]


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
    """Abstraction of the `torch_scatter.scatter` function.
    See https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html
    for how `scatter` works in PyTorch.

    Args:
        src (list): The source array.
        index (list of ints): The indices of elements to scatter.
        func (function, optional): Function that operates on elements with the same indices.

    :rtype: generator
    """
    return (func(g) for g in np_groupby(src, index))


def arg_same_rows(A, B):
    """Find indices of common rows in 2D numpy arrays A and B.
    Returns two sets of indices: one w.r.t. A and one w.r.t. B.
    """
    return np.where(abs((A[:, None, :] - B)).sum(axis=2) == 0)


def encode_labels(labels):
    """Encode categorical labels into integer representation.
    For example, [4, 5, 10, 5, 4, 4] would be encoded to [0, 1, 2, 1, 0, 0],
    and ['paris', 'paris', 'tokyo', 'amsterdam'] would be encoded to [1, 1, 2, 0].
    """
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def summary(model):
    """Returns a dataframe describing the numbers of trainable parameters in a torch model.
    """
    params = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    total_num = sum(n for _, n in params)
    params.append(('Total', total_num))
    return pd.DataFrame(params, columns=['Layer', 'Params'])
