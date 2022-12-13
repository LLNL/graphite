import torch


mask2index = lambda mask: torch.nonzero(mask)


def index2mask(idx_arr, n):
    mask = torch.zeros(n, dtype=torch.long, device=idx_arr.device)
    mask[idx_arr] = 1
    return mask.to(torch.bool)


def arg_same_rows(A, B):
    """Find indices of common rows in 2D numpy arrays A and B.
    Returns two sets of indices: one w.r.t. A and one w.r.t. B.
    """
    return torch.where((A[:, None, :] - B).abs().sum(dim=2) == 0)