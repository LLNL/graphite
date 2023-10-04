import torch


mask2index = lambda mask: torch.nonzero(mask)


def index2mask(idx_arr, n):
    mask = torch.zeros(n, dtype=torch.long, device=idx_arr.device)
    mask[idx_arr] = 1
    return mask.to(torch.bool)


def torch_groupby(arr, groups):
    sort_idx = groups.argsort()
    arr, groups = arr[sort_idx], groups[sort_idx]
    return torch.split(arr, torch.bincount(groups).tolist())