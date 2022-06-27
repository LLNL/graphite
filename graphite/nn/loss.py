from torch.nn.functional import log_softmax


def soft_cross_entropy(input, target):
    """Multi-category cross entropy loss for soft labels.
    `torch.nn.CrossEntropyLoss` in earlier PyTorch versions does not allow soft labels.
    """
    logprobs = log_softmax(input, dim=1)
    return -(target * logprobs).sum() / input.shape[0]
