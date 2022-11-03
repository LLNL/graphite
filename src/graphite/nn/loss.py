import torch
from torch.nn.functional import log_softmax
import torch.autograd as autograd


def soft_cross_entropy(input, target):
    """Multi-category cross entropy loss for soft labels.
    `torch.nn.CrossEntropyLoss` in earlier PyTorch versions does not allow soft labels.
    """
    logprobs = log_softmax(input, dim=1)
    return -(target * logprobs).sum() / input.shape[0]


# Adopted from https://github.com/ermongroup/sliced_score_matching
def sliced_score_estimation(score_net, data):
    x = data.view(-1, *data.shape[1:])
    x.requires_grad = True
    v = torch.randn_like(x)
    v = v / torch.norm(v, dim=-1, keepdim=True)

    s = score_net(x)
    v_dot_s      = (v * s).sum()
    loss1        = (v * s).sum(dim=-1).pow(2).mul(0.5)
    v_dot_grad_s = autograd.grad(v_dot_s, x, create_graph=True)[0]
    loss2        = (v_dot_grad_s * v).sum(dim=-1)

    return (loss1 + loss2).mean()