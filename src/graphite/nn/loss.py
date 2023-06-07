import torch
import torch.nn.functional as F
import torch.autograd as autograd


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


def jensen_shannon(input, target, reduction='batchmean', eps=1e-5):
    """Compute the square root of the Jensen-Shannon divergence.
    `input` and `target` are assumed to be BxD tensors, where B is the batch dimension,
    and D is the feature dimension.
    """
    P, Q = input+eps, target+eps
    P, Q = P/P.sum(-1), Q/Q.sum(-1)
    M = ((P+Q)/2).log()
    loss = 0.5 * F.kl_div(M, P, reduction=reduction) \
         + 0.5 * F.kl_div(M, Q, reduction=reduction)
    return loss.sqrt()
