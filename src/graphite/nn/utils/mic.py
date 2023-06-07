import torch


@torch.jit.script
def dx_mic_ortho(dx, box):
    """Minimal distance vectors in orthormbic boxes.

    References:
    - https://en.wikipedia.org/wiki/Periodic_boundary_conditions
    - https://zenodo.org/record/894522
    """
    dx -= (dx >  box/2) * box
    dx += (dx < -box/2) * box
    return dx


def dx_mic(dx, cell):
    """Minimal distance vectors in general triclinic cells.

    This implementation is inaccurate for large distances relative to the cell.
    See references:
    - https://pure.rug.nl/ws/portalfiles/portal/2839530/03_c3.pdf
    - https://gist.github.com/kain88-de/d004620a57b08e45812b7f5108a375d7
    """
    dx @= torch.linalg.pinv(cell)
    box = torch.ones(cell.size(0), device=dx.device)
    dx = dx_mic_ortho(dx, box)
    dx @= cell
    return dx
