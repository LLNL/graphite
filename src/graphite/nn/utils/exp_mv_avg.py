import torch


class ExponentialMovingAverage():
    """Exponential moving average for model weights during training.
    """
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    @torch.no_grad()
    def __call__(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.mu)*x + self.mu*self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average