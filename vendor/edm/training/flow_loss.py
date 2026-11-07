import torch


class FlowEDMLoss:
    """Conditional flow matching along a Gaussian-to-data path."""

    def __init__(self, sigma_min=0.01):
        if not 0 <= sigma_min < 1:
            raise ValueError("sigma_min must be in [0, 1)")
        self.sigma_min = sigma_min

    def __call__(
        self,
        net,
        images,
        t,
        condition,
        mask=None,
    ):

        if t.ndim != 1 or t.shape[0] != images.shape[0]:
            raise ValueError("t must have one value per image")

        x0 = torch.randn_like(images)
        time = t.reshape(-1, *([1] * (images.ndim - 1)))
        x_t = (1 - (1 - self.sigma_min) * time) * x0 + time * images
        u_target = images - (1 - self.sigma_min) * x0
        vt = net(x_t, t, condition=condition)

        loss = (vt - u_target) ** 2

        if mask is not None:
            loss = loss * mask

        return loss
