"""Shared conditional EDM model and sampler for ERA5-Land downscaling."""

import torch

from training.climate_networks import ClimateEDMPrecond


MODEL_CONFIG = {
    "img_resolution": 256,
    "img_channels": 1,
    "condition_channels": 1,
    "model_type": "SongUNet",
    "model_channels": 32,
    "channel_mult": [1, 2, 2, 2],
    "num_blocks": 1,
    "attn_resolutions": [],
    "dropout": 0.0,
    "embedding_type": "positional",
    "channel_mult_noise": 1,
    "encoder_type": "standard",
    "decoder_type": "standard",
    "resample_filter": [1, 1],
    "sigma_data": 0.5,
}


def make_model(device, config=None):
    return ClimateEDMPrecond(**(config or MODEL_CONFIG)).to(device)


@torch.no_grad()
#从随机噪声出发，以lr_up为condition，多次调用已经训练好的模型，逐步生成 HR
def edm_sampler(net, condition, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7):
    """EDM deterministic Euler/Heun sampler conditioned on upsampled LR."""
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    device = condition.device
    indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    steps = (sigma_max ** (1 / rho) + indices / (num_steps - 1)
             * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    steps = torch.cat([net.round_sigma(steps), torch.zeros_like(steps[:1])])
    x = torch.randn(condition.shape, device=device, dtype=torch.float64) * steps[0]
    for i, (current, following) in enumerate(zip(steps[:-1], steps[1:])):
        denoised = net(x, current, condition=condition).to(torch.float64)
        derivative = (x - denoised) / current
        next_x = x + (following - current) * derivative
        if i < num_steps - 1:
            denoised_next = net(next_x, following, condition=condition).to(torch.float64)
            next_derivative = (next_x - denoised_next) / following
            next_x = x + (following - current) * (derivative + next_derivative) / 2
        x = next_x
    return x.float()
