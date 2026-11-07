"""Conditional flow matching model and sampler for ERA5-Land downscaling."""

import torch

from training.flow_networks import FlowEDMPrecond


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
    return FlowEDMPrecond(**(config or MODEL_CONFIG)).to(device)

@torch.no_grad()
def flow_sampler(net, condition, num_steps=20):
    if num_steps < 1:
        raise ValueError("num_steps must be positive")

    device = condition.device
    
    condition = condition.to(torch.float32)

    # Initial Gaussian noise
    x = torch.randn_like(condition)

    # Uniform time grid: t = 0 -> 1
    time_steps = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=torch.float32)

    net.eval()

    for i in range(num_steps):

        t = time_steps[i]
        t_next = time_steps[i + 1]
        dt = t_next - t

        # Time tensors for the entire batch
        t_batch = torch.full(
            (x.shape[0],),
            t.item(),
            device=device,
            dtype=torch.float32,
        )

        t_next_batch = torch.full(
            (x.shape[0],),
            t_next.item(),
            device=device,
            dtype=torch.float32,
        )

        # Step 1: Evaluate velocity at (t, x)
        k1 = net(
            x,
            t_batch,
            condition=condition)

        # Step 2: Euler predictor
        x_pred = x + dt * k1

        # Step 3: Evaluate velocity at predicted state
        k2 = net(
            x_pred,
            t_next_batch,
            condition=condition)

        # Step 4: Heun corrector
        x = x + 0.5 * dt * (k1 + k2)

    return x
