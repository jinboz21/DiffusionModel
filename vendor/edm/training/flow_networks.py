import torch

from training.networks import SongUNet


class FlowEDMPrecond(torch.nn.Module):
    """
    Conditional velocity network for flow matching climate downscaling.

    Diffusion variable:
        x : interpolated field x_t
            shape = [B, 1, H, W]

    Condition:
        condition : upsampled LR field
            shape = [B, 1, H, W]

    U-Net input:
        concat(x_t, condition)
            shape = [B, 2, H, W]

    U-Net output:
        one velocity channel
            shape = [B, 1, H, W]
    """

    def __init__(
        self,
        img_resolution=256,
        img_channels=1,
        condition_channels=1,
        label_dim=0,
        use_fp16=False,
        sigma_min=0,
        sigma_max=float("inf"),
        sigma_data=0.5,
        model_type="SongUNet",
        **model_kwargs,
    ):
        super().__init__()

        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.condition_channels = condition_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        if model_type != "SongUNet":
            raise NotImplementedError(
                "First prototype only supports SongUNet."
            )

        self.model = SongUNet(
            img_resolution=img_resolution,

            # noisy HR + LR condition
            in_channels=img_channels + condition_channels,

            # predict only HR residual/output
            out_channels=img_channels,

            label_dim=label_dim,

            **model_kwargs,
        )

    def forward(
        self,
        x,
        t,
        condition,
        class_labels=None,
        force_fp32=False,
        **model_kwargs,
    ):

        x = x.to(torch.float32)
        condition = condition.to(torch.float32)

        if self.label_dim == 0:
            class_labels = None
        else:
            if class_labels is None:
                class_labels = torch.zeros(
                    [x.shape[0], self.label_dim],
                    device=x.device,
                )
            else:
                class_labels = class_labels.to(
                    torch.float32
                ).reshape(
                    -1,
                    self.label_dim,
                )

        # We are currently running CPU,
        # so this will use float32.
        dtype = (
            torch.float16
            if (
                self.use_fp16
                and not force_fp32
                and x.device.type == "cuda"
            )
            else torch.float32
        )

        model_input = torch.cat([(x).to(dtype), condition.to(dtype),], dim=1,)

        time_embedding_input = t * 1000.0

        velocity = self.model(
            model_input,
            time_embedding_input,
            class_labels=None,
        )

        D_x = velocity

        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
