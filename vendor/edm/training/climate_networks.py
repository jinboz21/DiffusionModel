import torch

from training.networks import SongUNet


class ClimateEDMPrecond(torch.nn.Module):
    """
    Conditional EDM preconditioning for climate downscaling.

    Diffusion variable:
        x : noisy HR field
            shape = [B, 1, H, W]

    Condition:
        condition : upsampled LR field
            shape = [B, 1, H, W]

    U-Net input:
        concat(c_in * x, condition)
            shape = [B, 2, H, W]

    U-Net output:
        one HR channel
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
        sigma,
        condition,
        class_labels=None,
        force_fp32=False,
        **model_kwargs,
    ):

        x = x.to(torch.float32)
        condition = condition.to(torch.float32)

        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)

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

        # ---------------------------------------------
        # EDM preconditioning coefficients
        # ---------------------------------------------

        c_skip = (self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2))

        c_out = (sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt())

        c_in = (1 / (self.sigma_data ** 2 + sigma ** 2).sqrt())

        c_noise = sigma.log() / 4

        # ---------------------------------------------
        # Concatenate noisy HR and LR condition
        # ---------------------------------------------

        model_input = torch.cat([(c_in * x).to(dtype), condition.to(dtype),], dim=1,)

        # [B,2,H,W] -> [B,1,H,W]
        F_x = self.model(
            model_input,
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )

        # ---------------------------------------------
        # EDM denoised estimate
        #
        # IMPORTANT:
        # skip connection applies only to noisy HR x.
        # ---------------------------------------------

        D_x = (c_skip * x + c_out * F_x.to(torch.float32))

        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)