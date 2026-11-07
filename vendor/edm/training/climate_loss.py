import torch


class ClimateEDMLoss:

    def __init__(
        self,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(
        self,
        net,
        images,
        condition,
        mask=None,
    ):

        # ---------------------------------------------
        # 1. Sample EDM noise level sigma
        # ---------------------------------------------

        rnd_normal = torch.randn(
            [images.shape[0], 1, 1, 1],
            device=images.device,
        )

        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        # ---------------------------------------------
        # 2. EDM loss weight
        # ---------------------------------------------

        weight = (
            sigma ** 2
            + self.sigma_data ** 2
        ) / (
            sigma * self.sigma_data
        ) ** 2

        # ---------------------------------------------
        # 3. Add Gaussian noise to HR
        # ---------------------------------------------

        noise = torch.randn_like(images) * sigma

        noisy_images = images + noise

        # ---------------------------------------------
        # 4. Conditional denoising
        # ---------------------------------------------

        denoised = net(
            noisy_images,
            sigma,
            condition=condition,
        )

        # ---------------------------------------------
        # 5. Weighted MSE
        # ---------------------------------------------

        loss = (weight * (denoised - images) ** 2)

        # ---------------------------------------------
        # 6. Optional land mask
        # ---------------------------------------------

        if mask is not None:
            loss = loss * mask

        return loss