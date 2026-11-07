# Extracted verbatim from the official SatRain notebook (code cells 12, 13, 15).
# Source commit: 1b0f40676724e1bb00cc9b32f621e9e0fcc1522e (v1.2.1).
# Copyright 2024 IPWG machine-learning working group. MIT license: upstream/LICENSE.txt.

from typing import Callable, List
import torch
from torch import nn

class ResNetBlock(nn.Module):
    """
    Implments a basic ResNet block consisting of two convolutions each
    followed by a normalization and activation layer.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        normalization_layer: Callable[[int], nn.Module] = nn.BatchNorm2d
    ):
        """
        Args:
            in_channels: The number of channels in the input tensor.
            out_channels: The number of channels within the layer and in the output tensor.
        """
        super().__init__()
        padding = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.projection = None
        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate input through block.
        """
        y = self.body(x)
        if self.projection is None:
            shortcut = x
        else:
            shortcut = self.projection(x)
        return y + shortcut
      

from typing import Dict

class UNet(nn.Module):
    """
    UNet encoder-decoder architecture for precipitation retrievals.

    The model provides scalar estimates of the 'surface_precip' as well as probabilistic estimates of the
    'probability_of_precip' and 'probability_of_heavy_precip'.

    The model has the following components:
        - Stem: The stem is applied directly to the input and maps the number of input features to
              the features of the first encoder stage.
        - Encoder: Applied to the output from the stem. Consists of multiple stages and performs 2x
              downsampling at the beginning of each stage.
        - Decoder: Each stage consists of bilinear upsampling followed by a convolution block
              that merges the upsampled features with the output from the corresponding encoder
              layer.
        - Heads: A separate head for the retrieval outputs each consisting of a single ResNetBlock
              followed by a fully-connected output layer.
    """
    def __init__(
        self,
        input_features: int,
        internal_features: List[int],
        **block_kwargs
    ):
        """
        Args:
            input_features: The number of input features.
            internal_features: A list containing the number of features/channels within each stage
                of the encoder.
            block_kwargs: Keyword arguments to forward to ResNetBlock factory.
        """
        super().__init__()

        self.stem = ResNetBlock(input_features, internal_features[0])
        chans_in = internal_features[0]
        encoder_stages = []
        for n_features in internal_features:
            encoder_stages.append(nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                ResNetBlock(chans_in, n_features, **block_kwargs),
            ))
            chans_in = n_features
        self.encoder = nn.ModuleList(encoder_stages)
        
        decoder_stages = []
        for n_features in internal_features[-2::-1]:
            decoder_stages.append(nn.Sequential(
                ResNetBlock(chans_in + n_features, n_features, **block_kwargs),
            ))
            chans_in = n_features
        decoder_stages.append(ResNetBlock(n_features + n_features, n_features, **block_kwargs))
            
        self.decoder = nn.ModuleList(decoder_stages)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        
        heads = {}
        for output in ["surface_precip", "probability_of_precip", "probability_of_heavy_precip"]:
            heads[output] = nn.Sequential(
                ResNetBlock(n_features, n_features, kernel_size=1, **block_kwargs),
                nn.Conv2d(n_features, 1, kernel_size=1)
            )
        self.heads = nn.ModuleDict(heads)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Propagate input through network.

        Args:
            x: The tensor containing the input data.

        Return:
            A dictionary containing mapping the keys 'surface_precip', 'probability_of_precip',
            'probability_of_heavy_precip' to the corresponding retrieval results.
        """
        y = self.stem(x)
        shortcuts = []
        for layer in self.encoder:
            shortcuts.append(y)
            y = layer(y)

        shortcuts.reverse()
        for shortcut, layer in zip(shortcuts, self.decoder):
            y = self.upsample(y)
            y = torch.cat([y, shortcut], dim=1)
            y = layer(y)

        return {
            name: head(y) for name, head in self.heads.items()
        }
            

from typing import Any, Callable, Dict

import torch
from torch import optim
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits
import lightning as L

OUTPUTS = [
    "surface_precip",
    "probability_of_precipitation",
    "probability_of_heavy_precipitation"
]

class IPWGUNet(L.LightningModule):
    """
    Lightning module implementing a multi-layer perceptron (MLP) for retrieving precipitation from satellite
    observations.
    """
    def __init__(
        self,
        input_features: int,
        internal_features: List[int],
        n_epochs: int = 20,
        **block_kwargs
    ):
        """
        Args:
            n_input_features: The number of features in the input
            n_hidden_layers: The number of hidden layers in the MLP
            n_neurons: The number of neurons in the hidden layers
            activation_fn: A callable to create activation function layers.
            normalization_layer: A callable to create normalization layers.
            n_epochs: The numebr of epochs the model will be trained for.
        """
        super().__init__()
        self.n_epochs = n_epochs
        self.model = UNet(input_features, internal_features)

    def forward(self, retrieval_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward retrieval input through network and produce dictionary with predictions.

        Args:
            retrieval_input: The retrieval input as a single torch.Tensor.

        Return:
            A dictionary containing the predictions for 'surface_precip', 'probability_of_precipitation',
            and 'probability_of_heavy_precipitation'.
        """
        return self.model(retrieval_input)
        
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Calculates the loss-function gradients for the MLP.

        The loss is calculated as the sum of the MSE for 'surface_precip' and the binary cross-entropy loss
        for precipitation detection and heavy precipitation detection.

        Args:
            batch: A tuple containing the training data loaded from the data loader.
            batch_idx: The index of the batch in the current epoch. Not used.

        Return:
            A scalar torch.Tensor containing the total loss.
        """
        inpt, target = batch
        pred = self(inpt)
        
        surface_precip = target["surface_precip"]
        precip_mask= target["precip_mask"]
        heavy_precip_mask= target["heavy_precip_mask"]

        valid = torch.isfinite(surface_precip)
        surface_precip = surface_precip[valid]
        precip_mask = precip_mask[valid]
        heavy_precip_mask = heavy_precip_mask[valid]

        surface_precip_pred = pred["surface_precip"][:, 0][valid]
        pop = pred["probability_of_precip"][:, 0][valid]
        pohp = pred["probability_of_heavy_precip"][:, 0][valid]
        
        # MSE loss for QPE
        loss_estim = ((surface_precip_pred - surface_precip) ** 2).mean()
        # BCE loss for detection targets
        loss_detect = binary_cross_entropy_with_logits(pop, precip_mask)
        loss_detect_heavy = binary_cross_entropy_with_logits(pohp, heavy_precip_mask)
        tot_loss =  loss_estim + loss_detect + loss_detect_heavy
        return tot_loss

    def validation_step(self, batch, batch_idx) -> None:
        """
        Calculates the loss-function values on validation data.

        Args:
            batch: A tuple containing the training data loaded from the data loader.
            batch_idx: The index of the batch in the current epoch. Not used.
        """
        inpt, target = batch
        pred = self(inpt)

        surface_precip = target["surface_precip"]
        precip_mask= target["precip_mask"]
        heavy_precip_mask= target["heavy_precip_mask"]

        valid = torch.isfinite(surface_precip)
        surface_precip = surface_precip[valid]
        precip_mask = precip_mask[valid]
        heavy_precip_mask = heavy_precip_mask[valid]
        surface_precip_pred = pred["surface_precip"][:, 0][valid]
        pop = pred["probability_of_precip"][:, 0][valid]
        pohp = pred["probability_of_heavy_precip"][:, 0][valid]
        
        # MSE loss for QPE
        loss_estim = ((surface_precip_pred - surface_precip) ** 2).mean()
        # BCE loss for detection targets
        loss_detect = binary_cross_entropy_with_logits(pop, precip_mask)
        loss_detect_heavy = binary_cross_entropy_with_logits(pohp, heavy_precip_mask)
        tot_loss =  loss_estim + loss_detect + loss_detect_heavy

        opt = self.optimizers()
        learning_rate = opt.param_groups[0]['lr']
        
        self.log_dict(
            {
                "val_loss": loss_estim + loss_detect + loss_detect_heavy,
                "val_loss_estim": loss_estim,
                "val_loss_detect": loss_detect,
                "val_loss_detect_heavy": loss_detect_heavy,
                "learning_rate": learning_rate
            },
            on_epoch=True,
            prog_bar=True
        )
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        We use the Adam optimizer with a cosine annealing learning rate schedule.
        """
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
