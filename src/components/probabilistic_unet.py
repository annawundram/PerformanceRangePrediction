import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F

from src.network_blocks import Conv2DSequence, MuSigmaBlock
from src.utils import ModuleIntDict
from src.custom_types import SamplerType, OutputDictType
from src.base import AbstractPrior, AbstractPosterior, AbstractLikelihood
from src.components.unet import UNetEncoder, UNetBase
from utils import convert_to_onehot


class ProbUNetPrior(AbstractPrior):
    def __init__(
        self,
        sampler: SamplerType,
        total_levels: int,
        zdim: int,
        n0: int = 32,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        self.total_levels = total_levels

        self.sampler: SamplerType = sampler
        self.encoder = UNetEncoder(
            total_levels=total_levels,
            input_channels=input_channels,
            n0=n0,
        )

        self.mu_sigma = MuSigmaBlock(
            in_channels=self.encoder.num_channels[total_levels - 1], zdim=zdim
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[OutputDictType, OutputDictType, OutputDictType]:
        # Get encoder features
        features = self.encoder(x)[self.total_levels - 1]

        # Global average pooling
        # NOTE: This returns a tensor with the two spatial dimensions equal to 1.
        # No need to squash since that is what we need for tiling again in the
        # likelihood network anyway.
        features_pooled = F.avg_pool2d(features, kernel_size=features.shape[2:])

        mu, sigma = self.mu_sigma(features_pooled)
        samples = {0: self.sampler(mu, sigma)}

        # mu and sigma as dict to be consistent with PHiSeg
        mu, sigma = {0: mu}, {0: sigma}

        return mu, sigma, samples


class ProbUNetPosterior(AbstractPosterior):
    def __init__(
        self,
        sampler: SamplerType,
        total_levels: int,
        zdim: int,
        num_classes: int,
        n0: int = 32,
        input_channels: int = 3,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.total_levels = total_levels
        self.sampler: SamplerType = sampler
        self.encoder = UNetEncoder(
            total_levels=total_levels,
            input_channels=num_classes + input_channels,
            n0=n0,
        )

        self.mu_sigma = MuSigmaBlock(
            in_channels=self.encoder.num_channels[total_levels - 1], zdim=zdim
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[OutputDictType, OutputDictType, OutputDictType]:
        y_onehot = convert_to_onehot(
            y.to(torch.float32),
            num_classes=self.num_classes,
        )
        xy = torch.cat([x, torch.add(y_onehot, -0.5)], dim=1)

        # Get encoder features
        features = self.encoder(xy)[self.total_levels - 1]

        # Global average pooling
        features_pooled = F.avg_pool2d(features, kernel_size=features.shape[2:])

        mu, sigma = self.mu_sigma(features_pooled)
        samples = {0: self.sampler(mu, sigma)}

        # mu and sigma as dict to be consistent with PHiSeg
        mu, sigma = {0: mu}, {0: sigma}

        return mu, sigma, samples


class ProbUNetLikelihood(AbstractLikelihood):
    def __init__(
        self,
        total_levels: int,
        zdim: int,
        num_classes: int,
        n0: int = 32,
    ) -> None:
        super().__init__()

        self.total_levels = total_levels
        self.num_classes = num_classes
        self.zdim = zdim

        self.unet = UNetBase(total_levels=total_levels, n0=n0)

        self.fcomb = Conv2DSequence(
            in_channels=n0 + zdim,
            out_channels=n0,
            depth=3,
            kernel_size=1,
        )

        self.final_prediction_layer = nn.Conv2d(
            in_channels=n0,
            out_channels=num_classes,
            kernel_size=1,
        )

    def forward(self, z, x) -> OutputDictType:  # type: ignore[override]
        # NOTE: The Prob. U-Net could be made more efficient by taking a list of
        # z's here, and outputing a list of samples, so we don't need to do the
        # whole forward pass everytime. However, then it would not be consistent
        # with the interfaces of PHiSeg. Leaving like this for now.

        z = z[0]

        assert z.shape[1] == self.zdim, f"z.shape = {z.shape}; self.zdim = {self.zdim}"

        unet_features = self.unet(x)
        bs, _, sx, sy = x.shape

        z = z.view(bs, self.zdim, 1, 1)
        z = torch.tile(z, (1, 1, sx, sy))

        combined = torch.cat([unet_features, z], axis=1)
        fcomb = self.fcomb(combined)
        seg_logits = self.final_prediction_layer(fcomb)

        # Returning dictionary with one level to be consistent with PHiSeg
        return {0: seg_logits}


if __name__ == "__main__":
    latent_dim = 10

    likelihood_net = ProbUNetLikelihood(
        total_levels=7, zdim=latent_dim, num_classes=4, n0=32
    )

    inp = torch.randn(4, 1, 256, 256)
    z = torch.randn(4, latent_dim)

    out = likelihood_net(inp, z)
