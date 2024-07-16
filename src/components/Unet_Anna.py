import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.custom_types import SamplerType, OutputDictType
from src.utils import ModuleIntDict
from src.network_blocks import Conv2DSequence, MuSigmaBlock
from src.base import AbstractPrior, AbstractPosterior, AbstractLikelihood
from utils import convert_to_onehot


class UnetEncoder(nn.Module):
    """Generic encoder that can be used for PhISeg prior and posterior."""

    def __init__(
        self,
        total_levels: int,
        input_channels: int = 3,  # changed input channels
        n0: int = 32,
    ) -> None:
        super().__init__()
        self.total_levels = total_levels

        # increase num_channels until the 4th level, then use n0*6 channels
        num_channels = {
            k: n0 * v for k, v in enumerate([1, 2, 4] + [6] * (total_levels - 3))
        }

        # Create upsampling and downsampling layers (those can be reused)
        self.downsample = nn.AvgPool2d(
            kernel_size=2, stride=2, padding=0, ceil_mode=True
        )

        # Create layers of main downstream path
        self.down_blocks = ModuleIntDict(
            {
                0: Conv2DSequence(
                    in_channels=input_channels, out_channels=num_channels[0], depth=3
                )
            }
        )
        for k in range(0, total_levels - 1):
            self.down_blocks[k + 1] = Conv2DSequence(
                in_channels=num_channels[k],
                out_channels=num_channels[k + 1],
                depth=3,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Going all the way down on the encoding path
        down_activations = {0: self.down_blocks[0](x)}
        for k in range(1, self.total_levels):
            down_sampled = self.downsample(down_activations[k - 1])
            down_activations[k] = self.down_blocks[k](down_sampled)
        return down_activations[self.total_levels - 1]


class UnetDecoder(AbstractLikelihood):
    def __init__(
        self,
        total_levels: int,
        num_classes: int,
        n0: int = 32,
    ) -> None:
        super().__init__()

        self.total_levels = total_levels
        self.num_classes = num_classes

        # Use the same rule as for the encoder
        num_channels = {
            k: n0 * v for k, v in enumerate([1, 2, 4] + [6] * (total_levels - 3))
        }

        # Create up layers
        channel_idx = list(reversed(range(len(num_channels))))
        self.up_blocks = ModuleIntDict()
        for k in range(total_levels - 1):
            self.up_blocks[k] = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                Conv2DSequence(
                    in_channels=num_channels[channel_idx[k]],
                    out_channels=num_channels[channel_idx[k + 1]],
                    depth=2,
                ),
            )

        # Create final prediction layers
        self.final_prediction_layer = ModuleIntDict()
        self.final_prediction_layer = nn.Conv2d(
            in_channels=num_channels[0],
            out_channels=num_classes,
            kernel_size=1,
        )

    def _make_increase_resolution_block(
        self, times: int, in_channels: int, out_channels: int
    ):
        module_list: list[nn.Module] = []
        for i in range(times):
            module_list.append(
                nn.Upsample(mode="bilinear", scale_factor=2, align_corners=True)
            )
            if i != 0:
                in_channels = out_channels
            module_list.append(
                Conv2DSequence(
                    in_channels=in_channels, out_channels=out_channels, depth=1
                )
            )
        return nn.Sequential(*module_list)

    def forward(self, x: OutputDictType) -> OutputDictType:  # type: ignore[override]
        # Going all the way up on the decoding path
        up_activations = {0: self.up_blocks[0](x)}
        for k in range(1, self.total_levels - 1):
            up_activations[k] = self.up_blocks[k](up_activations[k - 1])

        prediction = self.final_prediction_layer(up_activations[self.total_levels - 2])

        return prediction
