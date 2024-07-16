import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.custom_types import SamplerType, OutputDictType
from src.utils import ModuleIntDict
from src.network_blocks import Conv2DSequence, MuSigmaBlock
from src.base import AbstractPrior, AbstractPosterior, AbstractLikelihood
from utils import convert_to_onehot


# Note about indexing in this class: PHISeg has two types of levels:
# latent levels and resolution (total) levels. In this class latent
# levels are always index by l, total levels are indexed by k. All
# layers are indexed using total levels (k). All outputs in the
# forward pass are indexed using latent levels (l).
class PHISegEncoder(nn.Module):
    """Generic encoder that can be used for PhISeg prior and posterior."""

    def __init__(
        self,
        total_levels: int,
        latent_levels: int,
        zdim: int,
        input_channels: int = 3,  # changed input channels
        n0: int = 32,
    ) -> None:
        super().__init__()
        self.total_levels = total_levels
        self.latent_levels = latent_levels
        self.lk_offset = total_levels - latent_levels

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
        for k in range(1, total_levels):
            self.down_blocks[k] = Conv2DSequence(
                in_channels=num_channels[k - 1],
                out_channels=num_channels[k],
                depth=3,
            )

        # Create layers for feeding back latent variables to the hierarchy level above
        self.up_blocks = ModuleIntDict()
        for k in range(total_levels - latent_levels, total_levels - 1):
            self.up_blocks[k] = Conv2DSequence(
                in_channels=zdim,
                out_channels=n0 * zdim,
                depth=2,
            )

        # Create layers after concat and before mu sampling
        self.sample_merge_blocks = ModuleIntDict(
            {
                k: Conv2DSequence(
                    in_channels=num_channels[k] + n0 * zdim,
                    out_channels=num_channels[k],
                    depth=2,
                )
                for k in range(total_levels - latent_levels, total_levels)
            }
        )

        # Create all mu and sigma prediction layers
        self.mu_sigma = ModuleIntDict(
            {
                k: MuSigmaBlock(in_channels=num_channels[k], zdim=zdim)
                for k in range(total_levels - latent_levels, total_levels)
            }
        )

    def forward(
        self,
        x: torch.Tensor,
        sampler: SamplerType,
        override_samples: Optional[OutputDictType] = None,
    ) -> tuple[OutputDictType, OutputDictType, OutputDictType]:
        # Going all the way down on the encoding path
        down_activations = {0: self.down_blocks[0](x)}
        for k in range(1, self.total_levels):
            down_sampled = self.downsample(down_activations[k - 1])
            down_activations[k] = self.down_blocks[k](down_sampled)

        # Going back up (switching to indexing by latent level)
        mus, sigmas = {}, {}
        samples: OutputDictType = {}  # Explicit type definition to make mypy happy
        for l in reversed(range(self.latent_levels)):
            k = l + self.lk_offset
            # on the lowest level
            if l == self.latent_levels - 1:
                mus[l], sigmas[l] = self.mu_sigma[k](down_activations[k])
            # on all other levels
            else:
                if override_samples is not None:
                    sample_upsampled = F.interpolate(
                        override_samples[l + 1],
                        mode="bilinear",
                        scale_factor=2,
                        align_corners=True,
                    )
                else:
                    sample_upsampled = F.interpolate(
                        samples[l + 1],
                        mode="bilinear",
                        scale_factor=2,
                        align_corners=True,
                    )
                intermediate = self.up_blocks[k](sample_upsampled)
                intermediate = torch.cat([intermediate, down_activations[k]], dim=1)
                intermediate = self.sample_merge_blocks[k](intermediate)
                mus[l], sigmas[l] = self.mu_sigma[k](intermediate)

            # Generate samples
            samples[l] = sampler(mus[l], sigmas[l])

        return mus, sigmas, samples


class PHISegPrior(AbstractPrior):
    def __init__(
        self,
        sampler: SamplerType,
        total_levels: int,
        latent_levels: int,
        zdim: int,
        n0: int = 32,
        input_channels: int = 3,
    ) -> None:
        super().__init__()

        self.sampler: SamplerType = sampler
        self.encoder = PHISegEncoder(
            total_levels=total_levels,
            latent_levels=latent_levels,
            zdim=zdim,
            input_channels=input_channels,  # changed input channels
            n0=n0,
        )

    def forward(
        self,
        x: torch.Tensor,
        posterior_samples: Optional[OutputDictType] = None,
    ) -> tuple[OutputDictType, OutputDictType, OutputDictType]:
        # While training use posterior samples, when generating set to None
        # which will cause prior samples to be used
        return self.encoder(x, sampler=self.sampler, override_samples=posterior_samples)


class PHISegPosterior(AbstractPosterior):
    def __init__(
        self,
        sampler: SamplerType,
        total_levels: int,
        latent_levels: int,
        zdim: int,
        num_classes: int,
        n0: int = 32,
        input_channels: int = 3,
    ) -> None:
        super().__init__()

        self.sampler: SamplerType = sampler
        self.encoder = PHISegEncoder(
            total_levels=total_levels,
            latent_levels=latent_levels,
            zdim=zdim,
            input_channels=num_classes + input_channels,  # changed from 1 to 3 as rgb
            n0=n0,
        )
        self.num_classes = num_classes

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[OutputDictType, OutputDictType, OutputDictType]:
        y_onehot = convert_to_onehot(
            y.to(torch.float32),
            num_classes=self.num_classes,
        )
        xy = torch.cat([x, torch.add(y_onehot, -0.5)], dim=1)
        return self.encoder(xy, sampler=self.sampler, override_samples=None)


class PHISegLikelihood(AbstractLikelihood):
    def __init__(
        self,
        total_levels: int,
        latent_levels: int,
        zdim: int,
        num_classes: int,
        n0: int = 32,
    ) -> None:
        super().__init__()

        self.total_levels = total_levels
        self.latent_levels = latent_levels
        self.lk_offset = total_levels - latent_levels
        self.num_classes = num_classes

        # Use the same rule as for the encoder
        num_channels = {
            k: n0 * v for k, v in enumerate([1, 2, 4] + [6] * (total_levels - 3))
        }

        # Create layers post processing the samples
        self.post_sample_blocks = ModuleIntDict()
        for l in range(latent_levels):
            self.post_sample_blocks[l] = nn.Sequential(
                Conv2DSequence(
                    in_channels=zdim,
                    out_channels=num_channels[l + self.lk_offset],
                    depth=2,
                ),
                self._make_increase_resolution_block(
                    times=self.lk_offset,
                    in_channels=num_channels[l + self.lk_offset],
                    out_channels=num_channels[l],
                ),  # l above because after upsampling it will be lk_offset levels higher
            )

        # Create post concat convs
        # Note: Since, resolution has been increase by lk_offset we now should index
        # num_channels directly with 'l' instead of 'l+self.lk_offset' because l and
        # k now refer to the same resolution level.
        self.post_concat_blocks = ModuleIntDict()
        for l in range(latent_levels - 1):
            in_channels = num_channels[l] + num_channels[l + 1]
            self.post_concat_blocks[l] = Conv2DSequence(
                in_channels=in_channels,
                out_channels=num_channels[l],
                depth=2,
            )

        # Create final prediction layers
        self.final_prediction_layers = ModuleIntDict()
        for l in range(latent_levels):
            self.final_prediction_layers[l] = nn.Conv2d(
                in_channels=num_channels[l],
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

    def forward(self, samples: OutputDictType, x=None) -> OutputDictType:  # type: ignore[override]
        # Input x is just a dummy here. TODO: Remove this when refactoring

        # Path converting the samples to activations scaled up by (total_levels - latent_levels)
        # such that the highest latent level output has the original image resolution
        post_sample_activations = {
            l: self.post_sample_blocks[l](samples[l]) for l in range(self.latent_levels)
        }

        # Upsample and merge the lower resolution levels with the higher levels
        post_concat_activations = {
            self.latent_levels - 1: post_sample_activations[self.latent_levels - 1]
        }
        for l in reversed(range(1, self.latent_levels)):
            upsampled = F.interpolate(
                post_concat_activations[l],
                mode="bilinear",
                scale_factor=2,
                align_corners=True,
            )
            concat = torch.cat(
                [upsampled, post_sample_activations[l - 1]], dim=1
            )  # Concatenate with the level above
            post_concat_activations[l - 1] = self.post_concat_blocks[l - 1](concat)

        # Final 1x1 Convs to get segmention masks at each level
        seg_logits = {
            l: self.final_prediction_layers[l](post_concat_activations[l])
            for l in range(self.latent_levels)
        }

        # Make sure everything is the right size before resizing everything to input dimensions
        for l in range(self.latent_levels):
            assert [*seg_logits[l].shape[1:4]] == [
                self.num_classes,
                samples[self.latent_levels - 1].shape[2]
                * 2 ** (self.latent_levels - l + self.lk_offset - 1),
                samples[self.latent_levels - 1].shape[3]
                * 2 ** (self.latent_levels - l + self.lk_offset - 1),
            ], f"Shape mismatch in likelhood network at level {l}."

        # Resize to final size
        seg_logits_resized = {0: seg_logits[0]}
        for l in range(1, self.latent_levels):
            seg_logits_resized[l] = torch.nn.functional.interpolate(
                seg_logits[l], scale_factor=2**l, mode="nearest"
            )

        return seg_logits_resized
