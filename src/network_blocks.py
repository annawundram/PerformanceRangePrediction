import torch
import torch.nn as nn
import torch.nn.functional as F


def gauss_sampler(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return mu + sigma * torch.randn_like(sigma, dtype=torch.float32)


class TemperatureScaling(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)

    def forward(self, logits):
        return logits / self.temperature


class Conv2DUnit(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int = None, kernel_size: int = 3
    ) -> None:
        super().__init__()

        if not out_channels:
            out_channels = in_channels

        if kernel_size == 1:
            padding = 0
        elif kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 3
        else:
            raise ValueError(
                f"Only kernel sizes 1, 3, and 5 are supported. You tried to use {kernel_size}."
            )

        self._op = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        return self._op(x)


class Conv2DSequence(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        first_conv = Conv2DUnit(in_channels, out_channels, kernel_size=kernel_size)
        convs = [first_conv] + [Conv2DUnit(out_channels) for _ in range(depth - 1)]
        self._op = nn.Sequential(*convs)

    def forward(self, x: torch.Tensor):
        return self._op(x)


class Conv2DSequenceWithPermanentDropout(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        kernel_size: int = 3,
        dropout_prob: int = 0.5,
    ) -> None:
        super().__init__()
        ops = [
            Conv2DUnit(in_channels, out_channels, kernel_size=kernel_size),
            PermanentDropout(dropout_prob=dropout_prob),
        ]
        for _ in range(depth - 1):
            ops.append(Conv2DUnit(out_channels))
            ops.append(PermanentDropout(dropout_prob=dropout_prob))
        self._op = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor):
        return self._op(x)


class MuSigmaBlock(nn.Module):
    def __init__(self, in_channels: int, zdim: int) -> None:
        super().__init__()
        self._conv_mu = nn.Conv2d(in_channels, zdim, kernel_size=1)
        self._conv_sigma = nn.Sequential(
            nn.Conv2d(in_channels, zdim, kernel_size=1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor):
        return [self._conv_mu(x), self._conv_sigma(x)]


class PermanentDropout(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(PermanentDropout, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, x):
        if self.dropout_prob == 0:
            return x
        else:
            # Apply dropout during both training and evaluation
            return F.dropout(x, p=self.dropout_prob, training=True)
