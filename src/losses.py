import torch
import torch.nn as nn
from typing import Union
from src.custom_types import OutputDictType, ReconstructionLossType, KLDivergenceType


def KL_two_gauss_with_diag_cov(
    mu0: torch.Tensor,
    sigma0: torch.Tensor,
    mu1: torch.Tensor,
    sigma1: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Returns KL[p0 || p1]"""

    sigma0_fs = torch.square(torch.flatten(sigma0, start_dim=1))
    sigma1_fs = torch.square(torch.flatten(sigma1, start_dim=1))

    logsigma0_fs = torch.log(sigma0_fs + eps)
    logsigma1_fs = torch.log(sigma1_fs + eps)

    mu0_f = torch.flatten(mu0, start_dim=1)
    mu1_f = torch.flatten(mu1, start_dim=1)

    return torch.mean(
        0.5
        * torch.sum(
            torch.div(
                sigma0_fs + torch.square(mu1_f - mu0_f),
                sigma1_fs + eps,
            )
            + logsigma1_fs
            - logsigma0_fs
            - 1,
            dim=1,
        )
    )


def multinoulli_loss_2d(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # TODO: Change this loss to general ND later by passing one-hot GT in
    # training and flattening input and target here.

    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    return torch.mean(
        torch.sum(criterion(input=input, target=target.long()), dim=(1, 2))
    )


class HierarchicalKLLoss(nn.Module):
    def __init__(
        self,
        KL_divergence: KLDivergenceType,
        weight_dict: dict[int, float],
    ) -> None:
        super().__init__()

        self.weight_dict = weight_dict
        self.KL_divergence = KL_divergence

    def forward(
        self,
        prior_mus: OutputDictType,
        prior_sigmas: OutputDictType,
        posterior_mus: OutputDictType,
        posterior_sigmas: OutputDictType,
        return_all_levels: bool = False,
    ):
        assert self.weight_dict.keys() == prior_mus.keys()
        assert (
            prior_mus.keys()
            == prior_sigmas.keys()
            == posterior_mus.keys()
            == posterior_sigmas.keys()
        )

        kl_loss = 0.0
        all_levels = {}
        for l, w in self.weight_dict.items():
            all_levels[l] = w * self.KL_divergence(
                posterior_mus[l], posterior_sigmas[l], prior_mus[l], prior_sigmas[l]
            )
            kl_loss += all_levels[l]  # type: ignore[assignment]

        if return_all_levels:
            return kl_loss, all_levels
        return kl_loss


class HierarchicalReconstructionLoss(nn.Module):
    reconstruction_loss: ReconstructionLossType

    def __init__(
        self, reconstruction_loss: ReconstructionLossType, weight_dict: dict[int, float]
    ) -> None:
        super().__init__()

        self.reconstruction_loss = reconstruction_loss
        self.weight_dict = weight_dict

    def forward(
        self,
        inputs: OutputDictType,
        target: torch.Tensor,
        return_all_levels: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[int, torch.Tensor]]]:
        assert self.weight_dict.keys() == inputs.keys()

        recon_loss: torch.Tensor = 0.0  # type: ignore[assignment]
        all_levels = {}

        for l, w in self.weight_dict.items():
            all_levels[l] = w * self.reconstruction_loss(input=inputs[l], target=target)
            recon_loss += all_levels[l]
        if return_all_levels:
            return recon_loss, all_levels
        return recon_loss
