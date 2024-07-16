import torch
from torchvision.utils import make_grid, draw_segmentation_masks  # type: ignore[import]
import math
from typing import Optional
from abc import ABC, abstractmethod
import pytorch_lightning as pl
import torch.nn.functional as F

from src.losses import (
    HierarchicalKLLoss,
    HierarchicalReconstructionLoss,
    multinoulli_loss_2d,
    KL_two_gauss_with_diag_cov,
)
from src.custom_types import SamplerType, OutputDictType
from src.network_blocks import gauss_sampler, TemperatureScaling
from src.base import AbstractPrior, AbstractPosterior, AbstractLikelihood
from src.metrics import per_label_dice, generalised_energy_distance
from utils import convert_to_onehot, find_onehot_dimension, harden_softmax_outputs, compute_ece_for_segmentation
from torchmetrics.classification import MulticlassCalibrationError


@torch.no_grad()
def prepare_log_image(
    labels: torch.Tensor, num_classes: int, background: Optional[torch.Tensor] = None
):
    # Normalise background to 0, 255
    if background is not None:
        min_vals = background.view(background.size(0), -1).min(dim=1, keepdim=True)[0]
        max_vals = background.view(background.size(0), -1).max(dim=1, keepdim=True)[0]
        background = (
            255
            * (background - min_vals.view(background.size(0), 1, 1, 1))
            / (max_vals - min_vals).view(background.size(0), 1, 1, 1)
        )
        background = background.to(dtype=torch.uint8)

    if find_onehot_dimension(labels) is None:
        labels = convert_to_onehot(labels, num_classes=num_classes)

    bs = labels.shape[0]
    nrow = int(math.sqrt(bs))
    labels_g = make_grid(labels, nrow=nrow)
    if background is not None:
        background_g = make_grid(background, nrow=nrow)
    else:
        background_g = torch.zeros_like(labels_g)

    background_g = background_g.cpu()
    labels_g = labels_g.to(torch.bool).cpu()

    log_image = draw_segmentation_masks(background_g, labels_g, alpha=0.6)
    return log_image


class AbstractHierarchicalProbabilisticModel(ABC, pl.LightningModule):
    """
    This is the base class for all conditional hierarchical models. This
    can be extended also for other use cases then segmentation.
    """

    prior: AbstractPrior
    posterior: AbstractPosterior
    likelihood: AbstractLikelihood

    hierarchical_kl_loss: HierarchicalKLLoss
    hierarchical_recon_loss: HierarchicalReconstructionLoss

    total_levels: int
    latent_levels: int
    num_classes: int

    @abstractmethod
    def _aggregate_levels(self, level_outputs: OutputDictType) -> OutputDictType: ...

    def reconstruct_from_z(
        self, z: OutputDictType, x: Optional[torch.Tensor] = None
    ) -> OutputDictType:
        level_outputs = self.likelihood(z, x)
        return self._aggregate_levels(level_outputs)

    def predict_output_samples(self, x: torch.Tensor, N: int = 1) -> torch.Tensor:
        bs = x.shape[0]
        xb = torch.vstack([x for _ in range(N)])
        _, _, z = self.prior(xb)
        agg = self.reconstruct_from_z(z, xb)
        yb_hat = agg[0]
        yb_hat = yb_hat.view([N, bs] + [*yb_hat.shape][1:])
        return yb_hat.transpose(0, 1)  # B x N x spatial_dims

    def predict(self, x: torch.Tensor, N: int = 1) -> torch.Tensor:
        return torch.mean(self.predict_output_samples(x, N), dim=1)

    def posterior_samples(
        self, x: torch.Tensor, y: torch.Tensor, N: int = 1
    ) -> OutputDictType:
        bs = x.shape[0]
        xb = torch.vstack([x for _ in range(N)])
        yb = torch.vstack([y for _ in range(N)])
        _, _, z = self.posterior(xb, yb)

        return {l: zl.view([N, bs] + [*zl.shape][1:]) for l, zl in z.items()}

    def posterior_outputs(
        self, x: torch.Tensor, y: torch.Tensor, N: int = 1
    ) -> torch.Tensor:
        bs = x.shape[0]
        xb = torch.vstack([x for _ in range(N)])
        yb = torch.vstack([y for _ in range(N)])
        _, _, z = self.posterior(xb, yb)
        agg = self.reconstruct_from_z(z)
        yb_hat = agg[0]
        yb_hat = yb_hat.view([N, bs] + [*yb_hat.shape][1:])
        return yb_hat.transpose(0, 1)  # B x N x spatial_dims

    def output_level_predictions(self, x: torch.Tensor):
        _, _, z = self.prior(x)
        return self.likelihood(z)

    @staticmethod
    def samples_to_list(yb: torch.Tensor) -> list[torch.Tensor]:
        return [yb[i] for i in range(yb.shape[0])]


class AbstractHierarchicalProbabilisticSegmentationModel(
    AbstractHierarchicalProbabilisticModel
):
    """
    This is the base class for all hierarchical probabilistic segmentation models such as
    the probabilistic U-Net and PHIseg
    """

    def _log_labels_batch_on_background(
        self, labels: torch.Tensor, name: str, background: Optional[torch.Tensor] = None
    ):
        log_image = prepare_log_image(
            labels=labels, num_classes=self.num_classes, background=background
        )
        self.logger.experiment.add_image(name, log_image, self.trainer.global_step)  # type: ignore[union-attr]

    def _aggregate_levels(
        self,
        level_outputs: OutputDictType,
    ) -> OutputDictType:
        assert list(level_outputs.keys()) == list(range(self.latent_levels))

        combined_outputs = {
            self.latent_levels - 1: level_outputs[self.latent_levels - 1]
        }

        for l in reversed(range(self.latent_levels - 1)):
            combined_outputs[l] = combined_outputs[l + 1] + level_outputs[l]

        return combined_outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class PHISeg(AbstractHierarchicalProbabilisticSegmentationModel):
    """
    The actual PHISeg is now just defined by the network architecture and
    the losses
    """

    latent_levels: int
    total_levels: int

    def __init__(
        self,
        total_levels: int,
        latent_levels: int,
        zdim: int,
        num_classes: int,
        beta: float = 1.0,
        input_channels: int = 3,  # TODO: Make input channels variable
    ) -> None:
        super().__init__()

        from src.components.phiseg import PHISegPrior, PHISegPosterior, PHISegLikelihood

        # Make all arguments to init accessible via self.hparams (e.g.
        # self.hparams.total_levels) and save hyperparameters to checkpoint
        # and potentially logger
        self.save_hyperparameters()

        sampler: SamplerType = gauss_sampler

        # Could access these vars via self.hparams but make member of self for less clutter
        self.latent_levels = latent_levels
        self.total_levels = total_levels
        self.num_classes = num_classes
        self.beta = beta

        self.prior = PHISegPrior(
            sampler=sampler,
            total_levels=total_levels,
            latent_levels=latent_levels,
            zdim=zdim,
            input_channels=input_channels,
        )
        self.posterior = PHISegPosterior(
            sampler=sampler,
            total_levels=total_levels,
            latent_levels=latent_levels,
            zdim=zdim,
            num_classes=num_classes,
            input_channels=input_channels,
        )
        self.likelihood = PHISegLikelihood(
            total_levels=total_levels,
            latent_levels=latent_levels,
            zdim=zdim,
            num_classes=num_classes,
        )

        kl_weight_dict = {l: 4.0**l for l in range(latent_levels)}
        recon_weight_dict = {l: 1.0 for l in range(latent_levels)}

        self.hierarchical_kl_loss = HierarchicalKLLoss(
            KL_divergence=KL_two_gauss_with_diag_cov, weight_dict=kl_weight_dict
        )

        self.hierarchical_recon_loss = HierarchicalReconstructionLoss(
            reconstruction_loss=multinoulli_loss_2d, weight_dict=recon_weight_dict
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # For now the default behaviour is just a single forward pass, i.e. single sample
        # through the prior network likelihood network logic.

        _, _, zs = self.prior(x)
        ys = self.likelihood(zs)
        return self._aggregate_levels(ys)[0]

    def training_step(self, batch, batch_idx):
        x, y = batch
        posterior_mus, posterior_sigmas, posterior_samples = self.posterior(x, y)
        prior_mus, prior_sigmas, _ = self.prior(x, posterior_samples=posterior_samples)
        y_hat = self.reconstruct_from_z(posterior_samples)

        kl_loss, kl_loss_levels = self.hierarchical_kl_loss(
            prior_mus,
            prior_sigmas,
            posterior_mus,
            posterior_sigmas,
            return_all_levels=True,
        )

        reconstruction_loss, reconstruction_loss_levels = self.hierarchical_recon_loss(
            y_hat, y, return_all_levels=True
        )
        total_loss = self.beta * kl_loss + reconstruction_loss

        self.log("train/kl_loss", kl_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/reconstruction_loss",
            reconstruction_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True
        )

        for level, level_loss in reconstruction_loss_levels.items():
            self.log(
                f"train_levels/recon loss level {level}",
                level_loss,
                on_step=False,
                on_epoch=True,
            )
        for level, level_loss in kl_loss_levels.items():
            self.log(
                f"train_levels/kl loss level {level}",
                level_loss,
                on_step=False,
                on_epoch=True,
            )

        for level in kl_loss_levels.keys():
            self.log(
                f"train_distribution_levels/mean_prior_mu_{level}",
                torch.mean(prior_mus[level]),
            )
            self.log(
                f"train_distribution_levels/mean_prior_sigma_{level}",
                torch.mean(prior_sigmas[level]),
            )
            self.log(
                f"train_distribution_levels/mean_posterior_mu_{level}",
                torch.mean(posterior_mus[level]),
            )
            self.log(
                f"train_distribution_levels/mean_posterior_sigma_{level}",
                torch.mean(posterior_sigmas[level]),
            )

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        posterior_mus, posterior_sigmas, posterior_samples = self.posterior(x, y)
        prior_mus, prior_sigmas, _ = self.prior(x)
        y_hat = self.reconstruct_from_z(posterior_samples)

        kl_loss, kl_loss_levels = self.hierarchical_kl_loss(
            prior_mus,
            prior_sigmas,
            posterior_mus,
            posterior_sigmas,
            return_all_levels=True,
        )

        reconstruction_loss, reconstruction_loss_levels = self.hierarchical_recon_loss(
            y_hat, y, return_all_levels=True
        )
        total_loss = self.beta * kl_loss + reconstruction_loss

        self.log("val/kl_loss", kl_loss, on_epoch=True)
        self.log("val/reconstruction_loss", reconstruction_loss, on_epoch=True)
        self.log("val/total_loss", total_loss, on_epoch=True)

        for level, level_loss in reconstruction_loss_levels.items():
            self.log(f"val_levels/recon loss level {level}", level_loss, on_epoch=True)
        for level, level_loss in kl_loss_levels.items():
            self.log(f"val_levels/kl loss level {level}", level_loss, on_epoch=True)

        for level in kl_loss_levels.keys():
            self.log(
                f"val_distribution_levels/mean_prior_mu_{level}",
                torch.mean(prior_mus[level]),
            )
            self.log(
                f"val_distribution_levels/mean_prior_sigma_{level}",
                torch.mean(prior_sigmas[level]),
            )
            self.log(
                f"val_distribution_levels/mean_posterior_mu_{level}",
                torch.mean(posterior_mus[level]),
            )
            self.log(
                f"val_distribution_levels/mean_posterior_sigma_{level}",
                torch.mean(posterior_sigmas[level]),
            )

        # Get network predictions
        y_pred_samples = self.predict_output_samples(x, N=10)
        y_pred = torch.mean(y_pred_samples, dim=1)

        # Log Dice
        y_pred = harden_softmax_outputs(y_pred, dim=1)
        y = convert_to_onehot(y, num_classes=self.num_classes)
        dice = per_label_dice(
            input=y_pred,
            target=y,
            input_is_batch=True,
        )
        self.log("val/dice", torch.mean(dice[1:]), on_epoch=True)

        # Remember validation images for visualisation later
        self.val_x = x
        self.val_y = y
        self.val_y_hat = y_hat
        self.val_y_pred = y_pred

        return total_loss

    def on_validation_epoch_end(self):
        # Log outputs
        x, y, y_hat, y_pred = self.val_x, self.val_y, self.val_y_hat, self.val_y_pred
        y_recon = torch.argmax(y_hat[0], dim=1)
        y_pred = torch.argmax(y_pred, dim=1)
        y = torch.argmax(y, dim=1)

        for name, lbl in zip(
            ["val/GT labels", "val/reconstruction", "val/prediction"],
            [y, y_recon, y_pred],
        ):
            self._log_labels_batch_on_background(lbl, name, x)

        for l in range(self.latent_levels):
            self._log_labels_batch_on_background(
                torch.argmax(y_hat[l], dim=1), f"val_levels/recon level {l}", x
            )

    def test_step(self, batch, batch_idx):
        x, y = batch

        posterior_mus, posterior_sigmas, posterior_samples = self.posterior(x, y)
        prior_mus, prior_sigmas, _ = self.prior(x)
        y_hat = self.reconstruct_from_z(posterior_samples)

        kl_loss, kl_loss_levels = self.hierarchical_kl_loss(
            prior_mus,
            prior_sigmas,
            posterior_mus,
            posterior_sigmas,
            return_all_levels=True,
        )

        reconstruction_loss, reconstruction_loss_levels = self.hierarchical_recon_loss(
            y_hat, y, return_all_levels=True
        )
        total_loss = self.beta * kl_loss + reconstruction_loss

        self.log("val/total_loss", total_loss, on_epoch=True)

        # Get network predictions
        y_pred_samples = self.predict_output_samples(x, N=100)
        y_pred = torch.mean(y_pred_samples, dim=1)

        ECE = MulticlassCalibrationError(num_classes=self.num_classes, n_bins=15, norm='l1')
        ece = ECE(y_pred, y)
        self.log("test/ece", ece, on_epoch=True)

        # Log Dice
        y_pred = harden_softmax_outputs(y_pred, dim=1)
        y = convert_to_onehot(y, num_classes=self.num_classes)
        dice = per_label_dice(
            input=y_pred,
            target=y,
            input_is_batch=True,
        )

        self.log("test/dice", torch.mean(dice[1:]), on_epoch=True)

        return (
            kl_loss,
            kl_loss_levels,
            reconstruction_loss,
            reconstruction_loss_levels,
            total_loss,
            torch.mean(dice[1:]),
        )


class ProbUNet(AbstractHierarchicalProbabilisticSegmentationModel):
    """
    The actual PHISeg is now just defined by the network architecture and
    the losses
    """

    total_levels: int

    def __init__(
        self,
        total_levels: int,
        zdim: int,
        num_classes: int,
        beta: float = 1.0,
        input_channels: int = 3,
    ) -> None:
        super().__init__()

        from src.components.probabilistic_unet import (
            ProbUNetPrior,
            ProbUNetPosterior,
            ProbUNetLikelihood,
        )

        # Make all arguments to init accessible via self.hparams (e.g.
        # self.hparams.total_levels) and save hyperparameters to checkpoint
        # and potentially logger
        self.save_hyperparameters()

        sampler: SamplerType = gauss_sampler

        # Could access these vars via self.hparams but make member of self for less clutter
        self.total_levels = total_levels
        self.num_classes = num_classes
        self.beta = beta
        self.latent_levels = 1

        self.prior = ProbUNetPrior(
            sampler=sampler,
            total_levels=total_levels,
            zdim=zdim,
            input_channels=input_channels,
        )
        self.posterior = ProbUNetPosterior(
            sampler=sampler,
            total_levels=total_levels,
            zdim=zdim,
            num_classes=num_classes,
            input_channels=input_channels,
        )
        self.likelihood = ProbUNetLikelihood(
            total_levels=total_levels,
            zdim=zdim,
            num_classes=num_classes,
        )

        self.hierarchical_kl_loss = HierarchicalKLLoss(
            KL_divergence=KL_two_gauss_with_diag_cov, weight_dict={0: 1.0}
        )
        self.hierarchical_recon_loss = HierarchicalReconstructionLoss(
            reconstruction_loss=multinoulli_loss_2d, weight_dict={0: 1.0}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # For now the default behaviour is just a single forward pass, i.e. single sample
        # through the prior network likelihood network logic.

        _, _, zs = self.prior(x)
        ys = self.likelihood(zs, x)
        return self._aggregate_levels(ys)[0]

    def training_step(self, batch, batch_idx):
        x, y = batch

        posterior_mus, posterior_sigmas, posterior_samples = self.posterior(x, y)
        prior_mus, prior_sigmas, _ = self.prior(x)
        y_hat = self.reconstruct_from_z(posterior_samples, x)

        kl_loss = self.hierarchical_kl_loss(
            prior_mus,
            prior_sigmas,
            posterior_mus,
            posterior_sigmas,
        )

        reconstruction_loss = self.hierarchical_recon_loss(
            y_hat,
            y,
        )
        total_loss = self.beta * kl_loss + reconstruction_loss

        self.log("train/kl_loss", kl_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/reconstruction_loss",
            reconstruction_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        posterior_mus, posterior_sigmas, posterior_samples = self.posterior(x, y)
        prior_mus, prior_sigmas, _ = self.prior(x)
        y_hat = self.reconstruct_from_z(posterior_samples, x)

        kl_loss = self.hierarchical_kl_loss(
            prior_mus,
            prior_sigmas,
            posterior_mus,
            posterior_sigmas,
        )

        reconstruction_loss = self.hierarchical_recon_loss(y_hat, y)
        total_loss = self.beta * kl_loss + reconstruction_loss

        self.log("val/kl_loss", kl_loss, on_epoch=True)
        self.log("val/reconstruction_loss", reconstruction_loss, on_epoch=True)
        self.log("val/total_loss", total_loss, on_epoch=True)

        # Get network predictions
        y_pred_samples = self.predict_output_samples(x, N=10)
        y_pred = torch.mean(y_pred_samples, dim=1)

        # Log Dice
        y_pred = harden_softmax_outputs(y_pred, dim=1)
        y = convert_to_onehot(y, num_classes=self.num_classes)
        dice = per_label_dice(
            input=y_pred,
            target=y,
            input_is_batch=True,
        )
        self.log("val/dice", torch.mean(dice[1:]), on_epoch=True)

        # Remember validation images for visualisation later
        self.val_x = x
        self.val_y = y
        self.val_y_hat = y_hat
        self.val_y_pred = y_pred

        return total_loss

    def on_validation_epoch_end(self):
        # Log outputs
        x, y, y_hat, y_pred = self.val_x, self.val_y, self.val_y_hat, self.val_y_pred
        y_recon = torch.argmax(y_hat[0], dim=1)
        y_pred = torch.argmax(y_pred, dim=1)
        y = torch.argmax(y, dim=1)

        for name, lbl in zip(
            ["val/GT labels", "val/reconstruction", "val/prediction"],
            [y, y_recon, y_pred],
        ):
            self._log_labels_batch_on_background(lbl, name, x)

    def test_step(self, batch, batch_idx):
        x, y = batch

        posterior_mus, posterior_sigmas, posterior_samples = self.posterior(x, y)
        prior_mus, prior_sigmas, _ = self.prior(x)
        y_hat = self.reconstruct_from_z(posterior_samples, x)

        kl_loss, kl_loss_levels = self.hierarchical_kl_loss(
            prior_mus,
            prior_sigmas,
            posterior_mus,
            posterior_sigmas,
            return_all_levels=True,
        )

        reconstruction_loss, reconstruction_loss_levels = self.hierarchical_recon_loss(
            y_hat, y, return_all_levels=True
        )
        total_loss = self.beta * kl_loss + reconstruction_loss

        self.log("val/total_loss", total_loss, on_epoch=True)

        # Get network predictions
        y_pred_samples = self.predict_output_samples(x, N=100)
        y_pred = torch.mean(y_pred_samples, dim=1)

        ECE = MulticlassCalibrationError(num_classes=self.num_classes, n_bins=15, norm='l1')
        ece = ECE(y_pred, y)
        self.log("test/ece", ece, on_epoch=True)

        # Log Dice
        y_pred = harden_softmax_outputs(y_pred, dim=1)
        y = convert_to_onehot(y, num_classes=self.num_classes)
        dice = per_label_dice(
            input=y_pred,
            target=y,
            input_is_batch=True,
        )

        self.log("test/dice", torch.mean(dice[1:]), on_epoch=True)

        return (
            kl_loss,
            kl_loss_levels,
            reconstruction_loss,
            reconstruction_loss_levels,
            total_loss,
            torch.mean(dice[1:]),
        )


class UNet(pl.LightningModule):
    def __init__(
        self,
        total_levels: int,
        num_classes: int,
        input_channels: int = 3,
    ) -> None:
        super().__init__()

        from src.components.unet import UNet

        # Make all arguments to init accessible via self.hparams (e.g.
        # self.hparams.total_levels) and save hyperparameters to checkpoint
        # and potentially logger
        self.save_hyperparameters()

        # Could access these vars via self.hparams but make member of self for less clutter
        self.total_levels = total_levels
        self.num_classes = num_classes

        self.unet = UNet(
            num_classes=num_classes,
            total_levels=total_levels,
            input_channels=input_channels,
        )

        self.temperature_scaling = TemperatureScaling()

        self.loss = multinoulli_loss_2d  # TODO change to dice + xent loss

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.predict(x)

        loss = self.loss(y_hat, y)

        self.log(
            "train/total_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.predict(x)

        loss = self.loss(y_hat, y)

        self.log("val/total_loss", loss, on_epoch=True)

        # Log Dice
        y_pred = harden_softmax_outputs(y_hat, dim=1)
        y = convert_to_onehot(y, num_classes=self.num_classes)
        dice = per_label_dice(
            input=y_pred,
            target=y,
            input_is_batch=True,
        )
        self.log("val/dice", torch.mean(dice[1:]), on_epoch=True)

        # Remember validation images for visualisation later
        self.val_x = x
        self.val_y = y
        self.val_y_hat = y_hat
        self.val_y_pred = y_pred

        return loss

    def on_validation_epoch_end(self):
        # Log outputs
        x, y, y_hat, y_pred = self.val_x, self.val_y, self.val_y_hat, self.val_y_pred
        y_pred = torch.argmax(y_pred, dim=1)
        y = torch.argmax(y, dim=1)

        for name, lbl in zip(
            ["val/GT labels", "val/prediction"],
            [y, y_pred],
        ):
            self._log_labels_batch_on_background(lbl, name, x)

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.predict(x)

        loss = self.loss(y_hat, y)

        self.log("val/total_loss", loss, on_epoch=True)

        ECE = MulticlassCalibrationError(num_classes=self.num_classes, n_bins=15, norm='l1')
        ece = ECE(y_hat, y)
        self.log("test/ece", ece, on_epoch=True)

        # Log Dice
        y_pred = harden_softmax_outputs(y_hat, dim=1)
        y = convert_to_onehot(y, num_classes=self.num_classes)
        dice = per_label_dice(
            input=y_pred,
            target=y,
            input_is_batch=True,
        )
        self.log("test/dice", torch.mean(dice[1:]), on_epoch=True)

        return loss

    def _log_labels_batch_on_background(
        self, labels: torch.Tensor, name: str, background: Optional[torch.Tensor] = None
    ):

        log_image = prepare_log_image(
            labels=labels, num_classes=self.num_classes, background=background
        )
        self.logger.experiment.add_image(name, log_image, self.trainer.global_step)  # type: ignore[union-attr]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class UNetMCDropout(UNet):
    def __init__(
        self,
        total_levels: int,
        num_classes: int,
        input_channels: int = 3,
    ) -> None:
        super().__init__(
            total_levels=total_levels,
            num_classes=num_classes,
            input_channels=input_channels,
        )

        # Modify the type of UNet that is used.
        from src.components.unet import UNet

        self.dropout_rate = 0.2
        self.unet = UNet(
            num_classes=num_classes,
            total_levels=total_levels,
            input_channels=input_channels,
            permanent_dropout=self.dropout_rate,
        )

    def predict_output_samples(self, x: torch.Tensor, N: int = 1) -> torch.Tensor:
        bs = x.shape[0]
        xb = torch.vstack([x for _ in range(N)])
        yb_hat = self.predict(xb)
        yb_hat = yb_hat.view([N, bs] + [*yb_hat.shape][1:])
        return yb_hat.transpose(0, 1)  # B x N x spatial_dims

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.predict(x)
        loss = self.loss(y_hat, y)

        self.log("val/total_loss", loss, on_epoch=True)

        # Get network predictions
        y_pred_samples = self.predict_output_samples(x, N=10)
        y_pred = torch.mean(y_pred_samples, dim=1)

        # Log Dice
        y_pred = harden_softmax_outputs(y_pred, dim=1)
        y = convert_to_onehot(y, num_classes=self.num_classes)
        dice = per_label_dice(
            input=y_pred,
            target=y,
            input_is_batch=True,
        )
        self.log("val/dice", torch.mean(dice[1:]), on_epoch=True)

        # Remember validation images for visualisation later
        self.val_x = x
        self.val_y = y
        self.val_y_hat = y_hat
        self.val_y_pred = y_pred

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.predict(x)

        loss = self.loss(y_hat, y)

        self.log("val/total_loss", loss, on_epoch=True)

        ECE = MulticlassCalibrationError(num_classes=self.num_classes, n_bins=15, norm='l1')
        ece = ECE(y_hat, y)
        self.log("test/ece", ece, on_epoch=True)

        # Log Dice
        y_pred = harden_softmax_outputs(y_hat, dim=1)
        y = convert_to_onehot(y, num_classes=self.num_classes)
        dice = per_label_dice(
            input=y_pred,
            target=y,
            input_is_batch=True,
        )
        self.log("test/dice", torch.mean(dice[1:]), on_epoch=True)

        return loss
