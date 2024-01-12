from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.classification.accuracy import Accuracy
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class UNetModule(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        compile: bool,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["model"])

        self.model = model

        self.criterion = torch.nn.MSELoss()

        self.train_psnr = PeakSignalNoiseRatio()
        self.val_psnr = PeakSignalNoiseRatio()
        self.test_psnr = PeakSignalNoiseRatio()

        self.train_mean_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unclean_images, clean_images = batch

        pred_images = self.forward(unclean_images)

        loss = self.criterion(pred_images, clean_images)

        return loss, pred_images, clean_images

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, pred_images, clean_images = self.model_step(batch)

        # update and log metrics
        self.train_psnr(pred_images, clean_images)
        self.train_mean_loss(loss)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log(
            "train/psnr",
            self.train_psnr.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/mean_loss",
            self.train_mean_loss.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss, pred_images, clean_images = self.model_step(batch)

        # update and log metrics
        self.val_psnr(pred_images, clean_images)

    def on_validation_epoch_end(self) -> None:
        self.log(
            "val/psnr",
            self.val_psnr.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss, pred_images, clean_images = self.model_step(batch)

        # update and log metrics
        self.test_psnr(pred_images, clean_images)

    def on_test_epoch_end(self) -> None:
        self.log(
            "test/psnr",
            self.test_psnr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = Adam(self.parameters(), lr=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=10, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/psnr",
                "interval": "epoch",
                "frequency": 1,
            },
        }
