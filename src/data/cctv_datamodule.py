from typing import Any, Dict, Optional, Tuple, List

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from utils.dataset import ApplyTransform
from utils.transform import SquarePad

from .cctv_dataset import CCTVDataset

IMAGE_SIZE = 640


class CCTVDataModule(LightningDataModule):
    def __init__(
        self,
        clean_data_dir: str,
        unclean_data_dir: str,
        train_val_test_split: Tuple[int, int, int],
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.train_transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomResizedCrop(
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    scale=(0.5, 1.5),
                    ratio=(0.5, 1.5),
                ),
                A.LongestMaxSize(max_size=IMAGE_SIZE, interpolation=1),
                A.PadIfNeeded(
                    min_height=IMAGE_SIZE,
                    min_width=IMAGE_SIZE,
                    border_mode=0,
                    value=(0, 0, 0),
                ),
                A.Normalize(),
                ToTensorV2(),
            ],
            additional_targets={
                "clean_image": "image",
            },
        )

        self.transforms = A.Compose(
            [
                A.LongestMaxSize(max_size=IMAGE_SIZE, interpolation=1),
                A.PadIfNeeded(
                    min_height=IMAGE_SIZE,
                    min_width=IMAGE_SIZE,
                    border_mode=0,
                    value=(0, 0, 0),
                ),
                A.Normalize(),
                ToTensorV2(),
            ],
            additional_targets={
                "clean_image": "image",
            },
        )

        self.dataset: Optional[CCTVDataset] = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return self.dataseA.num_classes()

    def prepare_data(self) -> None:
        return

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.dataset = CCTVDataset(
                clean_data_dir=self.hparams.clean_data_dir,
                unclean_data_dir=self.hparams.unclean_data_dir,
                transform=self.transforms,
            )

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=self.dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(1234),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        return self._create_dataloader(self.data_train, self.batch_size_per_device)

    def val_dataloader(self) -> DataLoader[Any]:
        return self._create_dataloader(self.data_val, self.batch_size_per_device)

    def test_dataloader(self) -> DataLoader[Any]:
        return self._create_dataloader(self.data_test, self.batch_size_per_device)

    def _create_dataloader(self, dataset: Dataset, batch_size: int) -> DataLoader[Any]:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, List[dict]]:
        unclean_images = torch.stack([sample["unclean_image"] for sample in batch])
        clean_images = torch.stack([sample["clean_image"] for sample in batch])

        return unclean_images, clean_images
