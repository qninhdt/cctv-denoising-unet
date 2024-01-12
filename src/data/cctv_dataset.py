from typing import List
from pathlib import Path

import torch
import numpy as np
from random import Random
from torchvision.transforms import v2 as T
from torch.utils.data import Dataset
import cv2


class CCTVDataset(Dataset):
    def __init__(
        self, unclean_data_dir: str, clean_data_dir: str, transform: torch.nn.Module
    ) -> None:
        super().__init__()

        self.unclean_data_dir = Path(unclean_data_dir)
        self.clean_data_dir = Path(clean_data_dir)
        self.transform = transform
        self.image_paths = self.load_dataset()

        for _ in range(4):
            Random(1).shuffle(self.image_paths)

    def load_dataset(self) -> List[str]:
        files = (self.unclean_data_dir / "images").glob("*.jpg")
        return [file.name for file in files]

    def __getitem__(self, idx: int) -> dict:
        path = self.image_paths[idx]

        clean_image = cv2.imread(str(self.clean_data_dir / "images" / path))
        unclean_image = cv2.imread(str(self.unclean_data_dir / "images" / path))

        if self.transform:
            transformed = self.transform(clean_image=clean_image, image=unclean_image)
            clean_image = transformed["clean_image"]
            unclean_image = transformed["image"]

        sample = {
            "clean_image": clean_image,
            "unclean_image": unclean_image,
        }

        return sample

    def __len__(self) -> int:
        return len(self.image_paths)
