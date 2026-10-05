import torch

from torch.utils.data import random_split, DataLoader, WeightedRandomSampler

from torchvision.datasets import ImageFolder
from torchvision.transforms import v2, RandAugment

import lightning as L


class ImageClassificationDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.tansform = v2.Compose(
            [
                RandAugment(2, 9),
                v2.Resize((224, 224)),
                v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.num_workers = 4

    def setup(self, stage=None):
        self.full_dataset = ImageFolder(self.data_dir, transform=self.tansform)

        # print folder - label mapping
        print(self.full_dataset.class_to_idx)

        # split using random_split
        train_size = int(0.8 * len(self.full_dataset))
        val_size = int(0.1 * len(self.full_dataset))
        test_size = len(self.full_dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=WeightedRandomSampler(
                [1.0 / len(self.train_dataset) for _ in range(len(self.train_dataset))],
                len(self.train_dataset),
            ),
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
