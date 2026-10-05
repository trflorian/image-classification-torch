import torch

from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

from torchvision.datasets import ImageFolder
from torchvision.transforms import v2, RandAugment

import lightning as L


class ImageClassificationDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.train_transform = v2.Compose(
            [
                RandAugment(2, 9),
                v2.Resize((224, 224)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.eval_transform = v2.Compose(
            [
                v2.Resize((224, 224)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.num_workers = 4

    def setup(self, stage=None):
        train_dataset = ImageFolder(self.data_dir, transform=self.train_transform)
        eval_dataset = ImageFolder(self.data_dir, transform=self.eval_transform)

        # print folder - label mapping
        print(train_dataset.class_to_idx)

        # Share split indices while giving validation/test deterministic transforms.
        dataset_size = len(train_dataset)
        train_size = int(0.8 * dataset_size)
        val_size = int(0.1 * dataset_size)
        indices = torch.randperm(dataset_size).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        self.train_dataset = Subset(train_dataset, train_indices)
        self.val_dataset = Subset(eval_dataset, val_indices)
        self.test_dataset = Subset(eval_dataset, test_indices)

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
