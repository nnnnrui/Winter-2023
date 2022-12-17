import os.path as osp
from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Distance
from torch_geometric.transforms import Distance as SparseDistance
from torch_geometric.data import Data

from src.datamodule.datasets.kellerdataset import KellerDataset
from src.utils.iterative_stratification import make_multilabel_stratified_split


class KellerDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/keller",
        batch_size: int = 256,
        num_workers: int = 8,
        transform=None,
        **kwargs,
    ):
        super().__init__()
        self.dataset = KellerDataset(data_dir, transform=transform)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

        self.data_train: Optional[Data] = None
        self.data_val: Optional[Data] = None
        self.split = self.get_split()

    @property
    def num_classes(self) -> int:
        return 21

    @property
    def edge_attr_dim(self) -> int:
        if isinstance(self.transform, Distance) or isinstance(
            self.transform, SparseDistance
        ):
            return 5
        else:
            return 4

    def prepare_data(self):
        pass

    def get_split(self):
        if osp.exists(f"{self.data_dir}/processed/split.pt"):
            split = torch.load(f"{self.data_dir}/processed/split.pt")
        else:
            split = make_multilabel_stratified_split(self.dataset)
            torch.save(split, f"{self.data_dir}/processed/split.pt")
        return split

    def setup(self, stage: Optional[str] = None):
        dataset = KellerDataset(self.data_dir, transform=self.transform)
        split = self.get_split(dataset)
        train_idx = split["fold_0"]["train"]
        valid_idx = split["fold_0"]["valid"]
        test_idx = split["test"]
        self.data_train = dataset[train_idx]
        self.data_valid = dataset[valid_idx]
        self.data_test = dataset[test_idx]
    
    def get_cv_splits(self):
        split = []
        for k, v in self.split.items():
            if k.startswith("fold"):
                split.append((v["train"], v["valid"]))

        dataset = self.dataset
        for train_idx, val_idx in split:
            train_dataset = dataset[train_idx]
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=False,
            )

            val_dataset = dataset[val_idx]
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False,
            )

            yield train_loader, val_loader

    def get_test(self):
        dataset = self.dataset
        test_dataset = dataset[self.split["test"]]
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )
        return test_loader

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
