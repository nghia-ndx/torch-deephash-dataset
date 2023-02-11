import os
import shutil
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset

from .utils.log import logger


class BaseDeepHashDataset(VisionDataset, ABC):
    allowed_dataset_splits = ['train', 'test', 'db']

    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        force_download: bool = False,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)

        if split not in self.allowed_dataset_splits:
            raise ValueError(
                f'Dataset split must be one of {self.allowed_dataset_splits}'
            )
        self.split = split

        if force_download and os.path.exists(self.root):
            logger.warn(f'Directory {self.root} will be erased.')
            shutil.rmtree(self.root)

        os.makedirs(self.root, exist_ok=True)
        if force_download or not self.is_dataset_existed():
            self.download_dataset()

        self.img_paths = []
        self.labels = []

        for img_path, label in self.get_split_iterator(split):
            self.img_paths.append(self.get_full_path(img_path))
            self.labels.append(np.array(label, dtype=np.int8))

    def get_full_path(self, path):
        return os.path.join(self.root, path)

    @abstractmethod
    def download_dataset(self):
        pass

    @abstractmethod
    def is_dataset_existed(self):
        pass

    @abstractmethod
    def get_split_iterator(self, split: str):
        pass

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        img = self.transform(img) if self.transform else img
        label = self.labels[index]
        label = self.target_transform(label) if self.target_transform else label
        return img, label

    def __len__(self):
        return len(self.img_paths)
