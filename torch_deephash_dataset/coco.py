import json
import os
import shutil
from functools import lru_cache

import numpy as np

from .base import BaseDeepHashDataset
from .utils.download import download_file_list
from .utils.log import logger

_DOWNLOAD_BASE_URL = 'http://images.cocodataset.org'
_IMAGE_ARCHIVE_DIR_URL = f'{_DOWNLOAD_BASE_URL}/zips'
_ANNOTATION_ARCHIVE_DIR_URL = f'{_DOWNLOAD_BASE_URL}/annotations'

_IMAGE_ARCHIVES = ['train2014.zip', 'val2014.zip']
_ANNOTATION_ARCHIVE = 'annotations_trainval2014.zip'

_ARCHIVE_SAVE_DIR = 'archives'
_EXTRACTED_ANNOTATION_SAVE_PATHS = [
    'annotations/instances_train2014.json',
    'annotations/instances_val2014.json',
]


class COCODataset(BaseDeepHashDataset):
    train_size = 5000
    test_size = 1000
    k_classes = 91

    @property
    def archive_save_dir(self):
        return self.get_full_path(_ARCHIVE_SAVE_DIR)

    @property
    def annotation_save_paths(self):
        return [
            self.get_full_path(annotation_save_path)
            for annotation_save_path in _EXTRACTED_ANNOTATION_SAVE_PATHS
        ]

    def download_dataset(self):
        os.makedirs(self.archive_save_dir, exist_ok=True)

        logger.info('Donwloading dataset...')
        all_archive_urls = [
            *(f'{_IMAGE_ARCHIVE_DIR_URL}/{archive}' for archive in _IMAGE_ARCHIVES),
            f'{_ANNOTATION_ARCHIVE_DIR_URL}/{_ANNOTATION_ARCHIVE}',
        ]

        download_file_list(all_archive_urls, self.archive_save_dir)

        logger.info('Extracting archives')
        for archive in _IMAGE_ARCHIVES + [_ANNOTATION_ARCHIVE]:
            shutil.unpack_archive(f'{self.archive_save_dir}/{archive}', self.root)

    @lru_cache(maxsize=None)
    def process_annotations_json(self, annotation_path):
        logger.info(f'Start processing {annotation_path}')
        with open(annotation_path) as file:
            annotation_json = json.load(file)

        images = {
            image['id']: {
                'path': image['coco_url'].replace('http://images.cocodataset.org/', ''),
                'label_vals': set(),
            }
            for image in annotation_json['images']
        }
        for annotation in annotation_json['annotations']:
            images[annotation['image_id']]['label_vals'].add(annotation['category_id'])

        logger.info(f'Finish processing {annotation_path}')
        return [(image['path'], list(image['label_vals'])) for image in images.values()]

    def _encode_one_hot(self, label_vals):
        return np.sum(np.eye(self.k_classes, dtype=np.int8)[label_vals], axis=0)

    def get_split_iterator(self, split: str):
        all_paths = []
        all_labels = []

        for annotation_path in self.annotation_save_paths:
            for path, label_vals in self.process_annotations_json(annotation_path):
                all_paths.append(path)
                all_labels.append(self._encode_one_hot(label_vals))

        if split == 'train':
            all_paths = all_paths[: self.train_size]
            all_labels = all_labels[: self.train_size]
        elif split == 'test':
            all_paths = all_paths[self.train_size : self.train_size + self.test_size]
            all_labels = all_labels[self.train_size : self.train_size + self.test_size]
        else:
            all_paths = all_paths[self.train_size + self.test_size :]
            all_labels = all_labels[self.train_size + self.test_size :]

        for path, label in zip(all_paths, all_labels):
            yield path, label

    def is_dataset_existed(self):
        for annotation_path in self.annotation_save_paths:
            if not os.path.exists(annotation_path):
                return False

        for split in self.allowed_dataset_splits:
            for img_path, _ in self.get_split_iterator(split):
                img_path = os.path.join(self.root, img_path)
                if not os.path.exists(img_path):
                    return False
        return True
